"""LLaMA Neural Network Layers.

Contains only nn.Module classes:
- RMSNorm: Root Mean Square Layer Normalization
- LlamaAttention: Multi-head attention with RoPE and GQA
- SwiGLU: Gated Linear Unit with SiLU activation
- LlamaBlock: Single transformer block
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

from .config import LlamaConfig
from .kv_cache import KVCache, repeat_kv
from ..utils.rope import apply_rope


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.
    
    Computes in fp32 for numerical stability, then converts back.
    """
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (self.weight * x).to(input_dtype)


class LlamaAttention(nn.Module):
    """Multi-head attention with RoPE and Grouped Query Attention (GQA)."""
    
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.d_model // config.num_heads
        self.d_model = config.d_model
        self.n_rep = self.num_heads // self.num_kv_heads
        
        # Q projection: full size
        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        
        # K/V projections: smaller if using GQA
        kv_dim = self.num_kv_heads * self.head_dim
        self.k_proj = nn.Linear(config.d_model, kv_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, kv_dim, bias=False)
        
        self.o_proj = nn.Linear(config.d_model, config.d_model, bias=False)
    
    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor (B, L, d_model)
            cos, sin: RoPE frequencies from parent model
            mask: Causal attention mask
            kv_cache: Optional KVCache for generation (updated in-place)
        """
        B, L, _ = x.shape
        
        # Project to Q, K, V
        Q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE and handle KV cache
        if kv_cache is not None:
            offset = kv_cache.seq_len
            Q = apply_rope(Q, cos[offset:offset+L], sin[offset:offset+L])
            K = apply_rope(K, cos[offset:offset+L], sin[offset:offset+L])
            K, V = kv_cache.update(K, V)
        else:
            Q = apply_rope(Q, cos[:L], sin[:L])
            K = apply_rope(K, cos[:L], sin[:L])
        
        # Expand K/V heads for GQA
        K = repeat_kv(K, self.n_rep)
        V = repeat_kv(V, self.n_rep)
        
        # Scaled dot-product attention
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        out = attn @ V
        
        out = out.transpose(1, 2).contiguous().view(B, L, self.d_model)
        return self.o_proj(out)


class SwiGLU(nn.Module):
    """SwiGLU activation for feed-forward network."""
    
    def __init__(self, d_model: int, hidden_dim: int):
        super().__init__()
        self.w_gate = nn.Linear(d_model, hidden_dim, bias=False)
        self.w_up = nn.Linear(d_model, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, d_model, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class LlamaBlock(nn.Module):
    """A single LLaMA transformer block."""
    
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.attention = LlamaAttention(config)
        self.feed_forward = SwiGLU(config.d_model, config.hidden_dim)
        self.attention_norm = RMSNorm(config.d_model, config.eps)
        self.ffn_norm = RMSNorm(config.d_model, config.eps)
    
    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> torch.Tensor:
        """KV cache is updated in-place if provided."""
        attn_out = self.attention(self.attention_norm(x), cos, sin, mask, kv_cache)
        x = x + attn_out
        x = x + self.feed_forward(self.ffn_norm(x))
        return x
