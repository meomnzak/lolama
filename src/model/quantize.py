"""Weight-only Quantization for LLaMA.

Implements int8 weight quantization with on-the-fly dequantization.
This reduces memory by ~4x (fp32->int8) or ~2x (fp16->int8).

Usage:
    model = load_model(...)
    quantize_model_int8(model)  # In-place quantization
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuantizedLinear(nn.Module):
    """Linear layer with int8 quantized weights.
    
    Stores weights as int8 + scale, dequantizes during forward pass.
    Uses per-channel (per-output-feature) quantization for better accuracy.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dtype = dtype
        
        # Quantized weights: int8
        self.register_buffer(
            'weight_int8',
            torch.zeros(out_features, in_features, dtype=torch.int8)
        )
        
        # Scale per output channel for dequantization
        self.register_buffer(
            'weight_scale',
            torch.ones(out_features, dtype=dtype)
        )
        
        # Optional bias (not quantized)
        if bias:
            self.register_buffer('bias', torch.zeros(out_features, dtype=dtype))
        else:
            self.bias = None
    
    @staticmethod
    def from_linear(linear: nn.Linear, dtype: torch.dtype | None = None) -> QuantizedLinear:
        """Convert a regular Linear layer to QuantizedLinear.
        
        Args:
            linear: The linear layer to quantize
            dtype: Target dtype for scales and computation (default: weight dtype)
        
        Returns:
            QuantizedLinear with int8 weights
        """
        if dtype is None:
            dtype = linear.weight.dtype
        
        device: torch.device = linear.weight.device
        
        qlayer: QuantizedLinear = QuantizedLinear(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            dtype=dtype,
        )
        
        # Quantize weights per output channel (do on CPU for speed, then move)
        weight: torch.Tensor = linear.weight.data.float().cpu()
        
        # Per-channel absmax quantization
        # Scale each row (output channel) independently
        absmax: torch.Tensor = weight.abs().max(dim=1, keepdim=True)[0].clamp(min=1e-8)
        scale: torch.Tensor = absmax / 127.0  # int8 range is [-127, 127]
        
        # Quantize: weight_int8 = round(weight / scale)
        weight_int8: torch.Tensor = (weight / scale).round().clamp(-127, 127).to(torch.int8)
        
        # Set quantized weights
        qlayer.weight_int8 = weight_int8
        qlayer.weight_scale = scale.squeeze(1).to(dtype)
        
        if linear.bias is not None:
            qlayer.bias = linear.bias.data.to(dtype)
        
        # Move entire module to original device
        return qlayer.to(device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with on-the-fly dequantization."""
        # Use cached dequantized weights if available
        if hasattr(self, '_cached_weight') and self._cached_weight is not None:
            return F.linear(x, self._cached_weight, self.bias)
        
        # Dequantize weights: weight = weight_int8 * scale
        # Use float32 for dequantization to avoid precision issues, then convert
        weight_f32: torch.Tensor = self.weight_int8.float() * self.weight_scale.float().unsqueeze(1)
        weight: torch.Tensor = weight_f32.to(x.dtype)
        
        return F.linear(x, weight, self.bias)
    
    def dequantize_and_cache(self, dtype: torch.dtype = torch.float16) -> None:
        """Pre-dequantize weights and cache them for fast inference.
        
        This trades memory for speed - weights are stored as fp16 during inference.
        """
        weight_f32: torch.Tensor = self.weight_int8.float() * self.weight_scale.float().unsqueeze(1)
        self._cached_weight: torch.Tensor = weight_f32.to(dtype).to(self.weight_int8.device)
    
    def clear_cache(self) -> None:
        """Clear cached dequantized weights to save memory."""
        if hasattr(self, '_cached_weight'):
            self._cached_weight = None
    
    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, quantized=int8'


def quantize_model_int8(model: nn.Module, skip_layers: list[str] | None = None) -> nn.Module:
    """Quantize all Linear layers in a model to int8.
    
    Args:
        model: The model to quantize (modified in-place)
        skip_layers: List of layer name patterns to skip (e.g., ['lm_head'])
    
    Returns:
        The quantized model (same object, modified in-place)
    """
    skip_layers = skip_layers or []
    
    def should_skip(name: str) -> bool:
        return any(skip in name for skip in skip_layers)
    
    # Find all Linear layers and calculate original size
    linear_layers: list[tuple[str, nn.Linear]] = []
    original_linear_size: int = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and not should_skip(name):
            linear_layers.append((name, module))
            # Track original size before we replace them
            original_linear_size += module.weight.numel() * module.weight.element_size()
            if module.bias is not None:
                original_linear_size += module.bias.numel() * module.bias.element_size()
    
    print(f"Quantizing {len(linear_layers)} Linear layers to int8...")
    print(f"  Original Linear layers: {original_linear_size / 1e6:.1f} MB")
    
    # Replace each Linear with QuantizedLinear
    for name, linear in linear_layers:
        # Navigate to parent module
        parts: list[str] = name.split('.')
        parent: nn.Module = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        
        # Replace the layer
        qlayer: QuantizedLinear = QuantizedLinear.from_linear(linear)
        setattr(parent, parts[-1], qlayer)
    
    # Calculate memory of quantized layers
    quantized_size: int = sum(
        m.weight_int8.numel() * 1 + m.weight_scale.numel() * m.weight_scale.element_size()
        for m in model.modules() if isinstance(m, QuantizedLinear)
    )
    
    reduction_pct: float = (1 - quantized_size / original_linear_size) * 100
    print(f"  Quantized layers: {quantized_size / 1e6:.1f} MB")
    print(f"  Memory reduction: {reduction_pct:.0f}% (on quantized layers)")
    
    return model


def dequantize_model_for_inference(model: nn.Module, dtype: torch.dtype = torch.float16) -> nn.Module:
    """Pre-dequantize all QuantizedLinear layers for fast inference.
    
    This caches dequantized fp16 weights in memory for faster forward passes.
    Trades memory for speed - use when you want storage savings but normal inference speed.
    
    Args:
        model: Model with QuantizedLinear layers
        dtype: Target dtype for dequantized weights
    
    Returns:
        The same model with cached dequantized weights
    """
    count: int = 0
    for module in model.modules():
        if isinstance(module, QuantizedLinear):
            module.dequantize_and_cache(dtype)
            count += 1
    
    if count > 0:
        print(f"Pre-dequantized {count} layers for fast inference")
    
    return model


def get_model_size_mb(model: nn.Module) -> float:
    """Get model size in MB (parameters + buffers)."""
    param_size: int = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size: int = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / 1e6


def save_quantized_model(model: nn.Module, path: str) -> None:
    """Save a quantized model to disk.
    
    Args:
        model: The quantized model
        path: Path to save to (e.g., 'weights/tinyllama-int8.pt')
    """
    import os
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    
    # Save state dict (includes int8 weights and scales)
    torch.save({
        'state_dict': model.state_dict(),
        'config': model.config if hasattr(model, 'config') else None,
        'quantized': True,
    }, path)
    
    size_mb: float = os.path.getsize(path) / 1e6
    print(f"Saved quantized model to {path} ({size_mb:.1f} MB)")


def load_quantized_model(path: str, model: nn.Module) -> nn.Module:
    """Load a quantized model from disk.
    
    Note: The model architecture must already have QuantizedLinear layers.
    Call quantize_model_int8() first to convert the architecture, then load weights.
    
    Args:
        path: Path to the saved quantized model
        model: Model with QuantizedLinear layers (call quantize_model_int8 first)
    
    Returns:
        Model with loaded quantized weights
    """
    checkpoint: dict = torch.load(path, map_location='cpu', weights_only=False)
    
    if not checkpoint.get('quantized', False):
        raise ValueError(f"{path} is not a quantized model checkpoint")
    
    model.load_state_dict(checkpoint['state_dict'])
    print(f"Loaded quantized model from {path}")
    
    return model


def is_quantized_checkpoint(path: str) -> bool:
    """Check if a checkpoint file is a quantized model."""
    if not path.endswith('.pt'):
        return False
    try:
        checkpoint: dict = torch.load(path, map_location='cpu', weights_only=False)
        return checkpoint.get('quantized', False)
    except Exception:
        return False
