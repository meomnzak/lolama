"""LLaVA (Large Language and Vision Assistant) Model.

Combines CLIP vision encoder with LLaMA language model through an MLP projector.

Architecture:
    Image → CLIP ViT → [576 patches × 1024] → MLP Projector → [576 × 4096] → LLaMA → logits

The model replaces <image> tokens in the input with encoded image features.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import LlamaConfig
from .kv_cache import KVCache
from .llama import Llama
from .vlm_config import VLMConfig
from ..vision.clip import CLIPVisionTransformer
from ..vision.projector import MultiModalProjector


class LLaVA(nn.Module):
    """LLaVA Vision-Language Model.

    Implements the LLaVA-1.5 architecture:
    - Vision tower: CLIP ViT-L/14 @ 336px
    - Projector: 2-layer MLP
    - Language model: LLaMA (7B or 13B)

    Compatible with TextGenerator via GenerativeModel protocol.
    """

    def __init__(self, config: VLMConfig, init_weights: bool = True) -> None:
        """
        Args:
            config: VLM configuration containing vision and LLM settings
            init_weights: If True, initialize weights randomly. Set to False when
                         loading pretrained weights to skip unnecessary init.
        """
        super().__init__()
        self.config = config
        self._vlm_config = config  # Keep VLM config accessible

        # Vision components
        self.vision_tower = CLIPVisionTransformer(config.vision_config)
        self.multi_modal_projector = MultiModalProjector(config)

        # Language model
        self.language_model = Llama(config.llm_config, init_weights=init_weights)

        # Track whether image features are cached (for generation)
        self._image_features_cached = False

    @property
    def device(self) -> torch.device:
        """Get model device."""
        return self.language_model.embed_tokens.weight.device

    @property
    def dtype(self) -> torch.dtype:
        """Get model dtype."""
        return self.language_model.embed_tokens.weight.dtype

    def encode_images(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Encode images to feature vectors.

        Args:
            pixel_values: (batch, 3, H, W) normalized images

        Returns:
            (batch, num_patches, text_hidden_size) projected image features
        """
        # Extract vision features from specified layer
        image_features = self.vision_tower.get_image_features(
            pixel_values,
            vision_feature_layer=self._vlm_config.vision_feature_layer,
            vision_feature_select_strategy=self._vlm_config.vision_feature_select_strategy,
        )

        # Project to LLM embedding space
        image_features = self.multi_modal_projector(image_features)

        return image_features

    def _merge_input_ids_with_image_features(
        self,
        input_ids: torch.Tensor,
        image_features: torch.Tensor,
    ) -> torch.Tensor:
        """Replace <image> tokens with image features.

        Args:
            input_ids: (batch, seq_len) token IDs with <image> placeholders
            image_features: (batch, num_image_tokens, hidden_size) encoded images

        Returns:
            (batch, new_seq_len, hidden_size) merged embeddings
        """
        batch_size, seq_len = input_ids.shape
        num_image_tokens = image_features.shape[1]
        image_token_id = self._vlm_config.image_token_id

        # Get text embeddings
        text_embeds = self.language_model.embed_tokens(input_ids)

        # Find image token positions
        image_token_mask = input_ids == image_token_id

        # Count image tokens per sample (should be 1 per image)
        num_image_tokens_per_sample = image_token_mask.sum(dim=1)

        # For now, assume single image per sample
        # More complex handling would be needed for multi-image

        # Build output embeddings
        # New sequence length: original - 1 (image token) + num_image_tokens
        new_seq_len = seq_len - 1 + num_image_tokens

        merged_embeds = torch.zeros(
            batch_size,
            new_seq_len,
            text_embeds.shape[-1],
            dtype=text_embeds.dtype,
            device=text_embeds.device,
        )

        for i in range(batch_size):
            # Find position of <image> token
            image_positions = (input_ids[i] == image_token_id).nonzero(as_tuple=True)[0]

            if len(image_positions) == 0:
                # No image token - just use text embeddings
                merged_embeds[i, :seq_len] = text_embeds[i]
                continue

            image_pos = image_positions[0].item()

            # Copy text before image
            if image_pos > 0:
                merged_embeds[i, :image_pos] = text_embeds[i, :image_pos]

            # Insert image features
            merged_embeds[i, image_pos : image_pos + num_image_tokens] = image_features[i]

            # Copy text after image
            text_after_pos = image_pos + 1
            merged_pos = image_pos + num_image_tokens
            remaining_text_len = seq_len - text_after_pos
            if remaining_text_len > 0:
                merged_embeds[i, merged_pos : merged_pos + remaining_text_len] = (
                    text_embeds[i, text_after_pos:]
                )

        return merged_embeds

    def create_kv_caches(
        self,
        batch_size: int,
        max_seq_len: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> list[KVCache]:
        """Create KV caches for generation.

        For VLMs, the max_seq_len should account for image tokens (576 per image).
        """
        # Add image token space to max_seq_len
        if max_seq_len is not None:
            max_seq_len = max_seq_len + self._vlm_config.num_image_tokens

        return self.language_model.create_kv_caches(
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        kv_caches: list[KVCache] | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass for LLaVA.

        For initial prompt with image:
            - Pass input_ids with <image> token and pixel_values
            - Image features replace the <image> token

        For subsequent generation steps:
            - Pass only input_ids (the new token)
            - Image features are already in KV cache

        Args:
            input_ids: (B, L) token IDs. May contain <image> token placeholder.
            pixel_values: (B, 3, H, W) normalized images. Only needed on first pass.
            inputs_embeds: (B, L, D) pre-computed embeddings (for advanced use).
            kv_caches: Optional KV caches for generation.
            attention_mask: Optional (B, L) attention mask.

        Returns:
            logits: (B, L, vocab_size) or (B, L', vocab_size) if image was merged
        """
        # Handle inputs_embeds directly (for advanced usage)
        if inputs_embeds is not None:
            return self.language_model(
                inputs_embeds=inputs_embeds,
                kv_caches=kv_caches,
                attention_mask=attention_mask,
            )

        # Check if we have images to process
        has_images = pixel_values is not None and not self._image_features_cached

        if has_images:
            # Encode images
            image_features = self.encode_images(pixel_values)

            # Merge image features with text embeddings
            merged_embeds = self._merge_input_ids_with_image_features(
                input_ids, image_features
            )

            # Mark images as processed (cached in KV cache after first pass)
            if kv_caches is not None:
                self._image_features_cached = True

            # Forward through LLM
            return self.language_model(
                inputs_embeds=merged_embeds,
                kv_caches=kv_caches,
                attention_mask=attention_mask,
            )
        else:
            # No image or already cached - standard LLM forward
            return self.language_model(
                input_ids=input_ids,
                kv_caches=kv_caches,
                attention_mask=attention_mask,
            )

    def reset_image_cache(self) -> None:
        """Reset image cache flag for new generation."""
        self._image_features_cached = False

    def init_rope(self) -> None:
        """Re-initialize RoPE buffers in language model."""
        self.language_model.init_rope()

    def count_parameters(self) -> dict[str, int]:
        """Count model parameters by component."""
        vision_params = sum(p.numel() for p in self.vision_tower.parameters())
        projector_params = sum(p.numel() for p in self.multi_modal_projector.parameters())
        llm_counts = self.language_model.count_parameters()

        return {
            "total": vision_params + projector_params + llm_counts["total"],
            "vision_tower": vision_params,
            "projector": projector_params,
            "language_model": llm_counts["total"],
            **{f"llm_{k}": v for k, v in llm_counts.items() if k != "total"},
        }
