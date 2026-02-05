"""Vision-Language Model Configuration."""

from __future__ import annotations

from dataclasses import dataclass, field

from .config import LlamaConfig


@dataclass
class VisionConfig:
    """CLIP Vision Transformer configuration."""

    image_size: int = 336
    patch_size: int = 14
    hidden_size: int = 1024
    intermediate_size: int = 4096
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    layer_norm_eps: float = 1e-5

    @property
    def num_patches(self) -> int:
        """Number of patches per image (excluding CLS token)."""
        return (self.image_size // self.patch_size) ** 2

    @property
    def num_positions(self) -> int:
        """Number of position embeddings (patches + CLS token)."""
        return self.num_patches + 1

    @property
    def head_dim(self) -> int:
        """Dimension per attention head."""
        return self.hidden_size // self.num_attention_heads

    def __post_init__(self) -> None:
        if self.image_size % self.patch_size != 0:
            raise ValueError(
                f"image_size ({self.image_size}) must be divisible by patch_size ({self.patch_size})"
            )
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by num_attention_heads ({self.num_attention_heads})"
            )


@dataclass
class VLMConfig:
    """Vision-Language Model (LLaVA) configuration."""

    vision_config: VisionConfig = field(default_factory=VisionConfig)
    llm_config: LlamaConfig = field(default_factory=LlamaConfig)

    # Projector settings
    projector_hidden_size: int = 4096  # Output dimension (matches llm_config.d_model)

    # Special tokens
    image_token_id: int = 32000  # Token ID for <image> placeholder

    # Vision feature extraction
    vision_feature_layer: int = -2  # Which vision layer to extract features from
    vision_feature_select_strategy: str = "default"  # "default" or "full"

    @property
    def vision_hidden_size(self) -> int:
        """Input dimension for projector (from vision encoder)."""
        return self.vision_config.hidden_size

    @property
    def text_hidden_size(self) -> int:
        """Output dimension for projector (for LLM)."""
        return self.llm_config.d_model

    @property
    def num_image_tokens(self) -> int:
        """Number of tokens per image (576 for 336x336 with 14x14 patches)."""
        return self.vision_config.num_patches
