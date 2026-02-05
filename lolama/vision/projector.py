"""Multi-Modal Projector for LLaVA.

Projects vision features from CLIP (1024-dim) to LLM embedding space (4096-dim).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..model.vlm_config import VLMConfig


class MultiModalProjector(nn.Module):
    """2-layer MLP projector from vision to language space.

    Architecture:
        vision_features (1024) -> linear_1 -> GELU -> linear_2 -> llm_features (4096)

    This bridges the representation gap between CLIP vision encoder and LLaMA LLM.
    """

    def __init__(self, config: VLMConfig) -> None:
        super().__init__()
        self.config = config

        self.linear_1 = nn.Linear(
            config.vision_hidden_size,  # 1024 (from CLIP)
            config.projector_hidden_size,  # 4096 (intermediate)
        )
        self.linear_2 = nn.Linear(
            config.projector_hidden_size,  # 4096 (intermediate)
            config.text_hidden_size,  # 4096 (to LLM)
        )

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        """Project vision features to LLM embedding space.

        Args:
            image_features: (batch, num_patches, vision_hidden_size)
                           e.g., (1, 576, 1024) for 336x336 image with 14x14 patches

        Returns:
            (batch, num_patches, text_hidden_size) projected features
            e.g., (1, 576, 4096) ready to be merged with text embeddings
        """
        hidden_states = self.linear_1(image_features)
        hidden_states = F.gelu(hidden_states, approximate="tanh")
        hidden_states = self.linear_2(hidden_states)
        return hidden_states
