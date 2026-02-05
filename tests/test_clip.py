"""Tests for CLIP vision components."""

import pytest
import torch

from lolama.model.vlm_config import VisionConfig
from lolama.vision.clip import (
    CLIPVisionEmbeddings,
    CLIPAttention,
    CLIPMLP,
    CLIPEncoderLayer,
    CLIPEncoder,
    CLIPVisionTransformer,
)
from lolama.vision.processor import CLIPImageProcessor
from lolama.vision.projector import MultiModalProjector


class TestVisionConfig:
    """Tests for VisionConfig dataclass."""

    def test_default_config(self):
        config = VisionConfig()
        assert config.image_size == 336
        assert config.patch_size == 14
        assert config.hidden_size == 1024
        assert config.num_hidden_layers == 24
        assert config.num_attention_heads == 16

    def test_num_patches(self):
        config = VisionConfig(image_size=336, patch_size=14)
        assert config.num_patches == 576  # (336/14)^2

    def test_num_positions(self):
        config = VisionConfig(image_size=336, patch_size=14)
        assert config.num_positions == 577  # patches + CLS

    def test_head_dim(self):
        config = VisionConfig(hidden_size=1024, num_attention_heads=16)
        assert config.head_dim == 64

    def test_validation_image_size(self):
        with pytest.raises(ValueError, match="image_size.*must be divisible by patch_size"):
            VisionConfig(image_size=100, patch_size=14)

    def test_validation_hidden_size(self):
        with pytest.raises(ValueError, match="hidden_size.*must be divisible by num_attention_heads"):
            VisionConfig(hidden_size=100, num_attention_heads=16)


class TestCLIPImageProcessor:
    """Tests for CLIPImageProcessor."""

    def test_preprocess_single_image(self):
        from PIL import Image

        processor = CLIPImageProcessor(image_size=224)
        # Create dummy image
        image = Image.new("RGB", (640, 480), color="red")
        result = processor.preprocess(image)

        assert "pixel_values" in result
        assert result["pixel_values"].shape == (1, 3, 224, 224)
        assert result["pixel_values"].dtype == torch.float32

    def test_preprocess_batch(self):
        from PIL import Image

        processor = CLIPImageProcessor(image_size=224)
        images = [
            Image.new("RGB", (640, 480), color="red"),
            Image.new("RGB", (320, 240), color="blue"),
        ]
        result = processor.preprocess(images)

        assert result["pixel_values"].shape == (2, 3, 224, 224)

    def test_normalize_values(self):
        from PIL import Image

        processor = CLIPImageProcessor(image_size=224)
        image = Image.new("RGB", (224, 224), color=(255, 255, 255))  # White
        result = processor.preprocess(image)

        # After normalization, values should not be in [0, 1]
        # White (1.0) normalized: (1.0 - mean) / std
        assert result["pixel_values"].max() > 1.0 or result["pixel_values"].min() < 0.0

    def test_from_config(self, tiny_vision_config):
        processor = CLIPImageProcessor.from_config(tiny_vision_config)
        assert processor.image_size == tiny_vision_config.image_size


class TestCLIPVisionEmbeddings:
    """Tests for CLIPVisionEmbeddings."""

    def test_output_shape(self, tiny_vision_config):
        embeddings = CLIPVisionEmbeddings(tiny_vision_config)
        batch_size = 2
        pixel_values = torch.randn(
            batch_size, 3, tiny_vision_config.image_size, tiny_vision_config.image_size
        )

        output = embeddings(pixel_values)

        # Output: (batch, num_patches+1, hidden_size)
        expected_seq_len = tiny_vision_config.num_positions
        assert output.shape == (batch_size, expected_seq_len, tiny_vision_config.hidden_size)

    def test_class_embedding_present(self, tiny_vision_config):
        embeddings = CLIPVisionEmbeddings(tiny_vision_config)
        pixel_values = torch.randn(1, 3, tiny_vision_config.image_size, tiny_vision_config.image_size)

        output = embeddings(pixel_values)

        # First position should be CLS token
        # Check it's different from patch embeddings (not guaranteed but typical)
        assert output.shape[1] == tiny_vision_config.num_positions


class TestCLIPAttention:
    """Tests for CLIPAttention."""

    def test_output_shape(self, tiny_vision_config):
        attention = CLIPAttention(tiny_vision_config)
        batch_size, seq_len = 2, 17
        hidden_states = torch.randn(batch_size, seq_len, tiny_vision_config.hidden_size)

        output = attention(hidden_states)

        assert output.shape == hidden_states.shape

    def test_no_nan(self, tiny_vision_config):
        attention = CLIPAttention(tiny_vision_config)
        hidden_states = torch.randn(2, 17, tiny_vision_config.hidden_size)

        output = attention(hidden_states)

        assert not torch.isnan(output).any()


class TestCLIPMLP:
    """Tests for CLIPMLP."""

    def test_output_shape(self, tiny_vision_config):
        mlp = CLIPMLP(tiny_vision_config)
        batch_size, seq_len = 2, 17
        hidden_states = torch.randn(batch_size, seq_len, tiny_vision_config.hidden_size)

        output = mlp(hidden_states)

        assert output.shape == hidden_states.shape

    def test_intermediate_size(self, tiny_vision_config):
        mlp = CLIPMLP(tiny_vision_config)
        assert mlp.fc1.out_features == tiny_vision_config.intermediate_size
        assert mlp.fc2.in_features == tiny_vision_config.intermediate_size


class TestCLIPEncoderLayer:
    """Tests for CLIPEncoderLayer."""

    def test_output_shape(self, tiny_vision_config):
        layer = CLIPEncoderLayer(tiny_vision_config)
        batch_size, seq_len = 2, 17
        hidden_states = torch.randn(batch_size, seq_len, tiny_vision_config.hidden_size)

        output = layer(hidden_states)

        assert output.shape == hidden_states.shape

    def test_residual_connection(self, tiny_vision_config):
        layer = CLIPEncoderLayer(tiny_vision_config)
        hidden_states = torch.randn(2, 17, tiny_vision_config.hidden_size)

        # Output should be different from input (transformations applied)
        output = layer(hidden_states)
        assert not torch.allclose(output, hidden_states)


class TestCLIPEncoder:
    """Tests for CLIPEncoder."""

    def test_output_shape(self, tiny_vision_config):
        encoder = CLIPEncoder(tiny_vision_config)
        batch_size, seq_len = 2, 17
        hidden_states = torch.randn(batch_size, seq_len, tiny_vision_config.hidden_size)

        output, _ = encoder(hidden_states)

        assert output.shape == hidden_states.shape

    def test_hidden_states_output(self, tiny_vision_config):
        encoder = CLIPEncoder(tiny_vision_config)
        hidden_states = torch.randn(2, 17, tiny_vision_config.hidden_size)

        _, all_hidden_states = encoder(hidden_states, output_hidden_states=True)

        # Should have num_layers + 1 hidden states (input + each layer output)
        assert len(all_hidden_states) == tiny_vision_config.num_hidden_layers + 1

    def test_no_hidden_states(self, tiny_vision_config):
        encoder = CLIPEncoder(tiny_vision_config)
        hidden_states = torch.randn(2, 17, tiny_vision_config.hidden_size)

        _, all_hidden_states = encoder(hidden_states, output_hidden_states=False)

        assert all_hidden_states is None


class TestCLIPVisionTransformer:
    """Tests for CLIPVisionTransformer."""

    def test_forward_shape(self, tiny_vision_config):
        vit = CLIPVisionTransformer(tiny_vision_config)
        batch_size = 2
        pixel_values = torch.randn(
            batch_size, 3, tiny_vision_config.image_size, tiny_vision_config.image_size
        )

        last_hidden, hidden_states = vit(pixel_values)

        expected_seq_len = tiny_vision_config.num_positions
        assert last_hidden.shape == (batch_size, expected_seq_len, tiny_vision_config.hidden_size)

    def test_get_image_features_default(self, tiny_vision_config):
        vit = CLIPVisionTransformer(tiny_vision_config)
        batch_size = 2
        pixel_values = torch.randn(
            batch_size, 3, tiny_vision_config.image_size, tiny_vision_config.image_size
        )

        # Default strategy excludes CLS token
        features = vit.get_image_features(pixel_values)

        # Should be (batch, num_patches, hidden_size) - no CLS
        assert features.shape == (batch_size, tiny_vision_config.num_patches, tiny_vision_config.hidden_size)

    def test_get_image_features_full(self, tiny_vision_config):
        vit = CLIPVisionTransformer(tiny_vision_config)
        batch_size = 2
        pixel_values = torch.randn(
            batch_size, 3, tiny_vision_config.image_size, tiny_vision_config.image_size
        )

        # Full strategy includes CLS token
        features = vit.get_image_features(
            pixel_values, vision_feature_select_strategy="full"
        )

        # Should be (batch, num_positions, hidden_size) - with CLS
        assert features.shape == (batch_size, tiny_vision_config.num_positions, tiny_vision_config.hidden_size)

    def test_vision_feature_layer(self, tiny_vision_config):
        vit = CLIPVisionTransformer(tiny_vision_config)
        pixel_values = torch.randn(
            1, 3, tiny_vision_config.image_size, tiny_vision_config.image_size
        )

        # Get features from different layers
        features_last = vit.get_image_features(pixel_values, vision_feature_layer=-1)
        features_second_last = vit.get_image_features(pixel_values, vision_feature_layer=-2)

        # They should be different
        assert not torch.allclose(features_last, features_second_last)


class TestMultiModalProjector:
    """Tests for MultiModalProjector."""

    def test_output_shape(self, tiny_vlm_config):
        projector = MultiModalProjector(tiny_vlm_config)
        batch_size = 2
        num_patches = tiny_vlm_config.vision_config.num_patches
        image_features = torch.randn(
            batch_size, num_patches, tiny_vlm_config.vision_hidden_size
        )

        output = projector(image_features)

        assert output.shape == (batch_size, num_patches, tiny_vlm_config.text_hidden_size)

    def test_dimensions(self, tiny_vlm_config):
        projector = MultiModalProjector(tiny_vlm_config)
        assert projector.linear_1.in_features == tiny_vlm_config.vision_hidden_size
        assert projector.linear_2.out_features == tiny_vlm_config.text_hidden_size
