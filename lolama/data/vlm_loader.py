"""Weight loading for Vision-Language Models (LLaVA)."""

from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoConfig, AutoProcessor, LlavaForConditionalGeneration

from ..model import LlamaConfig
from ..model.vlm_config import VisionConfig, VLMConfig
from ..model.llava import LLaVA
from ..utils.logging import get_data_logger
from .registry import MODEL_REGISTRY
from .loader import resolve_model_source, WEIGHTS_DIR, WeightLoadingError

logger = get_data_logger()


def create_vlm_config_from_hf(
    hf_model_name: str | Path,
    trust_remote_code: bool = False,
    local_files_only: bool = False,
) -> VLMConfig:
    """Create VLMConfig from HuggingFace LLaVA model.

    Extracts both vision and language model configurations.

    Args:
        hf_model_name: HuggingFace model name or local path
        trust_remote_code: Whether to trust remote code
        local_files_only: Only use local files

    Returns:
        VLMConfig with vision and LLM settings
    """
    logger.info(f"Loading VLM config from {hf_model_name}...")

    hf_config = AutoConfig.from_pretrained(
        hf_model_name,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
    )

    # Extract vision config
    vision_cfg = hf_config.vision_config
    vision_config = VisionConfig(
        image_size=getattr(vision_cfg, "image_size", 336),
        patch_size=getattr(vision_cfg, "patch_size", 14),
        hidden_size=vision_cfg.hidden_size,
        intermediate_size=vision_cfg.intermediate_size,
        num_hidden_layers=vision_cfg.num_hidden_layers,
        num_attention_heads=vision_cfg.num_attention_heads,
        layer_norm_eps=getattr(vision_cfg, "layer_norm_eps", 1e-5),
    )

    # Extract LLM config
    text_cfg = hf_config.text_config
    num_heads = text_cfg.num_attention_heads
    llm_config = LlamaConfig(
        vocab_size=text_cfg.vocab_size,
        d_model=text_cfg.hidden_size,
        num_heads=num_heads,
        num_kv_heads=getattr(text_cfg, "num_key_value_heads", num_heads),
        num_layers=text_cfg.num_hidden_layers,
        hidden_dim=text_cfg.intermediate_size,
        max_seq_len=getattr(text_cfg, "max_position_embeddings", 4096),
        eps=getattr(text_cfg, "rms_norm_eps", 1e-5),
        tie_word_embeddings=getattr(text_cfg, "tie_word_embeddings", False),
        rope_base=int(getattr(text_cfg, "rope_theta", 10000)),
    )

    # Extract VLM-specific settings
    image_token_id = getattr(hf_config, "image_token_index", 32000)
    vision_feature_layer = getattr(hf_config, "vision_feature_layer", -2)
    vision_feature_select_strategy = getattr(
        hf_config, "vision_feature_select_strategy", "default"
    )

    # Projector output size (matches LLM hidden size)
    projector_hidden_size = llm_config.d_model

    vlm_config = VLMConfig(
        vision_config=vision_config,
        llm_config=llm_config,
        projector_hidden_size=projector_hidden_size,
        image_token_id=image_token_id,
        vision_feature_layer=vision_feature_layer,
        vision_feature_select_strategy=vision_feature_select_strategy,
    )

    logger.debug(
        f"VLM Config: vision_hidden={vision_config.hidden_size}, "
        f"llm_hidden={llm_config.d_model}, "
        f"image_token_id={image_token_id}, "
        f"vision_feature_layer={vision_feature_layer}"
    )

    return vlm_config


def build_llava_weight_mapping(config: VLMConfig) -> dict[str, str]:
    """Build weight mapping from HuggingFace LLaVA to our model.

    Args:
        config: VLM configuration

    Returns:
        Dictionary mapping HF keys to our keys
    """
    mapping: dict[str, str] = {}

    # Vision tower mappings
    # HF: model.vision_tower.vision_model.* -> Ours: vision_tower.*

    # Embeddings
    mapping["vision_tower.vision_model.embeddings.patch_embedding.weight"] = (
        "vision_tower.embeddings.patch_embedding.weight"
    )
    mapping["vision_tower.vision_model.embeddings.class_embedding"] = (
        "vision_tower.embeddings.class_embedding"
    )
    mapping["vision_tower.vision_model.embeddings.position_embedding.weight"] = (
        "vision_tower.embeddings.position_embedding.weight"
    )

    # Pre-layernorm
    mapping["vision_tower.vision_model.pre_layrnorm.weight"] = (
        "vision_tower.pre_layrnorm.weight"
    )
    mapping["vision_tower.vision_model.pre_layrnorm.bias"] = (
        "vision_tower.pre_layrnorm.bias"
    )

    # Vision encoder layers
    num_vision_layers = config.vision_config.num_hidden_layers
    for i in range(num_vision_layers):
        hf_prefix = f"vision_tower.vision_model.encoder.layers.{i}"
        our_prefix = f"vision_tower.encoder.layers.{i}"

        # Self-attention
        mapping[f"{hf_prefix}.self_attn.q_proj.weight"] = f"{our_prefix}.self_attn.q_proj.weight"
        mapping[f"{hf_prefix}.self_attn.q_proj.bias"] = f"{our_prefix}.self_attn.q_proj.bias"
        mapping[f"{hf_prefix}.self_attn.k_proj.weight"] = f"{our_prefix}.self_attn.k_proj.weight"
        mapping[f"{hf_prefix}.self_attn.k_proj.bias"] = f"{our_prefix}.self_attn.k_proj.bias"
        mapping[f"{hf_prefix}.self_attn.v_proj.weight"] = f"{our_prefix}.self_attn.v_proj.weight"
        mapping[f"{hf_prefix}.self_attn.v_proj.bias"] = f"{our_prefix}.self_attn.v_proj.bias"
        mapping[f"{hf_prefix}.self_attn.out_proj.weight"] = f"{our_prefix}.self_attn.out_proj.weight"
        mapping[f"{hf_prefix}.self_attn.out_proj.bias"] = f"{our_prefix}.self_attn.out_proj.bias"

        # Layer norms
        mapping[f"{hf_prefix}.layer_norm1.weight"] = f"{our_prefix}.layer_norm1.weight"
        mapping[f"{hf_prefix}.layer_norm1.bias"] = f"{our_prefix}.layer_norm1.bias"
        mapping[f"{hf_prefix}.layer_norm2.weight"] = f"{our_prefix}.layer_norm2.weight"
        mapping[f"{hf_prefix}.layer_norm2.bias"] = f"{our_prefix}.layer_norm2.bias"

        # MLP
        mapping[f"{hf_prefix}.mlp.fc1.weight"] = f"{our_prefix}.mlp.fc1.weight"
        mapping[f"{hf_prefix}.mlp.fc1.bias"] = f"{our_prefix}.mlp.fc1.bias"
        mapping[f"{hf_prefix}.mlp.fc2.weight"] = f"{our_prefix}.mlp.fc2.weight"
        mapping[f"{hf_prefix}.mlp.fc2.bias"] = f"{our_prefix}.mlp.fc2.bias"

    # Post-layernorm
    mapping["vision_tower.vision_model.post_layernorm.weight"] = (
        "vision_tower.post_layernorm.weight"
    )
    mapping["vision_tower.vision_model.post_layernorm.bias"] = (
        "vision_tower.post_layernorm.bias"
    )

    # Multi-modal projector mappings
    mapping["multi_modal_projector.linear_1.weight"] = "multi_modal_projector.linear_1.weight"
    mapping["multi_modal_projector.linear_1.bias"] = "multi_modal_projector.linear_1.bias"
    mapping["multi_modal_projector.linear_2.weight"] = "multi_modal_projector.linear_2.weight"
    mapping["multi_modal_projector.linear_2.bias"] = "multi_modal_projector.linear_2.bias"

    # Language model mappings
    # HF: language_model.model.* -> Ours: language_model.*
    mapping["language_model.model.embed_tokens.weight"] = "language_model.embed_tokens.weight"

    num_llm_layers = config.llm_config.num_layers
    for i in range(num_llm_layers):
        hf_prefix = f"language_model.model.layers.{i}"
        our_prefix = f"language_model.layers.{i}"

        # Attention
        mapping[f"{hf_prefix}.self_attn.q_proj.weight"] = f"{our_prefix}.attention.q_proj.weight"
        mapping[f"{hf_prefix}.self_attn.k_proj.weight"] = f"{our_prefix}.attention.k_proj.weight"
        mapping[f"{hf_prefix}.self_attn.v_proj.weight"] = f"{our_prefix}.attention.v_proj.weight"
        mapping[f"{hf_prefix}.self_attn.o_proj.weight"] = f"{our_prefix}.attention.o_proj.weight"

        # FFN
        mapping[f"{hf_prefix}.mlp.gate_proj.weight"] = f"{our_prefix}.feed_forward.w_gate.weight"
        mapping[f"{hf_prefix}.mlp.up_proj.weight"] = f"{our_prefix}.feed_forward.w_up.weight"
        mapping[f"{hf_prefix}.mlp.down_proj.weight"] = f"{our_prefix}.feed_forward.w_down.weight"

        # Norms
        mapping[f"{hf_prefix}.input_layernorm.weight"] = f"{our_prefix}.attention_norm.weight"
        mapping[f"{hf_prefix}.post_attention_layernorm.weight"] = f"{our_prefix}.ffn_norm.weight"

    # Final norm and LM head
    mapping["language_model.model.norm.weight"] = "language_model.norm.weight"
    mapping["language_model.lm_head.weight"] = "language_model.lm_head.weight"

    return mapping


def load_llava_weights(
    our_model: LLaVA,
    hf_model_path: str | Path,
    trust_remote_code: bool = False,
    local_files_only: bool = False,
    strict_threshold: float = 0.90,
) -> LLaVA:
    """Load weights from HuggingFace LLaVA model into our model.

    Args:
        our_model: Target LLaVA model
        hf_model_path: Path or HF model name
        trust_remote_code: Whether to trust remote code
        local_files_only: Only use local files
        strict_threshold: Minimum fraction of weights that must load successfully

    Returns:
        Model with loaded weights

    Raises:
        WeightLoadingError: If match rate falls below strict_threshold
    """
    logger.info(f"Loading LLaVA weights from {hf_model_path}...")

    # Load HF model
    hf_model = LlavaForConditionalGeneration.from_pretrained(
        hf_model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
    )

    hf_state = hf_model.state_dict()

    # Build mapping
    mapping = build_llava_weight_mapping(our_model._vlm_config)

    # Get current state dict
    our_state = our_model.state_dict()

    # Track results
    new_state_dict: dict[str, torch.Tensor] = {}
    matched = 0
    missing_in_hf: list[str] = []
    missing_in_ours: list[str] = []
    shape_mismatches: list[dict] = []

    for hf_key, our_key in mapping.items():
        if hf_key not in hf_state:
            missing_in_hf.append(hf_key)
            continue

        hf_tensor = hf_state[hf_key]

        if our_key not in our_state:
            missing_in_ours.append(our_key)
            continue

        our_tensor = our_state[our_key]

        if hf_tensor.shape != our_tensor.shape:
            shape_mismatches.append({
                "hf_key": hf_key,
                "our_key": our_key,
                "hf_shape": tuple(hf_tensor.shape),
                "our_shape": tuple(our_tensor.shape),
            })
            continue

        new_state_dict[our_key] = hf_tensor.to(our_tensor.dtype)
        matched += 1

    # Calculate match rate
    total_expected = len(mapping)
    match_rate = matched / total_expected if total_expected > 0 else 0.0

    # Load weights
    missing_after_load, unexpected = our_model.load_state_dict(new_state_dict, strict=False)

    # Report results
    logger.info(f"Successfully loaded {matched}/{total_expected} weights ({match_rate:.1%})")

    if missing_after_load:
        # Filter out expected missing keys (RoPE buffers, position_ids)
        expected_missing = [k for k in missing_after_load if any(x in k for x in ["cos", "sin", "position_ids"])]
        other_missing = [k for k in missing_after_load if k not in expected_missing]
        if expected_missing:
            logger.debug(f"{len(expected_missing)} expected missing keys (RoPE buffers, position_ids)")
        if other_missing:
            logger.warning(f"{len(other_missing)} unexpected missing keys: {other_missing[:5]}")

    if missing_in_hf:
        logger.warning(f"{len(missing_in_hf)} keys missing in HuggingFace model: {missing_in_hf[:5]}")

    if missing_in_ours:
        logger.warning(f"{len(missing_in_ours)} keys missing in our model: {missing_in_ours[:5]}")

    if shape_mismatches:
        for m in shape_mismatches[:5]:
            logger.error(f"Shape mismatch: {m['our_key']}: HF={m['hf_shape']} vs Ours={m['our_shape']}")

    # Strict threshold check
    if match_rate < strict_threshold:
        error_msg = (
            f"Weight loading failed: {match_rate:.1%} matched, "
            f"but {strict_threshold:.0%} required. "
            f"Missing in HF: {len(missing_in_hf)}, "
            f"Missing in model: {len(missing_in_ours)}, "
            f"Shape mismatches: {len(shape_mismatches)}"
        )
        raise WeightLoadingError(error_msg)

    if matched == total_expected:
        logger.info("All LLaVA weights loaded successfully!")

    return our_model


def download_llava_model(
    hf_name: str,
    save_dir: Path,
    trust_remote_code: bool = False,
) -> None:
    """Download and save LLaVA model + processor to local directory.

    Args:
        hf_name: HuggingFace model name
        save_dir: Local directory to save to
        trust_remote_code: Whether to trust remote code
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading LLaVA model {hf_name} to {save_dir}...")

    # Download model
    hf_model = LlavaForConditionalGeneration.from_pretrained(
        hf_name,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=trust_remote_code,
    )
    hf_model.save_pretrained(save_dir)

    # Download processor (includes tokenizer + image processor)
    processor = AutoProcessor.from_pretrained(hf_name, trust_remote_code=trust_remote_code)
    processor.save_pretrained(save_dir)

    total_params = sum(p.numel() for p in hf_model.parameters())
    size_mb = total_params * 2 / 1024**2
    logger.info(f"Saved to {save_dir} ({total_params:,} params, {size_mb:.1f} MB fp16)")


def load_llava_model(
    model_name_or_path: str,
    device: str | None = None,
    dtype: torch.dtype = torch.float16,
) -> LLaVA:
    """Load a pretrained LLaVA model.

    If weights are not found locally, downloads and saves them to the
    weights/ directory for future use.

    Args:
        model_name_or_path: HuggingFace model name, registry alias, or local path
        device: Device to load on (default: auto-detect)
        dtype: Model dtype (default: float16)

    Returns:
        LLaVA model with loaded weights
    """
    from ..utils.device import resolve_device

    if device is None:
        device = resolve_device()

    logger.info(f"Loading LLaVA model: {model_name_or_path}")

    source = resolve_model_source(model_name_or_path)
    trust_remote_code = source.get("trust_remote_code", False)

    # Auto-download if not found locally
    if source["local_path"] is None and source["hf_name"] is not None:
        hf_name = source["hf_name"]
        folder_name = source.get("folder") or hf_name.replace("/", "_").replace("\\", "_")
        save_dir = WEIGHTS_DIR / folder_name
        if not (save_dir.exists() and any(save_dir.iterdir())):
            download_llava_model(hf_name, save_dir, trust_remote_code)
        source["local_path"] = save_dir

    model_path = source["local_path"] if source["local_path"] is not None else source["hf_name"]

    # Create config
    config = create_vlm_config_from_hf(
        model_path,
        trust_remote_code=trust_remote_code,
        local_files_only=True,
    )

    # Create model architecture
    logger.info("Creating LLaVA architecture...")
    with torch.device("meta"):
        model = LLaVA(config, init_weights=False)

    model = model.to_empty(device="cpu").to(dtype)
    model.init_rope()

    # Load weights
    model = load_llava_weights(
        model,
        model_path,
        trust_remote_code=trust_remote_code,
        local_files_only=True,
    )

    # Move to target device
    if device != "cpu":
        logger.info(f"Moving model to {device}...")
        model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"LLaVA model ready! Total parameters: {total_params:,}")

    return model
