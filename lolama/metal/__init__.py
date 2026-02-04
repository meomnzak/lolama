"""Metal (MPS) fused W8A16 dequant+matmul kernel.

JIT-compiles an Obj-C++ extension on first import via torch.utils.cpp_extension.load().
Provides dequant_matmul(x, weight_int8, scales) for fused W8A16 matmul on Apple GPU.
"""

from __future__ import annotations

import os
import logging

import torch

logger = logging.getLogger("lolama.metal")

_ext = None
_available: bool | None = None


def _load_extension():
    """JIT-compile the Metal extension. Returns the module or None on failure."""
    global _ext
    if _ext is not None:
        return _ext

    try:
        from torch.utils.cpp_extension import load

        ext_dir = os.path.dirname(os.path.abspath(__file__))
        source = os.path.join(ext_dir, "metal_ext.cpp")

        _ext = load(
            name="lolama_metal_ext",
            sources=[source],
            extra_cflags=["-ObjC++", "-std=c++17", "-O2"],
            extra_ldflags=["-framework", "Metal", "-framework", "Foundation"],
            verbose=False,
        )
        return _ext
    except Exception as e:
        logger.warning(f"Failed to build Metal extension: {e}")
        return None


def is_available() -> bool:
    """Check if the Metal fused kernel is available."""
    global _available
    if _available is not None:
        return _available

    if not torch.backends.mps.is_available():
        _available = False
        return False

    _available = _load_extension() is not None
    return _available


def dequant_matmul(
    x: torch.Tensor,
    weight_int8: torch.Tensor,
    scales: torch.Tensor,
) -> torch.Tensor:
    """Fused W8A16 dequant+matmul on Metal.

    Computes: output = x @ (weight_int8 * scales)^T
    without materializing the full fp16 weight matrix in DRAM.

    Args:
        x: Input activations [*, K] fp16 on MPS
        weight_int8: Quantized weights [N, K] int8 on MPS
        scales: Per-channel scales [N] fp16 on MPS

    Returns:
        Output tensor [*, N] fp16 on MPS
    """
    ext = _load_extension()
    if ext is None:
        raise RuntimeError("Metal extension not available")

    # Flatten batched input [*, K] → [M, K]
    orig_shape = x.shape
    if x.dim() > 2:
        x = x.reshape(-1, x.size(-1))

    output = ext.dequant_matmul(x, weight_int8, scales)

    # Restore batch dimensions [M, N] → [*, N]
    if len(orig_shape) > 2:
        output = output.reshape(*orig_shape[:-1], output.size(-1))

    return output
