"""Precision policy helpers for AMP dtype resolution and OBB geometry FP32 casts.

These helpers are pure and hold no solver state: the training loop resolves the
autocast dtype and validates hardware support once, then recursively converts
only OBB geometry tensors to FP32 before criterion evaluation.
"""

from __future__ import annotations

import torch

#: Keys whose tensors carry OBB geometry and must be evaluated in FP32.
GEOMETRY_KEYS = frozenset({"pred_boxes", "pred_corners", "ref_points"})

_AMP_DTYPE_NAMES: dict[str, torch.dtype] = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def resolve_amp_dtype(name: str | None) -> torch.dtype:
    """Map a YAML `amp_dtype` value to a ``torch.dtype``.

    Missing names preserve the existing CUDA FP16 autocast default.
    """
    if name is None:
        return torch.float16
    try:
        return _AMP_DTYPE_NAMES[name]
    except KeyError:
        raise ValueError(
            f"unsupported amp_dtype {name!r}; supported values are 'float16' and 'bfloat16'"
        ) from None


def validate_amp_dtype_support(dtype: torch.dtype, device: torch.device) -> None:
    """Raise before the first batch if the dtype is unusable on the device.

    Only BF16 on CUDA hardware is checked; CPU behavior is intentionally
    unchanged, matching the existing non-CUDA AMP path.
    """
    if dtype is not torch.bfloat16 or device.type != "cuda":
        return
    if not torch.cuda.is_bf16_supported():
        index = (
            device.index if device.index is not None else torch.cuda.current_device()
        )
        device_name = torch.cuda.get_device_name(index)
        raise RuntimeError(
            f"BF16 AMP is not supported on CUDA device {index} ({device_name}); "
            "set amp_dtype: float16 instead"
        )


def cast_obb_geometry_fp32(outputs: object) -> object:
    """Return a copy of ``outputs`` with every OBB geometry tensor in FP32.

    Rebuilds nested mappings, lists, and tuples non-mutatingly. Tensors under
    the ``pred_boxes``, ``pred_corners``, or ``ref_points`` keys are cast with
    ``.float()`` (differentiable, so gradients flow into the FP16 model
    forward). Logits, metadata, and unrelated tensors are returned unchanged.
    """
    if isinstance(outputs, dict):
        return {
            key: _cast_geometry_value(value)
            if key in GEOMETRY_KEYS
            else cast_obb_geometry_fp32(value)
            for key, value in outputs.items()
        }
    if isinstance(outputs, list):
        return [cast_obb_geometry_fp32(item) for item in outputs]
    if isinstance(outputs, tuple):
        return tuple(cast_obb_geometry_fp32(item) for item in outputs)
    return outputs


def _cast_geometry_value(value: object) -> object:
    """Cast every tensor under a geometry slot to FP32, keeping container shape."""
    if isinstance(value, torch.Tensor):
        return value.float()
    if isinstance(value, dict):
        return {key: _cast_geometry_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cast_geometry_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cast_geometry_value(item) for item in value)
    return value
