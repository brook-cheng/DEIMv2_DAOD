"""Checkpoint state-dict selection for the DEIMv2 application layer.

This module is the ONLY place that decides which sub-dict of a raw
``torch.load`` checkpoint becomes the model's input state. It runs BEFORE any
``model.load_state_dict`` call so the adapter can validate class-count
compatibility on a normalized state.

Order (when ``prefer_ema=True``):

  1. ``checkpoint['ema']['module']`` if present (EMA weights are the
     validation-time weights in DEIM training).
  2. ``checkpoint['model']`` otherwise.

When ``prefer_ema=False``: always ``checkpoint['model']``.

A leading ``module.`` prefix (an artifact of DDP / ``DataParallel`` training,
where every parameter is registered under ``module.<name>``) is stripped from
every key so the resulting state matches the non-DDP model's parameter names.

Raises:
    CheckpointCompatibilityError: if neither ``ema`` nor ``model`` is present,
        or if the selected sub-dict is not a mapping.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from deim_app.errors import CheckpointCompatibilityError

__all__ = ["select_model_state"]


#: Prefix emitted by DDP / ``nn.DataParallel`` wrapping (see
#: ``engine/solver/_solver.py:remove_module_prefix`` for the engine's own copy).
_MODULE_PREFIX = "module."


def select_model_state(
    checkpoint: Mapping[str, Any],
    prefer_ema: bool = True,
) -> dict[str, Any]:
    """Select and normalize the model state dict from a raw checkpoint.

    Args:
        checkpoint: A raw ``torch.load`` result. Must be a mapping containing
            either an ``ema`` dict (with a ``module`` sub-key) or a top-level
            ``model`` dict.
        prefer_ema: When ``True`` (default) prefer ``ema.module`` over
            ``model`` — EMA weights are what DEIM validators evaluate against.
            When ``False``, always take ``model``.

    Returns:
        A fresh ``dict`` of parameter-name → tensor, with any leading
        ``module.`` prefix stripped from each key.

    Raises:
        CheckpointCompatibilityError: if neither ``ema`` nor ``model`` is
            present, or if the selected entry is not a mapping.
    """
    raw_state = _select_raw_state(checkpoint, prefer_ema=prefer_ema)
    if not isinstance(raw_state, Mapping):
        raise CheckpointCompatibilityError(
            f"Selected checkpoint entry is not a mapping; got {type(raw_state).__name__}"
        )
    return _strip_module_prefix(raw_state)


def _select_raw_state(
    checkpoint: Mapping[str, Any],
    *,
    prefer_ema: bool,
) -> Any:
    """Pick the ``ema.module`` or ``model`` sub-dict per the precedence rules.

    Returns ``Any``; the caller validates the result is a ``Mapping``.
    """
    if prefer_ema:
        ema_entry = checkpoint.get("ema")
        if isinstance(ema_entry, Mapping) and "module" in ema_entry:
            return ema_entry["module"]
        if "model" in checkpoint:
            return checkpoint["model"]
        raise CheckpointCompatibilityError(
            "Checkpoint contains neither an 'ema.module' state nor a 'model' "
            f"state; cannot select model weights. "
            f"Available keys: {sorted(checkpoint.keys())}"
        )
    if "model" not in checkpoint:
        raise CheckpointCompatibilityError(
            f"prefer_ema=False but checkpoint has no 'model' state. "
            f"Available keys: {sorted(checkpoint.keys())}"
        )
    return checkpoint["model"]


def _strip_module_prefix(state: Mapping[str, Any]) -> dict[str, Any]:
    """Return a new dict with a leading ``module.`` removed from each key.

    Keys without the prefix are passed through unchanged. A new dict is always
    returned so the caller may mutate it freely without touching the original
    checkpoint.
    """
    normalized: dict[str, Any] = {}
    for key, value in state.items():
        if key.startswith(_MODULE_PREFIX):
            normalized[key[len(_MODULE_PREFIX):]] = value
        else:
            normalized[key] = value
    return normalized
