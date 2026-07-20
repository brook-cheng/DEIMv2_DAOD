"""Training diagnostics: pure validation functions for loss/gradient sanity.

All functions are pure — no mutation, no side effects, no model/optimizer calls.
They exist to catch training anomalies early, before backward/step mutation.
"""

from __future__ import annotations

from typing import Iterator

import torch


def validate_max_optimizer_steps(value) -> int | None:
    """Return validated int or None. Raise ValueError for invalid types/values.

    Accepts:
        None          → None (unlimited)
        positive int  → int (cap)

    Rejects:
        0, -1, bool, float, str, list, tuple, etc.
    """
    if value is None:
        return None

    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(
            f"max_optimizer_steps must be None or int, got {type(value).__name__}: {value!r}"
        )
    if value < 1:
        raise ValueError(
            f"max_optimizer_steps must be >= 1 or None, got {value}"
        )
    return value


def raise_for_nonfinite_losses(
    loss_dict: dict[str, torch.Tensor],
    *,
    epoch: int,
    step: int,
    global_step: int,
) -> None:
    """Raise FloatingPointError if any loss component is non-finite.

    Error message includes: key, value dtype, device, NaN/+Inf/-Inf counts.
    Called BEFORE loss aggregation and backward, so optimizer never mutates.
    """
    for key, tensor in loss_dict.items():
        if torch.isfinite(tensor).all():
            continue

        flat = tensor.detach().float().flatten()
        nan_count = torch.isnan(flat).sum().item()
        pos_inf_count = torch.isposinf(flat).sum().item()
        neg_inf_count = torch.isneginf(flat).sum().item()
        raise FloatingPointError(
            f"Non-finite loss component at epoch={epoch} step={step} global_step={global_step}: "
            f"key='{key}' dtype={tensor.dtype} device={tensor.device} "
            f"shape={tuple(tensor.shape)} "
            f"NaN={int(nan_count)} +Inf={int(pos_inf_count)} -Inf={int(neg_inf_count)}"
        )


def raise_for_nonfinite_total(
    total,
    *,
    epoch: int,
    step: int,
    global_step: int,
) -> None:
    """Raise FloatingPointError if total loss is non-finite; TypeError if non-scalar.

    Called BEFORE .backward(), so the model grads are not yet changed.
    """
    if not isinstance(total, torch.Tensor):
        raise TypeError(
            f"Loss total must be a torch.Tensor, got {type(total).__name__}"
        )
    if total.ndim != 0:
        raise TypeError(
            f"Loss total must be scalar (0-d tensor), got ndim={total.ndim} shape={tuple(total.shape)}"
        )

    if torch.isfinite(total):
        return

    raise FloatingPointError(
        f"Non-finite loss total at epoch={epoch} step={step} global_step={global_step}: "
        f"value={total.item()} dtype={total.dtype} device={total.device}"
    )


def inspect_gradients(
    named_parameters: Iterator[tuple[str, torch.nn.Parameter]],
    *,
    fail_on_zero_grad: bool = False,
    epoch: int,
    step: int,
    global_step: int,
) -> float:
    """Check gradient finiteness (always fail on NaN/Inf). Optionally fail on zero aggregate.

    Returns the finite aggregate gradient norm (sqrt of sum of squared gradient values
    across ALL parameters with .grad; .grad=None treated as zero contribution).

    Uses ``torch.isfinite`` for the finiteness check (Catches NaN and Inf together).

    Raises:
        FloatingPointError: NaN or Inf gradient detected — names the first affected param.
        RuntimeError: ``fail_on_zero_grad=True`` and the aggregate gradient norm is 0
            (which includes the case where ALL params have ``.grad is None``).
    """
    aggregate_sq = 0.0

    all_none = True
    for name, param in named_parameters:
        if param.grad is None:
            continue
        all_none = False

        grad = param.grad
        if grad.is_sparse:
            grad = grad.coalesce()
            values = grad.values()
        else:
            values = grad

        if not torch.isfinite(values).all():
            nan_count = torch.isnan(values).sum().item()
            inf_count = torch.isinf(values).sum().item()
            raise FloatingPointError(
                f"Non-finite gradient at epoch={epoch} step={step} global_step={global_step}: "
                f"param='{name}' dtype={values.dtype} device={values.device} "
                f"shape={tuple(values.shape)} "
                f"NaN={int(nan_count)} Inf={int(inf_count)}"
            )

        aggregate_sq += values.float().pow(2).sum().item()

    aggregate_norm = aggregate_sq ** 0.5

    if fail_on_zero_grad and aggregate_norm == 0.0:
        if all_none:
            msg = (
                f"All parameters have no gradients (aggregate zero grad) "
                f"at epoch={epoch} step={step} global_step={global_step}"
            )
        else:
            msg = (
                f"Aggregate gradient norm is zero "
                f"at epoch={epoch} step={step} global_step={global_step}"
            )
        raise RuntimeError(msg)

    return float(aggregate_norm)
