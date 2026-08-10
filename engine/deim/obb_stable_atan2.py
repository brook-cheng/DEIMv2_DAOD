"""Stable atan2 autograd operator for OBB geometry.

``torch.atan2`` has a first-order backward singularity at ``(y, x) =
(0, 0)``: the derivative ``x / (x^2 + y^2)`` (and ``-y / (x^2 + y^2)``)
blows up when both components collapse to zero. Under BF16/FP16 training
the rep2 decoder can hit this degenerate case while decoding tiny
external-rectangle edges, poisoning every downstream gradient with NaN.

This module provides a private ``torch.autograd.Function`` whose
*forward* is exactly ``torch.atan2(y, x)`` — no input perturbation, no
output clamping, identical angle values to the native op — and whose
*first-order backward* floors the squared radius at ``eps`` before
dividing. FP16/BF16 operands and upstream gradients are lifted to FP32
for the backward arithmetic and cast back to the corresponding input
dtype for the returned gradients.

Only the first-order backward is defined; double-backward is not
supported.
"""

from typing import cast

import torch
from torch import Tensor


class _StableAtan2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y: Tensor, x: Tensor, eps: float) -> Tensor:
        ctx.save_for_backward(y, x)
        ctx.eps = float(eps)
        return torch.atan2(y, x)

    @staticmethod
    def backward(ctx, *grad_outputs: Tensor) -> tuple[Tensor, Tensor, None]:
        y, x = ctx.saved_tensors
        grad_output = grad_outputs[0]
        low_precision = (
            y.dtype in (torch.float16, torch.bfloat16)
            or x.dtype in (torch.float16, torch.bfloat16)
            or grad_output.dtype in (torch.float16, torch.bfloat16)
        )
        calc_dtype = torch.float32 if low_precision else y.dtype
        y_calc = y.to(calc_dtype)
        x_calc = x.to(calc_dtype)
        grad_calc = grad_output.to(calc_dtype)
        r2_safe = (x_calc.square() + y_calc.square()).clamp_min(ctx.eps)
        grad_y = grad_calc * x_calc / r2_safe
        grad_x = -grad_calc * y_calc / r2_safe
        return grad_y.to(y.dtype), grad_x.to(x.dtype), None


def stable_atan2(y: Tensor, x: Tensor, eps: float) -> Tensor:
    return cast(Tensor, _StableAtan2.apply(y, x, eps))
