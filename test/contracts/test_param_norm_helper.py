"""param_norm helper contract: only computed for comet logging, correct math.

Before the fix, ``train_one_epoch`` recomputed the L2 norm of every parameter
(one GPU→CPU ``.item()`` sync PER parameter tensor) on EVERY iteration,
unconditionally — even though the value is only consumed by the comet batch
log on rank0 every 50 steps. Under CPU/PCIe contention (the 20260820_182623
epochs 2-12 slowdown wave) that dead cost amplifies per-iteration latency.

Contract:
- ``compute_param_norm(model)`` returns sqrt(sum(param.norm(2)^2)) exactly.
- The call site gates it behind ``comet_exp and is_main_process() and
  global_step % 50 == 0`` (verified structurally in ``train_one_epoch``).

Run:
    pytest test/contracts/test_param_norm_helper.py -v
"""

import pytest
import torch

from engine.solver.det_engine import compute_param_norm


def test_compute_param_norm_matches_definition():
    model = torch.nn.Linear(2, 3, bias=False)
    with torch.no_grad():
        model.weight.fill_(2.0)

    expected = (model.weight.numel() * 2.0**2) ** 0.5

    assert abs(compute_param_norm(model) - expected) < 1e-6


def test_compute_param_norm_multi_tensor():
    model = torch.nn.Sequential(
        torch.nn.Linear(2, 3, bias=False),
        torch.nn.Linear(3, 1, bias=True),
    )
    with torch.no_grad():
        for p in model.parameters():
            p.fill_(1.5)

    expected = sum(p.numel() for p in model.parameters()) * 1.5**2
    expected = expected**0.5

    assert abs(compute_param_norm(model) - expected) < 1e-6


def test_compute_param_norm_no_trainable_params():
    model = torch.nn.Identity()
    assert compute_param_norm(model) == 0.0
