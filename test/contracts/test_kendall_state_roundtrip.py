"""Kendall state persistence contract: recovery must restore learned σ².

Before the recovery feature, ``kendall`` was a LOCAL variable inside
``DetSolver.fit()`` — its ``log_sigma`` and optimizer state were never
serialized into checkpoints, so any resume silently reset the learned loss
weights to their initial values. ``fit()`` now builds ``self.kendall`` /
``self.kendall_optimizer`` BEFORE the recovery load, so the generic
``BaseSolver.state_dict`` / ``load_state_dict`` attribute loop carries them.

Contract:
- ``KendallWeighting`` round-trips ``log_sigma`` (and prior) through
  ``state_dict``/``load_state_dict``.
- ``BaseSolver.state_dict`` includes ``kendall`` + ``kendall_optimizer``
  when present as attributes; ``load_state_dict`` restores them.
- Disabled (``None``) kendall is skipped without error.

Run:
    pytest test/contracts/test_kendall_state_roundtrip.py -v
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.solver._solver import BaseSolver  # noqa: E402
from engine.solver.kendall import KendallWeighting  # noqa: E402


def test_kendall_weighting_state_round_trip():
    kw = KendallWeighting(["loss_mal", "loss_bbox"], init_log_sigma=0.5)
    with torch.no_grad():
        kw.log_sigma += torch.tensor([1.25, -0.75])

    restored = KendallWeighting(["loss_mal", "loss_bbox"], init_log_sigma=0.0)
    restored.load_state_dict(kw.state_dict())

    assert torch.allclose(restored.log_sigma, kw.log_sigma)
    assert torch.allclose(restored.prior, kw.prior)


def _solver_stub(kendall, kendall_optimizer):
    return SimpleNamespace(
        model=torch.nn.Identity(),
        last_epoch=7,
        kendall=kendall,
        kendall_optimizer=kendall_optimizer,
    )


def test_base_solver_state_dict_carries_kendall():
    kw = KendallWeighting(["loss_mal", "loss_bbox"], init_log_sigma=0.3)
    opt = torch.optim.Adam([kw.log_sigma], lr=1e-3)
    with torch.no_grad():
        kw.log_sigma += 0.8

    state = BaseSolver.state_dict(_solver_stub(kw, opt))

    assert "kendall" in state and "kendall_optimizer" in state
    assert torch.allclose(state["kendall"]["log_sigma"], kw.log_sigma.data)


def test_base_solver_load_restores_kendall():
    saved_kw = KendallWeighting(["loss_mal", "loss_bbox"], init_log_sigma=0.3)
    saved_opt = torch.optim.Adam([saved_kw.log_sigma], lr=1e-3)
    with torch.no_grad():
        saved_kw.log_sigma += torch.tensor([1.5, -1.0])
    for _ in range(3):
        saved_opt.step()
    state = BaseSolver.state_dict(_solver_stub(saved_kw, saved_opt))

    fresh_kw = KendallWeighting(["loss_mal", "loss_bbox"], init_log_sigma=0.0)
    fresh_opt = torch.optim.Adam([fresh_kw.log_sigma], lr=1e-3)
    stub = _solver_stub(fresh_kw, fresh_opt)
    BaseSolver.load_state_dict(stub, state)

    assert torch.allclose(fresh_kw.log_sigma, saved_kw.log_sigma)
    assert stub.last_epoch == 7


def test_disabled_kendall_is_skipped_cleanly():
    stub = _solver_stub(None, None)
    state = BaseSolver.state_dict(stub)
    assert "kendall" not in state

    BaseSolver.load_state_dict(stub, state)  # must not raise
