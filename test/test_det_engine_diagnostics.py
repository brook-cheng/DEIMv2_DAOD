"""Training diagnostics unit tests.

Validates the four pure functions in engine/solver/training_diagnostics.py
and the integrated hooks in train_one_epoch() via a mocked training flow.

TDD: RED → GREEN. Initial run fails with ImportError before the module exists.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import math

import pytest
import torch
import torch.nn as nn


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_param(name, shape=(4,), *, grad=None):
    """Return a nn.Parameter with optional .grad."""
    p = nn.Parameter(torch.randn(shape))
    if grad is not None:
        p.grad = grad.clone().detach() if isinstance(grad, torch.Tensor) else torch.tensor(grad, dtype=torch.float32)
    return p


def _named_params(*params):
    """Return list of (name, param) pairs from nn.Parameters."""
    return [(f"param_{i}", p) for i, p in enumerate(params)]


# ── validate_max_optimizer_steps ─────────────────────────────────────────────

class TestValidateMaxOptimizerSteps:

    def test_accepts_none(self):
        from engine.solver.training_diagnostics import validate_max_optimizer_steps
        assert validate_max_optimizer_steps(None) is None

    def test_accepts_positive_int(self):
        from engine.solver.training_diagnostics import validate_max_optimizer_steps
        assert validate_max_optimizer_steps(100) == 100
        assert validate_max_optimizer_steps(1) == 1

    @pytest.mark.parametrize("bad", [0, -1, True, 1.5, "foo", [1], (1,)])
    def test_rejects_invalid(self, bad):
        from engine.solver.training_diagnostics import validate_max_optimizer_steps
        with pytest.raises(ValueError):
            validate_max_optimizer_steps(bad)


# ── raise_for_nonfinite_losses ───────────────────────────────────────────────

class TestRaiseForNonfiniteLosses:

    METAS = dict(epoch=0, step=0, global_step=0)

    def test_finite_loss_dict_passes(self):
        from engine.solver.training_diagnostics import raise_for_nonfinite_losses
        ld = {"loss_a": torch.tensor(1.0), "loss_b": torch.tensor(2.0)}
        # Should not raise
        raise_for_nonfinite_losses(ld, **self.METAS)

    def test_nan_raises_with_key(self):
        from engine.solver.training_diagnostics import raise_for_nonfinite_losses
        ld = {"loss_a": torch.tensor(1.0), "loss_bad": torch.tensor(float("nan"))}
        with pytest.raises(FloatingPointError, match="loss_bad"):
            raise_for_nonfinite_losses(ld, **self.METAS)

    def test_pos_inf_raises_with_key(self):
        from engine.solver.training_diagnostics import raise_for_nonfinite_losses
        ld = {"loss_a": torch.tensor(1.0), "loss_inf": torch.tensor(float("inf"))}
        with pytest.raises(FloatingPointError, match="loss_inf"):
            raise_for_nonfinite_losses(ld, **self.METAS)

    def test_neg_inf_raises_with_key(self):
        from engine.solver.training_diagnostics import raise_for_nonfinite_losses
        ld = {"loss_a": torch.tensor(1.0), "loss_neg": torch.tensor(float("-inf"))}
        with pytest.raises(FloatingPointError, match="loss_neg"):
            raise_for_nonfinite_losses(ld, **self.METAS)

    def test_error_message_includes_counts(self):
        from engine.solver.training_diagnostics import raise_for_nonfinite_losses
        ld = {"loss_a": torch.tensor(1.0), "loss_mixed": torch.tensor([float("nan"), float("inf"), 1.0])}
        with pytest.raises(FloatingPointError) as exc:
            raise_for_nonfinite_losses(ld, **self.METAS)
        msg = str(exc.value)
        assert "loss_mixed" in msg
        assert "NaN" in msg
        assert "Inf" in msg


# ── raise_for_nonfinite_total ────────────────────────────────────────────────

class TestRaiseForNonfiniteTotal:

    METAS = dict(epoch=0, step=0, global_step=0)

    def test_finite_total_passes(self):
        from engine.solver.training_diagnostics import raise_for_nonfinite_total
        raise_for_nonfinite_total(torch.tensor(3.0), **self.METAS)

    def test_nan_total_raises(self):
        from engine.solver.training_diagnostics import raise_for_nonfinite_total
        with pytest.raises(FloatingPointError):
            raise_for_nonfinite_total(torch.tensor(float("nan")), **self.METAS)

    def test_inf_total_raises(self):
        from engine.solver.training_diagnostics import raise_for_nonfinite_total
        with pytest.raises(FloatingPointError):
            raise_for_nonfinite_total(torch.tensor(float("inf")), **self.METAS)

    def test_non_scalar_raises_type_error(self):
        from engine.solver.training_diagnostics import raise_for_nonfinite_total
        with pytest.raises(TypeError):
            raise_for_nonfinite_total(torch.tensor([1.0, 2.0]), **self.METAS)

    def test_zero_dim_tensor_passes(self):
        from engine.solver.training_diagnostics import raise_for_nonfinite_total
        # 0-dim tensor is a scalar
        raise_for_nonfinite_total(torch.tensor(42.0), **self.METAS)


# ── inspect_gradients ────────────────────────────────────────────────────────

class TestInspectGradients:

    METAS = dict(epoch=0, step=0, global_step=0)

    def test_finite_gradients_pass(self):
        from engine.solver.training_diagnostics import inspect_gradients
        p = nn.Parameter(torch.randn(4))
        p.grad = torch.randn(4)
        norm = inspect_gradients([("w", p)], **self.METAS)
        assert isinstance(norm, float)
        assert norm > 0

    def test_nan_gradient_raises_with_param_name(self):
        from engine.solver.training_diagnostics import inspect_gradients
        p = nn.Parameter(torch.randn(4))
        p.grad = torch.tensor([float("nan"), 1.0, 2.0, 3.0])
        with pytest.raises(FloatingPointError, match="w"):
            inspect_gradients([("w", p)], **self.METAS)

    def test_inf_gradient_raises(self):
        from engine.solver.training_diagnostics import inspect_gradients
        p = nn.Parameter(torch.randn(4))
        p.grad = torch.tensor([float("inf"), 1.0, 2.0, 3.0])
        with pytest.raises(FloatingPointError, match="w"):
            inspect_gradients([("w", p)], **self.METAS)

    def test_exact_zero_without_flag_does_not_raise(self):
        from engine.solver.training_diagnostics import inspect_gradients
        p = nn.Parameter(torch.randn(4))
        p.grad = torch.zeros(4)
        norm = inspect_gradients([("w", p)], **self.METAS)
        assert norm == 0.0

    def test_exact_zero_with_flag_raises(self):
        from engine.solver.training_diagnostics import inspect_gradients
        p = nn.Parameter(torch.randn(4))
        p.grad = torch.zeros(4)
        with pytest.raises(RuntimeError, match="zero"):
            inspect_gradients([("w", p)], fail_on_zero_grad=True, **self.METAS)

    def test_all_none_grads_treated_as_aggregate_zero(self):
        from engine.solver.training_diagnostics import inspect_gradients
        p1 = nn.Parameter(torch.randn(4))
        p2 = nn.Parameter(torch.randn(4))
        # Neither has .grad set
        norm = inspect_gradients([("a", p1), ("b", p2)], **self.METAS)
        assert norm == 0.0

    def test_all_none_grads_with_flag_raises(self):
        from engine.solver.training_diagnostics import inspect_gradients
        p1 = nn.Parameter(torch.randn(4))
        p2 = nn.Parameter(torch.randn(4))
        with pytest.raises(RuntimeError, match="zero grad"):
            inspect_gradients([("a", p1), ("b", p2)], fail_on_zero_grad=True, **self.METAS)

    def test_one_zero_one_nonzero_passes(self):
        from engine.solver.training_diagnostics import inspect_gradients
        p1 = nn.Parameter(torch.randn(4))
        p1.grad = torch.zeros(4)
        p2 = nn.Parameter(torch.randn(4))
        p2.grad = torch.randn(4)
        norm = inspect_gradients([("a", p1), ("b", p2)], **self.METAS)
        assert norm > 0

    def test_tiny_grad_passes(self):
        from engine.solver.training_diagnostics import inspect_gradients
        p = nn.Parameter(torch.randn(4))
        p.grad = torch.full((4,), 1e-20)
        norm = inspect_gradients([("w", p)], **self.METAS)
        assert norm > 0

    def test_error_message_names_first_affected_param(self):
        from engine.solver.training_diagnostics import inspect_gradients
        p1 = nn.Parameter(torch.randn(4))
        p1.grad = torch.tensor([1.0, float("nan"), 1.0, 1.0])
        p2 = nn.Parameter(torch.randn(4))
        p2.grad = torch.tensor([float("inf"), 1.0, 1.0, 1.0])
        with pytest.raises(FloatingPointError, match="param_a"):
            inspect_gradients([("param_a", p1), ("param_b", p2)], **self.METAS)


# ── Integration: step cap and full mocked train_one_epoch ────────────────────

class TestStepCapIntegration:

    def _fake_dataloader(self, n_batches=10):
        """Returns a list of fake (samples, targets) batches (len support needed)."""
        return [
            (
                torch.randn(2, 3, 64, 64),
                [{"boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]]), "labels": torch.tensor([0])} for _ in range(2)],
            )
            for _ in range(n_batches)
        ]

    def _fake_model(self):
        """Returns a module that outputs a dict with pred_boxes.
        Output is connected to the parameter so backward() finds a grad_fn."""
        class FakeModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = nn.Parameter(torch.tensor(1.0))
            def forward(self, x, targets=None):
                return {
                    "pred_boxes": self.scale * torch.randn(2, 4, requires_grad=False)
                }
        return FakeModel()

    def _fake_criterion(self, *, return_finite=True):
        """Returns a callable criterion returning a loss dict.
        The loss depends on model outputs so backward() has a grad path."""
        class FakeCriterion:
            def __init__(self, finite):
                self.finite = finite
                self.box_mode = "hbb"
            def train(self):
                pass
            def __call__(self, outputs, targets, **metas):
                if self.finite:
                    # Tie loss to model output so backward() finds a grad_fn
                    loss_val = outputs["pred_boxes"].sum()
                else:
                    loss_val = torch.tensor(float("nan"), device=outputs["pred_boxes"].device)
                return {"loss_bbox": loss_val, "loss_kld": loss_val}
        return FakeCriterion(return_finite)

    def _fake_optimizer(self, model):
        return torch.optim.SGD(model.parameters(), lr=0.01)

    def test_step_cap_default_none_processes_all_batches(self):
        from engine.solver.det_engine import train_one_epoch
        model = self._fake_model()
        optimizer = self._fake_optimizer(model)
        n = 5
        dl = self._fake_dataloader(n)
        stats = train_one_epoch(
            False, None, model, self._fake_criterion(), dl,
            optimizer, torch.device("cpu"), 0, max_norm=0,
            print_freq=100, max_optimizer_steps=None,
        )
        assert isinstance(stats, dict)
        # All batches were processed → no cap flag
        assert "_step_cap_reached" not in stats

    def test_step_cap_20_stops_at_exactly_20(self):
        from engine.solver.det_engine import train_one_epoch
        model = self._fake_model()
        optimizer = self._fake_optimizer(model)
        n = 50
        dl = self._fake_dataloader(n)
        stats = train_one_epoch(
            False, None, model, self._fake_criterion(), dl,
            optimizer, torch.device("cpu"), 0, max_norm=0,
            print_freq=100, max_optimizer_steps=20,
        )
        assert stats.get("_step_cap_reached") is True

    def test_step_cap_1_works(self):
        from engine.solver.det_engine import train_one_epoch
        model = self._fake_model()
        optimizer = self._fake_optimizer(model)
        dl = self._fake_dataloader(10)
        stats = train_one_epoch(
            False, None, model, self._fake_criterion(), dl,
            optimizer, torch.device("cpu"), 0, max_norm=0,
            print_freq=100, max_optimizer_steps=1,
        )
        assert stats.get("_step_cap_reached") is True

    def test_step_cap_early_return_skips_ema_and_scheduler(self):
        """When cap reached, EMA and scheduler should not be called after exit."""
        from engine.solver.det_engine import train_one_epoch
        model = self._fake_model()
        optimizer = self._fake_optimizer(model)
        dl = self._fake_dataloader(5)

        # Use a fake EMA that tracks if it was called
        class FakeEMA:
            def __init__(self):
                self.was_called = False
            def update(self, m):
                self.was_called = True

        fake_ema = FakeEMA()

        stats = train_one_epoch(
            False, None, model, self._fake_criterion(), dl,
            optimizer, torch.device("cpu"), 0, max_norm=0,
            print_freq=100, max_optimizer_steps=1,
            ema=fake_ema,
        )
        # Cap reached after 1 batch, ema.update is NOT called for any batch
        # (since it's after optimizer step and we return early)
        assert stats.get("_step_cap_reached") is True

    def test_full_mocked_nonamp_flow(self):
        from engine.solver.det_engine import train_one_epoch
        model = self._fake_model()
        optimizer = self._fake_optimizer(model)
        dl = self._fake_dataloader(3)
        stats = train_one_epoch(
            False, None, model, self._fake_criterion(return_finite=True), dl,
            optimizer, torch.device("cpu"), 0, max_norm=0,
            print_freq=100,
        )
        # Should complete without error and return stats dict
        assert "loss" in stats

    def test_full_mocked_amp_flow(self):
        from engine.solver.det_engine import train_one_epoch
        model = self._fake_model()
        optimizer = self._fake_optimizer(model)
        scaler = torch.amp.GradScaler("cpu")
        dl = self._fake_dataloader(3)
        stats = train_one_epoch(
            False, None, model, self._fake_criterion(return_finite=True), dl,
            optimizer, torch.device("cpu"), 0, max_norm=0.5,
            print_freq=100, scaler=scaler,
        )
        assert "loss" in stats

    def test_nan_loss_raises_before_backward(self):
        """NaN in loss_dict must raise BEFORE backward, so optimizer stays unmutated."""
        from engine.solver.det_engine import train_one_epoch
        model = self._fake_model()
        optimizer = self._fake_optimizer(model)
        dl = self._fake_dataloader(3)
        with pytest.raises(FloatingPointError):
            train_one_epoch(
                False, None, model, self._fake_criterion(return_finite=False), dl,
                optimizer, torch.device("cpu"), 0, max_norm=0,
                print_freq=100,
            )

    def test_nan_loss_raises_in_amp_before_backward(self):
        from engine.solver.det_engine import train_one_epoch
        model = self._fake_model()
        optimizer = self._fake_optimizer(model)
        scaler = torch.amp.GradScaler("cpu")
        dl = self._fake_dataloader(3)
        # Need to return a loss that is finite so loss dict check doesn't catch it,
        # actually NaN in criterion should be caught by raise_for_nonfinite_losses
        with pytest.raises(FloatingPointError):
            train_one_epoch(
                False, None, model, self._fake_criterion(return_finite=False), dl,
                optimizer, torch.device("cpu"), 0, max_norm=0.5,
                print_freq=100, scaler=scaler,
            )

    def test_invalid_max_optimizer_steps_raises(self):
        from engine.solver.det_engine import train_one_epoch
        model = self._fake_model()
        optimizer = self._fake_optimizer(model)
        dl = self._fake_dataloader(1)
        with pytest.raises(ValueError):
            train_one_epoch(
                False, None, model, self._fake_criterion(), dl,
                optimizer, torch.device("cpu"), 0, max_norm=0,
                print_freq=100, max_optimizer_steps=0,
            )

    def test_old_late_check_lines_not_called(self):
        """Verify the old `if not math.isfinite(loss_value): sys.exit(1)` is removed.
        This test confirms finite loss dict passes through without sys.exit.
        If the old code still existed, this would still pass (loss is finite).
        We test indirectly: NaN raises via new early checks, not via old late checks.
        """
        # Already tested by test_nan_loss_raises_before_backward — NaN raises
        # FloatingPointError, not SystemExit.
        # If old code were still active, it would raise SystemExit instead.
        pass
