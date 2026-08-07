"""DetSolver early-stopping integration tests (mocked training flow).

Harness: DetSolver is built via __new__ (skips heavy _setup), the real
train()/load_resume_state are shadowed where noted, and det_solver module
functions (stats, train_one_epoch, evaluate) are monkeypatched. Mirrors the
mocking style of test_det_engine_diagnostics.py. CPU-only.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from engine.solver import det_solver
from engine.solver.det_solver import DetSolver
from engine.solver.early_stopping import EarlyStoppingState


def _es_yaml(**overrides):
    es = {
        "enabled": True,
        "metric": "mAP50_95",
        "mode": "max",
        "min_epochs": 2,
        "patience": 2,
        "min_delta": 0.001,
        "restore_best": True,
    }
    es.update(overrides)
    return es


def _make_solver(tmp_path, *, epoches=3, stop_epoch=999, es=None, ema=False):
    """Build a DetSolver with a stub graph; train() is shadowed to a no-op.

    es=None -> yaml_cfg has no early_stopping key (legacy behavior).
    ema=True -> self.ema is a stub with .module/.decay (needed by the
    stage-transition test).
    """
    model = nn.Linear(2, 2)
    yaml_cfg = {
        "max_optimizer_steps": None,
        "fail_on_zero_grad": False,
        "nan_max_events": 10,
        "amp_dtype": None,
        "obb_geometry_fp32": False,
    }
    if es is not None:
        yaml_cfg["early_stopping"] = es

    cfg = SimpleNamespace(
        lrsheduler=None,
        epoches=epoches,
        lr_gamma=0.1,
        warmup_iter=0,
        flat_epoch=1,
        no_aug_epoch=0,
        clip_max_norm=0.0,
        print_freq=1000,
        checkpoint_freq=1000,
        yaml_cfg=yaml_cfg,
        device="cpu",
        resume=None,
        use_ema=False,
        use_amp=False,
    )

    solver = DetSolver.__new__(DetSolver)
    solver.cfg = cfg
    solver.model = model
    solver.ema = (
        SimpleNamespace(module=model, decay=0.9999) if ema else None
    )
    solver.criterion = None
    solver.postprocessor = SimpleNamespace(box_mode="obb")
    solver.scaler = None
    solver.device = torch.device("cpu")
    solver.last_epoch = -1
    solver.output_dir = tmp_path
    solver.writer = None
    solver.optimizer = None
    solver.lr_scheduler = SimpleNamespace(step=lambda: None)
    solver.lr_warmup_scheduler = SimpleNamespace(finished=lambda: True)
    solver.self_lr_scheduler = False
    solver.train_dataloader = SimpleNamespace(
        set_epoch=lambda e: None,
        collate_fn=SimpleNamespace(stop_epoch=stop_epoch, ema_restart_decay=0.9999),
        sampler=SimpleNamespace(set_epoch=lambda e: None),
    )
    solver.val_dataloader = None
    solver.evaluator = None
    solver.train = lambda: None
    solver._comet_experiment = None
    return solver


def _run_fit(solver, metrics, monkeypatch, *, step_cap=False, restore_metric=None):
    """Monkeypatch engine fns and run fit().

    metrics: one value per training epoch (popped left-to-right).
    restore_metric: value returned by the final restore re-evaluation; defaults
    to 0.0 when the metrics list is exhausted.
    """
    monkeypatch.setattr(det_solver, "stats", lambda cfg: (0, ""))
    monkeypatch.setattr(
        det_solver,
        "train_one_epoch",
        lambda *a, **k: {"_step_cap_reached": True} if step_cap else {},
    )

    def fake_evaluate(*args, **kwargs):
        if metrics:
            return ({"mAP50_95": metrics.pop(0)}, None)
        return ({"mAP50_95": restore_metric if restore_metric is not None else 0.0}, None)

    monkeypatch.setattr(det_solver, "evaluate", fake_evaluate)
    solver.fit()


class TestDisabledBehavior:

    def test_missing_config_preserves_legacy_behavior(self, tmp_path, monkeypatch):
        """Design test 1: no ES key -> no best.pth, no meta, last.pth still written."""
        solver = _make_solver(tmp_path, epoches=3, es=None)
        _run_fit(solver, [0.10, 0.20, 0.30], monkeypatch)
        assert not (tmp_path / "best.pth").exists()
        assert not (tmp_path / "final_run_meta.json").exists()
        assert (tmp_path / "last.pth").exists()

    def test_explicit_disabled_block_is_legacy(self, tmp_path, monkeypatch):
        solver = _make_solver(
            tmp_path, epoches=3, es={"enabled": False}
        )
        _run_fit(solver, [0.10, 0.20, 0.30], monkeypatch)
        assert not (tmp_path / "best.pth").exists()
        assert not (tmp_path / "final_run_meta.json").exists()


def _load_ckpt(tmp_path, name):
    return torch.load(tmp_path / name, map_location="cpu", weights_only=False)


class TestBestCheckpointSaving:

    def test_strict_improvement_saves_best_pth(self, tmp_path, monkeypatch):
        """Design test 2: strict improvement writes a complete best.pth."""
        solver = _make_solver(tmp_path, epoches=3, es=_es_yaml())
        _run_fit(solver, [0.30, 0.32, 0.31], monkeypatch)
        best = _load_ckpt(tmp_path, "best.pth")
        es_state = best["early_stopping"]
        assert es_state["best_epoch"] == 1
        assert es_state["best_observed_metric"] == pytest.approx(0.32)
        assert "model" in best
        assert "last_epoch" in best

    def test_small_improvement_saves_best_but_keeps_patience(self, tmp_path, monkeypatch):
        """Design test 3 (integration): sub-min_delta gain still updates best.pth."""
        solver = _make_solver(
            tmp_path, epoches=4, es=_es_yaml(min_epochs=0, patience=10)
        )
        # best 0.30 at ep0; +0.0005 at ep1 is < min_delta
        _run_fit(solver, [0.30, 0.3005, 0.29, 0.28], monkeypatch)
        best = _load_ckpt(tmp_path, "best.pth")
        assert best["early_stopping"]["best_epoch"] == 1
        assert best["early_stopping"]["best_observed_metric"] == pytest.approx(0.3005)
        # patience kept counting: ep1 no reset -> ep2, ep3 increments
        last = _load_ckpt(tmp_path, "last.pth")
        assert last["early_stopping"]["epochs_without_improvement"] == 3


class TestPatienceStopping:

    def test_patience_exhausted_stops_after_min_epochs(self, tmp_path, monkeypatch):
        """Design test 5 (integration): stop fires at min_epochs, not before."""
        solver = _make_solver(
            tmp_path, epoches=10, es=_es_yaml(min_epochs=2, patience=2)
        )
        _run_fit(
            solver, [0.30, 0.29, 0.28, 0.27, 0.26, 0.25, 0.24, 0.23, 0.22, 0.21],
            monkeypatch,
        )
        last = _load_ckpt(tmp_path, "last.pth")
        # ep0 improves; eps 1,2 no improvement -> counter hits 2 at epoch 2
        assert last["last_epoch"] == 2
        assert last["early_stopping"]["epochs_without_improvement"] == 2

    def test_no_stop_before_min_epochs(self, tmp_path, monkeypatch):
        """min_epochs=5 blocks the stop even though patience reaches 2 early."""
        solver = _make_solver(
            tmp_path, epoches=10, es=_es_yaml(min_epochs=5, patience=2)
        )
        # ep0 improves; flat afterwards -> counter hits patience at ep2 but
        # 5-epoch floor holds until epoch 5
        _run_fit(solver, [0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30], monkeypatch)
        last = _load_ckpt(tmp_path, "last.pth")
        assert last["last_epoch"] == 5
        assert last["early_stopping"]["epochs_without_improvement"] == 5


class TestCheckpointOrdering:

    def test_last_pth_is_true_final_epoch(self, tmp_path, monkeypatch):
        """Design test 12 (partial): last.pth keeps the real final epoch."""
        solver = _make_solver(tmp_path, epoches=3, es=_es_yaml())
        _run_fit(solver, [0.30, 0.32, 0.31], monkeypatch)
        last = _load_ckpt(tmp_path, "last.pth")
        assert last["last_epoch"] == 2
        assert last["early_stopping"]["best_epoch"] == 1  # patience state current
        best = _load_ckpt(tmp_path, "best.pth")
        assert best["last_epoch"] == 1


class TestDDPBroadcast:

    def _init_solver(self, tmp_path):
        solver = _make_solver(tmp_path, es=_es_yaml())
        solver._init_early_stopping()
        return solver

    def test_main_rank_broadcasts_state_and_stop(self, tmp_path, monkeypatch):
        """Design test 6 (rank 0 side): state dict + stop flag are broadcast."""
        solver = self._init_solver(tmp_path)
        solver.early_stopping.update(0.5, epoch=1, min_delta=0.001)
        captured = {}

        monkeypatch.setattr(
            det_solver.dist_utils, "is_dist_available_and_initialized", lambda: True
        )
        monkeypatch.setattr(det_solver.dist_utils, "is_main_process", lambda: True)
        monkeypatch.setattr(
            torch.distributed,
            "broadcast_object_list",
            lambda payload, src=0: captured.update(payload=payload, src=src),
        )

        stop = solver._sync_early_stopping(True)
        assert stop is True
        assert captured["src"] == 0
        assert captured["payload"][1] is True
        assert captured["payload"][0]["best_epoch"] == 1

    def test_non_main_rank_receives_state_and_stop(self, tmp_path, monkeypatch):
        """Design test 6 (other ranks): receive state and stop from rank 0."""
        solver = self._init_solver(tmp_path)
        state = {
            "best_observed_metric": 0.5,
            "best_significant_metric": 0.5,
            "best_epoch": 7,
            "epochs_without_improvement": 3,
        }

        monkeypatch.setattr(
            det_solver.dist_utils, "is_dist_available_and_initialized", lambda: True
        )
        monkeypatch.setattr(det_solver.dist_utils, "is_main_process", lambda: False)
        monkeypatch.setattr(
            torch.distributed,
            "broadcast_object_list",
            lambda payload, src=0: payload.__setitem__(0, state)
            or payload.__setitem__(1, True),
        )

        stop = solver._sync_early_stopping(False)
        assert stop is True
        assert solver.early_stopping.best_epoch == 7
        assert solver.early_stopping.epochs_without_improvement == 3

    def test_non_distributed_is_passthrough(self, tmp_path, monkeypatch):
        solver = self._init_solver(tmp_path)
        assert solver._sync_early_stopping(True) is True
        assert solver._sync_early_stopping(False) is False


def _read_meta(tmp_path):
    return json.loads((tmp_path / "final_run_meta.json").read_text())


class TestFinalizeRestore:

    def test_max_epochs_restores_and_validates_best_pth(self, tmp_path, monkeypatch):
        """Design tests 7 + 11: epoch-limit exit restores best.pth, restored
        eval matches the saved best within tolerance."""
        solver = _make_solver(tmp_path, epoches=3, es=_es_yaml())
        _run_fit(
            solver, [0.30, 0.32, 0.31], monkeypatch, restore_metric=0.32
        )
        meta = _read_meta(tmp_path)
        assert meta["stop_reason"] == "max_epochs"
        assert meta["stop_epoch"] == 2
        assert meta["best_epoch"] == 1
        assert meta["best_mAP50_95"] == pytest.approx(0.32)
        assert meta["restored_mAP50_95"] == pytest.approx(0.32)
        assert meta["restore_match"] is True
        assert meta["restore_skipped"] is False
        # last.pth is NOT overwritten by the restored best state
        last = _load_ckpt(tmp_path, "last.pth")
        assert last["last_epoch"] == 2

    def test_early_stop_restores_best(self, tmp_path, monkeypatch):
        solver = _make_solver(tmp_path, epoches=10, es=_es_yaml())
        # Exactly 3 training metrics: consumed by epochs 0-2 before the stop at
        # epoch 2; the finalize restore-eval then falls through to
        # restore_metric=0.30 instead of a stale training value.
        _run_fit(
            solver, [0.30, 0.29, 0.28],
            monkeypatch,
            restore_metric=0.30,
        )
        meta = _read_meta(tmp_path)
        assert meta["stop_reason"] == "early_stopping"
        assert meta["stop_epoch"] == 2
        assert meta["best_epoch"] == 0
        assert meta["best_mAP50_95"] == pytest.approx(0.30)
        assert meta["restored_mAP50_95"] == pytest.approx(0.30)

    def test_diagnostic_exit_does_not_restore(self, tmp_path, monkeypatch):
        """Design test 13: max_optimizer_steps exit skips best.pth restore."""
        # A best.pth exists, but diagnostic exit must not load it.
        torch.save(
            {
                "last_epoch": 0,
                "model": nn.Linear(2, 2).state_dict(),
                "early_stopping": {
                    "best_observed_metric": 0.99,
                    "best_significant_metric": 0.99,
                    "best_epoch": 0,
                    "epochs_without_improvement": 0,
                },
            },
            tmp_path / "best.pth",
        )
        solver = _make_solver(tmp_path, epoches=3, es=_es_yaml())
        _run_fit(solver, [0.30], monkeypatch, step_cap=True)
        meta = _read_meta(tmp_path)
        assert meta["stop_reason"] == "diagnostic"
        assert meta["restore_skipped"] is True
        assert meta["restored_mAP50_95"] is None
        assert meta["best_mAP50_95"] is None  # -inf normalized to None


class TestResumeCompat:

    def test_new_last_pth_restores_full_es_state(self, tmp_path, monkeypatch):
        """Design test 8: resume from a new last.pth restores the patience window."""
        solver = _make_solver(tmp_path, epoches=3, es=_es_yaml())
        _run_fit(solver, [0.30, 0.32, 0.31], monkeypatch)

        fresh = DetSolver.__new__(DetSolver)
        fresh.model = nn.Linear(2, 2)
        fresh.ema = None
        fresh.early_stopping = EarlyStoppingState()
        fresh.last_epoch = -1
        fresh.load_resume_state(str(tmp_path / "last.pth"))
        assert fresh.last_epoch == 2
        assert fresh.early_stopping.best_epoch == 1
        assert fresh.early_stopping.best_observed_metric == pytest.approx(0.32)
        assert fresh.early_stopping.epochs_without_improvement == 1

    def test_old_checkpoint_inits_from_pre_resume_validation(self, tmp_path, monkeypatch):
        """Design test 9: legacy checkpoint (no ES key) initializes from the
        current EMA validation and preserves it as the global best."""
        solver = _make_solver(tmp_path, epoches=5, es=_es_yaml())
        solver.last_epoch = 3  # resumed run: pre-resume validation will run
        # pre-resume eval -> 0.25 (init), epoch 4 -> 0.24 (no improvement)
        _run_fit(solver, [0.25, 0.24], monkeypatch)
        meta = _read_meta(tmp_path)
        assert meta["best_epoch"] == 3
        assert meta["best_mAP50_95"] == pytest.approx(0.25)


class TestStageTransition:

    def test_stage_transition_resets_patience_preserves_best(self, tmp_path, monkeypatch):
        """Design test 10: patience resets at the stage boundary; the global
        best checkpoint is preserved."""
        solver = _make_solver(
            tmp_path, epoches=4, stop_epoch=1, es=_es_yaml(min_epochs=0, patience=50),
            ema=True,
        )
        orig_init = solver._init_early_stopping

        def seeded_init():
            orig_init()
            solver.early_stopping.best_observed_metric = 0.30
            solver.early_stopping.best_significant_metric = 0.30
            solver.early_stopping.best_epoch = 0
            solver.early_stopping.epochs_without_improvement = 5

        monkeypatch.setattr(solver, "_init_early_stopping", seeded_init)
        # stage rollback loads best_stg1.pth; neutralize it for this unit test
        monkeypatch.setattr(solver, "load_resume_state", lambda path: None)
        _run_fit(solver, [0.30, 0.29, 0.28, 0.27], monkeypatch)
        last = _load_ckpt(tmp_path, "last.pth")
        # without reset, counter would be 9; reset at epoch 1 leaves 3
        assert last["early_stopping"]["epochs_without_improvement"] == 3
        meta = _read_meta(tmp_path)
        assert meta["best_epoch"] == 0
        assert meta["best_mAP50_95"] == pytest.approx(0.30)
