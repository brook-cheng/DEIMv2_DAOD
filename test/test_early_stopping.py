"""Pure early-stopping config + state-machine tests.

TDD: RED -> GREEN. Fails with ImportError until engine/solver/early_stopping.py
exists. CPU-only; no DDP, no file I/O inside the module under test.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from engine.solver.early_stopping import EarlyStoppingConfig

VALID_BLOCK = {
    "enabled": True,
    "metric": "mAP50_95",
    "mode": "max",
    "min_epochs": 100,
    "patience": 12,
    "min_delta": 0.001,
    "restore_best": True,
}


class TestEarlyStoppingConfig:

    def test_missing_config_disables(self):
        cfg = EarlyStoppingConfig.from_yaml({})
        assert cfg.enabled is False

    def test_empty_block_disables(self):
        cfg = EarlyStoppingConfig.from_yaml({"early_stopping": {}})
        assert cfg.enabled is False

    def test_approved_block_parses(self):
        cfg = EarlyStoppingConfig.from_yaml({"early_stopping": VALID_BLOCK})
        assert cfg.enabled is True
        assert cfg.metric == "mAP50_95"
        assert cfg.mode == "max"
        assert cfg.min_epochs == 100
        assert cfg.patience == 12
        assert cfg.min_delta == 0.001
        assert cfg.restore_best is True

    def test_partial_block_uses_defaults(self):
        cfg = EarlyStoppingConfig.from_yaml({"early_stopping": {"enabled": True}})
        assert cfg.enabled is True
        assert cfg.metric == "mAP50_95"
        assert cfg.mode == "max"
        assert cfg.min_epochs == 100
        assert cfg.patience == 12
        assert cfg.min_delta == 0.001
        assert cfg.restore_best is True

    def test_explicit_enabled_false(self):
        cfg = EarlyStoppingConfig.from_yaml(
            {"early_stopping": {"enabled": False, "patience": 7}}
        )
        assert cfg.enabled is False
        assert cfg.patience == 7

    @pytest.mark.parametrize(
        "block",
        [
            {"mode": "min"},
            {"patience": 0},
            {"patience": -3},
            {"min_epochs": -1},
            {"min_delta": -0.5},
        ],
    )
    def test_invalid_block_raises(self, block):
        with pytest.raises(ValueError):
            EarlyStoppingConfig.from_yaml({"early_stopping": block})


from engine.solver.early_stopping import EarlyStoppingState


class TestEarlyStoppingState:

    def test_first_metric_is_observed_best_and_resets_patience(self):
        st = EarlyStoppingState()
        assert st.update(0.30, epoch=10, min_delta=0.001) is True
        assert st.best_observed_metric == 0.30
        assert st.best_epoch == 10
        assert st.best_significant_metric == 0.30
        assert st.epochs_without_improvement == 0

    def test_strict_improvement_saves_observed_best(self):
        st = EarlyStoppingState()
        st.update(0.30, epoch=10, min_delta=0.001)
        assert st.update(0.32, epoch=11, min_delta=0.001) is True
        assert st.best_observed_metric == 0.32
        assert st.best_epoch == 11

    def test_small_improvement_saves_but_does_not_reset_patience(self):
        st = EarlyStoppingState()
        st.update(0.50, epoch=1, min_delta=0.001)
        st.epochs_without_improvement = 3
        st.best_significant_metric = 0.50
        # +0.0005 < min_delta: observed best updates, patience keeps counting
        assert st.update(0.5005, epoch=2, min_delta=0.001) is True
        assert st.best_observed_metric == 0.5005
        assert st.epochs_without_improvement == 4

    def test_significant_improvement_resets_patience(self):
        st = EarlyStoppingState()
        st.update(0.50, epoch=1, min_delta=0.001)
        st.epochs_without_improvement = 5
        # +0.005 >= min_delta: resets patience. Also a strict observed
        # improvement, so update() still returns True (best.pth should save).
        assert st.update(0.505, epoch=2, min_delta=0.001) is True
        assert st.best_significant_metric == 0.505
        assert st.epochs_without_improvement == 0

    def test_no_improvement_increments_patience(self):
        st = EarlyStoppingState()
        st.update(0.50, epoch=1, min_delta=0.001)
        assert st.update(0.49, epoch=2, min_delta=0.001) is False
        assert st.epochs_without_improvement == 1

    def test_should_stop_respects_min_epochs_floor(self):
        st = EarlyStoppingState()
        st.epochs_without_improvement = 12
        assert st.should_stop(epoch=99, min_epochs=100, patience=12) is False
        assert st.should_stop(epoch=100, min_epochs=100, patience=12) is True

    def test_should_stop_requires_patience_exhausted(self):
        st = EarlyStoppingState()
        st.epochs_without_improvement = 11
        assert st.should_stop(epoch=150, min_epochs=100, patience=12) is False

    def test_reset_patience_preserves_best_values(self):
        st = EarlyStoppingState()
        st.update(0.50, epoch=5, min_delta=0.001)
        st.epochs_without_improvement = 9
        st.reset_patience()
        assert st.epochs_without_improvement == 0
        assert st.best_observed_metric == 0.50
        assert st.best_epoch == 5

    def test_initialize_from_metric_sets_all_fields(self):
        st = EarlyStoppingState()
        st.initialize_from_metric(0.31, epoch=6)
        assert st.best_observed_metric == 0.31
        assert st.best_significant_metric == 0.31
        assert st.best_epoch == 6
        assert st.epochs_without_improvement == 0


class TestEarlyStoppingStatePersistence:

    def test_state_dict_round_trip(self):
        st = EarlyStoppingState()
        st.update(0.42, epoch=17, min_delta=0.001)
        st.epochs_without_improvement = 4
        d = st.state_dict()
        st2 = EarlyStoppingState()
        st2.load_state_dict(d)
        assert st2.best_observed_metric == 0.42
        assert st2.best_significant_metric == 0.42
        assert st2.best_epoch == 17
        assert st2.epochs_without_improvement == 4

    def test_state_dict_keys(self):
        st = EarlyStoppingState()
        d = st.state_dict()
        assert set(d) == {
            "best_observed_metric",
            "best_significant_metric",
            "best_epoch",
            "epochs_without_improvement",
        }
