"""rep2 NaN 诊断 runner 单元测试（纯 CPU）。"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest  # noqa: E402 - 模块级导入（Task 7 skipif 装饰器依赖）
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402


def _make_param(name, shape=(4,), *, grad=None):
    if grad is not None:
        shape = (
            grad.shape
            if isinstance(grad, torch.Tensor)
            else torch.tensor(grad).shape
        )
    p = nn.Parameter(torch.randn(shape))
    if grad is not None:
        p.grad = (
            grad.clone().detach()
            if isinstance(grad, torch.Tensor)
            else torch.tensor(grad, dtype=torch.float32)
        )
    return p


class TestTensorStats:

    def test_finite_tensor(self):
        from tool_diagnose_rep2_nan import tensor_stats

        t = torch.tensor([1.0, -2.0, 3.5])
        s = tensor_stats(t)
        assert s["finite"] == 3
        assert s["nan"] == 0
        assert s["pos_inf"] == 0
        assert s["neg_inf"] == 0
        assert s["min"] == -2.0
        assert s["max"] == 3.5
        assert s["absmax"] == 3.5

    def test_mixed_nonfinite(self):
        from tool_diagnose_rep2_nan import tensor_stats

        t = torch.tensor([float("nan"), float("inf"), float("-inf"), 0.0, 5.0])
        s = tensor_stats(t)
        assert s["finite"] == 2
        assert s["nan"] == 1
        assert s["pos_inf"] == 1
        assert s["neg_inf"] == 1
        assert s["min"] == 0.0
        assert s["max"] == 5.0
        assert s["absmax"] == 5.0

    def test_all_nan_has_none_extremes(self):
        from tool_diagnose_rep2_nan import tensor_stats

        t = torch.tensor([float("nan"), float("nan")])
        s = tensor_stats(t)
        assert s["finite"] == 0
        assert s["nan"] == 2
        assert s["min"] is None
        assert s["max"] is None
        assert s["absmax"] is None

    def test_empty_tensor(self):
        from tool_diagnose_rep2_nan import tensor_stats

        s = tensor_stats(torch.empty(0))
        assert s["finite"] == 0
        assert s["min"] is None


class TestScanGradients:

    def test_reports_all_nonfinite_params_not_first_only(self):
        from tool_diagnose_rep2_nan import scan_gradients

        model = nn.Module()
        model.register_parameter("p0", _make_param("p0", grad=[1.0, 2.0, 3.0, 4.0]))
        model.register_parameter("p1", _make_param("p1", grad=[float("nan"), 1.0, 2.0, 3.0]))
        model.register_parameter("p2", _make_param("p2", grad=[float("inf"), 1.0, 2.0, 3.0]))
        norm, anomalies = scan_gradients(model)
        assert len(anomalies) == 2
        names = {a["name"] for a in anomalies}
        assert names == {"p1", "p2"}
        assert all(a["nan"] + a["pos_inf"] + a["neg_inf"] > 0 for a in anomalies)
        assert norm > 0.0

    def test_skips_params_without_grad(self):
        from tool_diagnose_rep2_nan import scan_gradients

        model = nn.Module()
        model.register_parameter("a", _make_param("a", grad=[1.0, 2.0]))
        model.register_parameter("b", _make_param("b", grad=None))
        norm, anomalies = scan_gradients(model)
        assert anomalies == []
        assert norm == float(1.0**2 + 2.0**2) ** 0.5

    def test_empty_grads_returns_zero(self):
        from tool_diagnose_rep2_nan import scan_gradients

        model = nn.Module()
        model.register_parameter("a", _make_param("a", grad=None))
        norm, anomalies = scan_gradients(model)
        assert anomalies == []
        assert norm == 0.0

    def test_finite_grad_reports_zero_anomalies(self):
        from tool_diagnose_rep2_nan import scan_gradients

        model = nn.Module()
        model.register_parameter("a", _make_param("a", grad=[1.0, 2.0]))
        norm, anomalies = scan_gradients(model)
        assert anomalies == []
        assert norm > 0.0


class TestCheckpointInspect:

    def test_full_checkpoint(self):
        from tool_diagnose_rep2_nan import checkpoint_inspect

        state = {
            "date": "2026-08-10T00:00:00",
            "last_epoch": 87,
            "model": {"backbone.sta.stem.0.weight": torch.zeros(2, 2)},
            "ema": {"module": {}, "updates": 100},
            "optimizer": {"state": {}, "param_groups": []},
        }
        r = checkpoint_inspect(state)
        assert r["kind"] == "full"
        assert r["last_epoch"] == 87
        assert r["has_model"] is True
        assert r["has_ema"] is True
        assert r["has_optimizer"] is True

    def test_weights_only_with_model_key(self):
        from tool_diagnose_rep2_nan import checkpoint_inspect

        state = {"model": {"a": torch.zeros(1)}}
        r = checkpoint_inspect(state)
        assert r["kind"] == "weights_only"
        assert r["has_optimizer"] is False
        assert r["last_epoch"] is None

    def test_weights_only_ema_module(self):
        from tool_diagnose_rep2_nan import checkpoint_inspect

        state = {"ema": {"module": {"a": torch.zeros(1)}, "updates": 5}}
        r = checkpoint_inspect(state)
        assert r["kind"] == "weights_only"
        assert r["has_model"] is False
        assert r["has_ema"] is True

    def test_weights_only_bare_state_dict(self):
        from tool_diagnose_rep2_nan import checkpoint_inspect

        state = {"backbone.sta.stem.0.weight": torch.zeros(2, 2)}
        r = checkpoint_inspect(state)
        assert r["kind"] == "weights_only"

    def test_invalid_empty_dict(self):
        from tool_diagnose_rep2_nan import checkpoint_inspect

        r = checkpoint_inspect({})
        assert r["kind"] == "invalid"

    def test_invalid_non_dict_value(self):
        from tool_diagnose_rep2_nan import checkpoint_inspect

        r = checkpoint_inspect({"model": "not-a-dict"})
        assert r["kind"] == "invalid"


class TestEnsureOutputDir:

    def test_new_dir_created(self, tmp_path):
        from tool_diagnose_rep2_nan import ensure_output_dir

        d = tmp_path / "out"
        out = ensure_output_dir(d, overwrite=False)
        assert out == d and d.is_dir()

    def test_empty_existing_dir_allowed(self, tmp_path):
        from tool_diagnose_rep2_nan import ensure_output_dir

        d = tmp_path / "out"
        d.mkdir()
        ensure_output_dir(d, overwrite=False)  # 不抛异常

    def test_nonempty_refused_without_overwrite(self, tmp_path):
        import pytest
        from tool_diagnose_rep2_nan import ensure_output_dir

        d = tmp_path / "out"
        d.mkdir()
        (d / "events.jsonl").write_text("{}")
        with pytest.raises(FileExistsError):
            ensure_output_dir(d, overwrite=False)

    def test_overwrite_clears_dir(self, tmp_path):
        from tool_diagnose_rep2_nan import ensure_output_dir

        d = tmp_path / "out"
        d.mkdir()
        (d / "events.jsonl").write_text("{}")
        ensure_output_dir(d, overwrite=True)
        assert list(d.iterdir()) == []
