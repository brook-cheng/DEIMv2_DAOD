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


class TestAtan2ZeroProbe:

    def test_forward_finite_backward_nonfinite(self):
        from tool_diagnose_rep2_nan import atan2_zero_probe

        r = atan2_zero_probe("cpu")
        assert r["forward_finite"] is True
        assert r["forward_value"] == 0.0
        assert r["backward_grads_finite"] is False


class TestComputeEdgeStats:

    def test_degenerate_edges_detected(self):
        from tool_diagnose_rep2_nan import compute_edge_stats

        # ext_rect=(0,0,0,0), offsets=(0,0) → v0=v1=v2 → 两退化边 + atan2(0,0)
        rect = torch.zeros(1, 1, 4)
        offs = torch.zeros(1, 1, 2)
        s = compute_edge_stats(rect, offs)
        assert s["n"] == 1
        assert s["edge_ab_zero"] == 1
        assert s["edge_bc_zero"] == 1
        assert s["atan2_zero_inputs"] == 1
        # eps=1e-9 稳定化使退化边长 ≈ sqrt(1e-9)=3.16e-5，非 0
        assert s["w_min"] < 1e-4
        assert s["h_min"] < 1e-4

    def test_normal_rect_no_atan2_zero(self):
        from tool_diagnose_rep2_nan import compute_edge_stats

        # ext_rect=(0,0,2,1), offsets=(0.5,0.25)
        rect = torch.tensor([[[0.0, 0.0, 2.0, 1.0]]])
        offs = torch.tensor([[[0.5, 0.25]]])
        s = compute_edge_stats(rect, offs)
        assert s["edge_ab_zero"] == 0
        assert s["edge_bc_zero"] == 0
        assert s["atan2_zero_inputs"] == 0
        assert s["w_min"] > 1.0
        assert s["h_min"] > 0.5

    def test_returns_scalar_types(self):
        from tool_diagnose_rep2_nan import compute_edge_stats

        rect = torch.tensor([[[0.0, 0.0, 1.0, 1.0]]])
        offs = torch.tensor([[[0.0, 0.0]]])
        s = compute_edge_stats(rect, offs)
        for k in ("n", "edge_ab_zero", "edge_bc_zero", "near_zero_edges", "atan2_zero_inputs"):
            assert isinstance(s[k], int)
        for k in ("w_min", "h_min", "w_dx_absmin", "w_dy_absmin", "eps_min", "eta_min", "eps_max", "eta_max"):
            assert isinstance(s[k], float)


class TestGeometryProbe:

    def test_install_snapshot_uninstall(self):
        from tool_diagnose_rep2_nan import GeometryProbe

        probe = GeometryProbe()
        probe.install()
        try:
            import torch
            # 探针包装的是消费模块绑定（dfine_utils 按名导入），必须经此调用才能命中包装器
            from engine.deim.dfine_utils import external_xyxy_rect_to_oriented_box

            rect = torch.tensor([[[0.0, 0.0, 0.0, 0.0]]])
            offs = torch.tensor([[[0.0, 0.0]]])
            out = external_xyxy_rect_to_oriented_box(rect, offs)
            snap = probe.snapshot()
            assert snap["calls"] >= 1
            assert snap.get("edge_ab_zero", 0) >= 1
            assert out.shape == (1, 1, 5)  # 输出不受影响
        finally:
            probe.uninstall()

    def test_probe_is_noop_after_uninstall(self):
        from tool_diagnose_rep2_nan import GeometryProbe

        probe = GeometryProbe()
        probe.install()
        probe.uninstall()
        import torch
        from engine.deim.dfine_utils import external_xyxy_rect_to_oriented_box

        rect = torch.tensor([[[0.0, 0.0, 1.0, 1.0]]])
        offs = torch.tensor([[[0.25, 0.25]]])
        external_xyxy_rect_to_oriented_box(rect, offs)
        assert probe.snapshot()["calls"] == 0


class TestArtifactWriters:

    def test_write_run_manifest(self, tmp_path):
        import json
        from tool_diagnose_rep2_nan import write_run_manifest

        p = tmp_path / "run_manifest.json"
        write_run_manifest(p, {"a": 1, "b": {"c": [1, 2]}})
        assert json.loads(p.read_text()) == {"a": 1, "b": {"c": [1, 2]}}

    def test_append_event_jsonl(self, tmp_path):
        import json
        from tool_diagnose_rep2_nan import append_event

        p = tmp_path / "events.jsonl"
        append_event(p, {"epoch": 0, "step": 0})
        append_event(p, {"epoch": 0, "step": 1})
        lines = p.read_text().strip().splitlines()
        assert len(lines) == 2
        assert json.loads(lines[1])["step"] == 1

    def test_write_progress(self, tmp_path):
        import json
        from tool_diagnose_rep2_nan import write_progress

        p = tmp_path / "progress.json"
        write_progress(p, {"epoch": 3, "global_step": 42})
        assert json.loads(p.read_text())["global_step"] == 42

    def test_save_failure_moves_tensors_to_cpu(self, tmp_path):
        import json
        from tool_diagnose_rep2_nan import save_failure

        out = tmp_path / "diag"
        paths = save_failure(
            out,
            traceback_text="Traceback (most recent call last):\n  boom",
            failure_summary={"exit_code": 2, "kind": "gradient"},
            trigger_batch={
                "samples": torch.randn(2, 3, 4, 4),
                "targets": [{"boxes": torch.randn(1, 5)}],
            },
            outputs={"pred_boxes": torch.randn(2, 10, 5)},
            losses={"loss_total": torch.tensor(float("nan"))},
            geometry_snapshot={"atan2_zero_inputs": 3},
            gradients_summary={"aggregate_norm": 1.0, "anomalies": []},
            model_state={"m": torch.randn(2)},
            optimizer_state={"state": {}},
        )
        fail_dir = out / "failure"
        assert (fail_dir / "traceback.txt").exists()
        assert (fail_dir / "trigger_batch.pt").exists()
        assert (fail_dir / "outputs.pt").exists()
        assert (fail_dir / "losses.pt").exists()
        assert (fail_dir / "model_state.pt").exists()
        assert (fail_dir / "optimizer_state.pt").exists()
        fs = json.loads((fail_dir / "failure_summary.json").read_text())
        assert fs["kind"] == "gradient"
        tb = torch.load(fail_dir / "trigger_batch.pt", map_location="cpu", weights_only=False)
        assert tb["samples"].device.type == "cpu"
        assert tb["samples"].requires_grad is False
        assert paths["traceback"].endswith("traceback.txt")

    def test_save_failure_records_secondary_errors(self, tmp_path, monkeypatch):
        import json
        from tool_diagnose_rep2_nan import save_failure

        out = tmp_path / "diag"
        save_failure(
            out,
            traceback_text="t",
            failure_summary={},
            trigger_batch={"samples": torch.randn(2)},
            outputs={},
            losses={},
            geometry_snapshot={},
            gradients_summary={},
            model_state={},
            optimizer_state={},
        )
        fs = json.loads((out / "failure" / "failure_summary.json").read_text())
        assert "secondary_errors" in fs
        assert (out / "failure" / "traceback.txt").read_text() == "t"


class TestRestoreCheckpoint:

    def _model(self):
        return torch.nn.Linear(2, 2)

    def test_full_restore(self):
        from tool_diagnose_rep2_nan import restore_checkpoint

        model = self._model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        state = {
            "last_epoch": 87,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        r = restore_checkpoint(model, optimizer, state)
        assert r["start_epoch"] == 88
        assert r["fidelity"] == "full"
        assert r["loaded"]["model"] == "ok"
        assert r["loaded"]["optimizer"] == "ok"
        assert r["missing"] == []
        assert r["unexpected"] == []

    def test_start_epoch_override(self):
        from tool_diagnose_rep2_nan import restore_checkpoint

        model = self._model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        state = {
            "last_epoch": 87,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        r = restore_checkpoint(model, optimizer, state, start_epoch_override=10)
        assert r["start_epoch"] == 10

    def test_weights_only(self):
        from tool_diagnose_rep2_nan import restore_checkpoint

        model = self._model()
        state = {"model": model.state_dict()}
        r = restore_checkpoint(model, None, state)
        assert r["fidelity"] == "weights_only"
        assert r["start_epoch"] == 0  # last_epoch 缺失 → -1 + 1

    def test_weights_only_ema_module(self):
        from tool_diagnose_rep2_nan import restore_checkpoint

        model = self._model()
        state = {"ema": {"module": model.state_dict(), "updates": 5}}
        r = restore_checkpoint(model, None, state)
        assert r["fidelity"] == "weights_only"
        assert any("ema.module" in n for n in r["notes"])

    def test_partial_on_optimizer_failure(self):
        from tool_diagnose_rep2_nan import restore_checkpoint

        model = self._model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        bad_opt = {
            "state": {"999999": {}},
            "param_groups": [
                {
                    "lr": 1e-4,
                    "betas": [0.9, 0.999],
                    "eps": 1e-8,
                    "weight_decay": 0.0,
                    "amsgrad": False,
                    "params": [999999],
                }
            ],
        }
        state = {"last_epoch": 3, "model": model.state_dict(), "optimizer": bad_opt}
        r = restore_checkpoint(model, optimizer, state)
        assert r["fidelity"] == "partial"
        assert r["loaded"]["optimizer"] != "ok"

    def test_missing_unexpected_recorded(self):
        from tool_diagnose_rep2_nan import restore_checkpoint

        model = self._model()
        sd = dict(model.state_dict())
        sd["extra.key"] = torch.zeros(1)  # unexpected
        r = restore_checkpoint(model, None, {"model": sd})
        assert "extra.key" in r["unexpected"]
        assert r["loaded"]["model"] == "ok"

    def test_invalid_raises(self):
        import pytest
        from tool_diagnose_rep2_nan import restore_checkpoint

        model = self._model()
        with pytest.raises(ValueError):
            restore_checkpoint(model, None, {"optimizer": {}})
