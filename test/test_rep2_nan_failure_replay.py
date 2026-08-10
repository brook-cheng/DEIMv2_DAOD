"""rep2 保存失败回放工具单元测试（纯 CPU）。

契约定义见 docs/superpowers/plans/2026-08-10-rep2-stable-atan2.md Task 3。
RED 阶段：全部测试因 ``tool_replay_rep2_nan_failure`` 尚不存在而失败。
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402


EXIT_OK = 0
EXIT_NUMERIC = 2
EXIT_CONFIG = 3
EXIT_RUNTIME = 4


class ToyReplayModel(nn.Module):
    """真实 ``nn.Linear(4, 5)`` 参数路径；按 mode 模拟不同失败形态。"""

    def __init__(self, mode: str = "normal"):
        super().__init__()
        self.mode = mode
        self.fc = nn.Linear(4, 5)

    def forward(self, samples, targets=None):
        out = self.fc(samples)
        if self.mode == "atan2_zero":
            # 原生 atan2(0, 0)：forward 有限，backward 在 (0,0) 产生 NaN。
            zy = out * 0.0
            zx = out * 0.0
            out = out + torch.atan2(zy, zx)
        elif self.mode == "oom":
            raise RuntimeError("CUDA out of memory")
        elif self.mode == "boom":
            raise ValueError("boom")
        return {
            "pred_logits": out,
            "pred_boxes": out,
            "pred_corners": out,
            "ref_points": out,
        }


class ToyReplayCriterion(nn.Module):
    """MSE 有限损失；``nan`` 模式返回标量 NaN 损失。"""

    def __init__(self, mode: str = "mse"):
        super().__init__()
        self.mode = mode

    def forward(self, outputs, targets, **meta):
        if self.mode == "nan":
            return {"loss_total": torch.tensor(float("nan"))}
        boxes = outputs["pred_boxes"]
        target = targets[0]["boxes"]
        return {"loss_xy": nn.functional.mse_loss(boxes, target)}


def _toy_targets(batch: int = 4, width: int = 5) -> list[dict]:
    return [{"boxes": torch.zeros(batch, width)}]


def _toy_samples(batch: int = 4, width: int = 4) -> torch.Tensor:
    return torch.randn(batch, width)


def _write_failure_dir(tmp_path, *, missing: list[str] | None = None) -> dict:
    """写入一组真实格式的失败现场产物，返回 {键: 绝对路径}。"""
    fail_dir = tmp_path / "failure"
    fail_dir.mkdir(parents=True, exist_ok=True)
    model = ToyReplayModel("normal")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    artifacts = {
        "trigger_batch": str(fail_dir / "trigger_batch.pt"),
        "model_state": str(fail_dir / "model_state.pt"),
        "optimizer_state": str(fail_dir / "optimizer_state.pt"),
        "failure_summary": str(fail_dir / "failure_summary.json"),
    }
    torch.save(
        {"samples": _toy_samples(), "targets": _toy_targets()},
        artifacts["trigger_batch"],
    )
    torch.save(model.state_dict(), artifacts["model_state"])
    torch.save(optimizer.state_dict(), artifacts["optimizer_state"])
    (fail_dir / "failure_summary.json").write_text(
        json.dumps(
            {
                "exit_code": 2,
                "kind": "backward_anomaly",
                "epoch": 115,
                "step": 10,
                "global_step": 59810,
            }
        )
    )
    for name in missing or []:
        path = fail_dir / name
        if path.exists():
            path.unlink()
    return artifacts


class TestReplayStepExitCodes:
    """CPU 上 replay_step 的退出码契约。"""

    def test_normal_returns_ok(self):
        from tool_replay_rep2_nan_failure import replay_step

        model = ToyReplayModel("normal")
        criterion = ToyReplayCriterion("mse")
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        result = replay_step(
            model,
            criterion,
            optimizer,
            _toy_samples(),
            _toy_targets(),
            device="cpu",
            use_amp=False,
            step_optimizer=False,
            clip_max_norm=0.0,
            detect_anomaly=True,
            metas=dict(epoch=115, step=10, global_step=59810, epoch_step=520),
        )
        assert result["exit_code"] == EXIT_OK
        assert result["kind"] == "ok"

    def test_atan2_anomaly_exit_2_with_anomaly_detection(self):
        from tool_replay_rep2_nan_failure import replay_step

        model = ToyReplayModel("atan2_zero")
        criterion = ToyReplayCriterion("mse")
        result = replay_step(
            model,
            criterion,
            None,
            _toy_samples(),
            _toy_targets(),
            device="cpu",
            use_amp=False,
            step_optimizer=False,
            clip_max_norm=0.0,
            detect_anomaly=True,
            metas=dict(epoch=115, step=10, global_step=59810, epoch_step=520),
        )
        assert result["exit_code"] == EXIT_NUMERIC
        assert result["kind"] == "backward_anomaly"

    def test_atan2_anomaly_exit_2_without_anomaly_detection(self):
        from tool_replay_rep2_nan_failure import replay_step

        model = ToyReplayModel("atan2_zero")
        criterion = ToyReplayCriterion("mse")
        result = replay_step(
            model,
            criterion,
            None,
            _toy_samples(),
            _toy_targets(),
            device="cpu",
            use_amp=False,
            step_optimizer=False,
            clip_max_norm=0.0,
            detect_anomaly=False,
            metas=dict(epoch=115, step=10, global_step=59810, epoch_step=520),
        )
        assert result["exit_code"] == EXIT_NUMERIC
        assert result["kind"] in ("backward_anomaly", "gradient")

    def test_nan_loss_exit_2_kind_loss(self):
        from tool_replay_rep2_nan_failure import replay_step

        model = ToyReplayModel("normal")
        criterion = ToyReplayCriterion("nan")
        result = replay_step(
            model,
            criterion,
            None,
            _toy_samples(),
            _toy_targets(),
            device="cpu",
            use_amp=False,
            step_optimizer=False,
            clip_max_norm=0.0,
            detect_anomaly=True,
            metas=dict(epoch=115, step=10, global_step=59810, epoch_step=520),
        )
        assert result["exit_code"] == EXIT_NUMERIC
        assert result["kind"] == "loss"

    def test_step_optimizer_changes_params_and_stays_finite(self):
        from tool_replay_rep2_nan_failure import replay_step

        model = ToyReplayModel("normal")
        criterion = ToyReplayCriterion("mse")
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        before = {n: p.detach().clone() for n, p in model.named_parameters()}
        result = replay_step(
            model,
            criterion,
            optimizer,
            _toy_samples(),
            _toy_targets(),
            device="cpu",
            use_amp=False,
            step_optimizer=True,
            clip_max_norm=0.1,
            detect_anomaly=True,
            metas=dict(epoch=115, step=10, global_step=59810, epoch_step=520),
        )
        assert result["exit_code"] == EXIT_OK
        assert result["kind"] == "ok"
        changed = [
            n
            for n, p in model.named_parameters()
            if not torch.equal(p.detach(), before[n])
        ]
        assert changed, "step_optimizer=True 必须至少改变一个参数"
        for p in model.parameters():
            assert torch.isfinite(p).all(), "参数必须全部有限"
        for state in optimizer.state.values():
            for t in state.values():
                if isinstance(t, torch.Tensor):
                    assert torch.isfinite(t).all(), "optimizer 张量状态必须全部有限"

    def test_forward_oom_propagates_runtime_exit_4(self):
        """forward 阶段 OOM 不吞异常：main 兜底返回 exit 4。"""
        from tool_replay_rep2_nan_failure import replay_step

        model = ToyReplayModel("oom")
        criterion = ToyReplayCriterion("mse")
        with pytest.raises(RuntimeError, match="out of memory"):
            replay_step(
                model,
                criterion,
                None,
                _toy_samples(),
                _toy_targets(),
                device="cpu",
                use_amp=False,
                step_optimizer=False,
                clip_max_norm=0.0,
                detect_anomaly=True,
                metas=dict(epoch=115, step=10, global_step=59810, epoch_step=520),
            )

    def test_forward_boom_propagates_exit_4(self):
        """forward 阶段非 RuntimeError 失败：传播到 main → exit 4。"""
        from tool_replay_rep2_nan_failure import replay_step

        model = ToyReplayModel("boom")
        criterion = ToyReplayCriterion("mse")
        with pytest.raises(ValueError, match="boom"):
            replay_step(
                model,
                criterion,
                None,
                _toy_samples(),
                _toy_targets(),
                device="cpu",
                use_amp=False,
                step_optimizer=False,
                clip_max_norm=0.0,
                detect_anomaly=True,
                metas=dict(epoch=115, step=10, global_step=59810, epoch_step=520),
            )


class TestLoadFailureArtifacts:

    def test_loads_all_values(self, tmp_path):
        from tool_replay_rep2_nan_failure import load_failure_artifacts

        _write_failure_dir(tmp_path)
        arts = load_failure_artifacts(tmp_path / "failure")
        assert set(arts) == {
            "trigger_batch",
            "model_state",
            "optimizer_state",
            "failure_summary",
        }
        assert set(arts["trigger_batch"]) == {"samples", "targets"}
        assert "fc.weight" in arts["model_state"]
        assert "state" in arts["optimizer_state"]
        assert arts["failure_summary"]["epoch"] == 115
        assert arts["failure_summary"]["step"] == 10
        assert arts["failure_summary"]["global_step"] == 59810

    def test_missing_mandatory_artifact_raises_naming_files(self, tmp_path):
        from tool_replay_rep2_nan_failure import load_failure_artifacts

        _write_failure_dir(tmp_path, missing=["model_state.pt", "trigger_batch.pt"])
        with pytest.raises(FileNotFoundError) as ei:
            load_failure_artifacts(tmp_path / "failure")
        msg = str(ei.value)
        assert "model_state.pt" in msg
        assert "trigger_batch.pt" in msg

    def test_missing_summary_uses_zero_metadata(self, tmp_path):
        from tool_replay_rep2_nan_failure import load_failure_artifacts

        _write_failure_dir(tmp_path, missing=["failure_summary.json"])
        arts = load_failure_artifacts(tmp_path / "failure")
        assert arts["failure_summary"] == {
            "epoch": 0,
            "step": 0,
            "global_step": 0,
        }

    def test_missing_directory_raises(self, tmp_path):
        from tool_replay_rep2_nan_failure import load_failure_artifacts

        with pytest.raises(FileNotFoundError):
            load_failure_artifacts(tmp_path / "nope")


class TestRestoreStates:

    def test_model_tensors_match_after_restore(self, tmp_path):
        from tool_replay_rep2_nan_failure import load_failure_artifacts, restore_states

        src = ToyReplayModel("normal")
        optimizer = torch.optim.Adam(src.parameters(), lr=1e-3)
        fail_dir = tmp_path / "failure"
        fail_dir.mkdir(parents=True)
        torch.save(
            {"samples": _toy_samples(), "targets": _toy_targets()},
            fail_dir / "trigger_batch.pt",
        )
        torch.save(src.state_dict(), fail_dir / "model_state.pt")
        torch.save(optimizer.state_dict(), fail_dir / "optimizer_state.pt")
        (fail_dir / "failure_summary.json").write_text("{}")
        arts = load_failure_artifacts(fail_dir)

        model = ToyReplayModel("normal")
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        restore_states(model, opt, arts)
        for name, p in model.named_parameters():
            assert torch.equal(p.detach(), src.state_dict()[name]), name

    def test_restore_missing_model_keys_raises(self, tmp_path):
        from tool_replay_rep2_nan_failure import load_failure_artifacts, restore_states

        fail_dir = tmp_path / "failure"
        fail_dir.mkdir(parents=True)
        torch.save(
            {"samples": _toy_samples(), "targets": _toy_targets()},
            fail_dir / "trigger_batch.pt",
        )
        torch.save({"unrelated": torch.zeros(1)}, fail_dir / "model_state.pt")
        torch.save({}, fail_dir / "optimizer_state.pt")
        (fail_dir / "failure_summary.json").write_text("{}")
        arts = load_failure_artifacts(fail_dir)

        model = ToyReplayModel("normal")
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        with pytest.raises(ValueError):
            restore_states(model, opt, arts)


class TestParseArgs:

    def test_config_and_failure_dir_required(self):
        from tool_replay_rep2_nan_failure import parse_args

        with pytest.raises(SystemExit):
            parse_args([])
        with pytest.raises(SystemExit):
            parse_args(["--config", "x.yml"])
        with pytest.raises(SystemExit):
            parse_args(["--failure-dir", "dir"])

    def test_defaults(self):
        from tool_replay_rep2_nan_failure import parse_args

        args = parse_args(["--config", "x.yml", "--failure-dir", "dir"])
        assert args.config == "x.yml"
        assert args.failure_dir == "dir"
        assert args.device == "cuda:0"
        assert args.detect_anomaly is True
        assert args.step_optimizer is False
        assert args.clip_max_norm == 0.0

    def test_overrides(self):
        from tool_replay_rep2_nan_failure import parse_args

        args = parse_args(
            [
                "--config",
                "x.yml",
                "--failure-dir",
                "dir",
                "--device",
                "cpu",
                "--step-optimizer",
                "--clip-max-norm",
                "0.1",
                "--no-detect-anomaly",
            ]
        )
        assert args.device == "cpu"
        assert args.detect_anomaly is False
        assert args.step_optimizer is True
        assert args.clip_max_norm == 0.1


class TestMainWiring:

    def test_main_propagates_replay_exit_2(self, tmp_path, monkeypatch):
        from tool_replay_rep2_nan_failure import main

        _write_failure_dir(tmp_path)
        monkeypatch.setattr(
            "tool_replay_rep2_nan_failure._build_components",
            lambda cfg, dev: (ToyReplayModel("normal"), ToyReplayCriterion("mse"), None, False),
        )
        monkeypatch.setattr(
            "tool_replay_rep2_nan_failure._derive_metas",
            lambda arts: dict(epoch=115, step=10, global_step=59810, epoch_step=520),
        )
        monkeypatch.setattr(
            "tool_replay_rep2_nan_failure.replay_step",
            lambda *a, **k: {"exit_code": EXIT_NUMERIC, "kind": "backward_anomaly"},
        )
        code = main(["--config", "x.yml", "--failure-dir", str(tmp_path / "failure")])
        assert code == EXIT_NUMERIC

    def test_main_build_failure_exit_3(self, tmp_path, monkeypatch):
        from tool_replay_rep2_nan_failure import main

        _write_failure_dir(tmp_path)

        def _boom(cfg, dev):
            raise ValueError("bad config")

        monkeypatch.setattr("tool_replay_rep2_nan_failure._build_components", _boom)
        code = main(["--config", "x.yml", "--failure-dir", str(tmp_path / "failure")])
        assert code == EXIT_CONFIG

    def test_main_artifact_load_failure_exit_3(self, tmp_path, monkeypatch):
        from tool_replay_rep2_nan_failure import main

        code = main(
            ["--config", "x.yml", "--failure-dir", str(tmp_path / "missing_failure")]
        )
        assert code == EXIT_CONFIG

    def test_main_state_restore_failure_exit_3(self, tmp_path, monkeypatch):
        from tool_replay_rep2_nan_failure import main

        _write_failure_dir(tmp_path)
        monkeypatch.setattr(
            "tool_replay_rep2_nan_failure._build_components",
            lambda cfg, dev: (ToyReplayModel("normal"), ToyReplayCriterion("mse"), None, False),
        )
        monkeypatch.setattr(
            "tool_replay_rep2_nan_failure.restore_states",
            lambda m, o, a: (_ for _ in ()).throw(ValueError("state mismatch")),
        )
        code = main(["--config", "x.yml", "--failure-dir", str(tmp_path / "failure")])
        assert code == EXIT_CONFIG

    def test_main_unexpected_replay_error_exit_4(self, tmp_path, monkeypatch):
        from tool_replay_rep2_nan_failure import main

        _write_failure_dir(tmp_path)
        monkeypatch.setattr(
            "tool_replay_rep2_nan_failure._build_components",
            lambda cfg, dev: (ToyReplayModel("normal"), ToyReplayCriterion("mse"), None, False),
        )
        monkeypatch.setattr(
            "tool_replay_rep2_nan_failure._derive_metas",
            lambda arts: dict(epoch=115, step=10, global_step=59810, epoch_step=520),
        )
        monkeypatch.setattr(
            "tool_replay_rep2_nan_failure.replay_step",
            lambda *a, **k: (_ for _ in ()).throw(ValueError("unexpected")),
        )
        code = main(["--config", "x.yml", "--failure-dir", str(tmp_path / "failure")])
        assert code == EXIT_RUNTIME
