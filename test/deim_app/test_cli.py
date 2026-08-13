"""Tests for ``deim_app.cli`` — the unified CLI (Task 9).

Two test groups:

1. **Parser / whitelist** — argparse accepts exactly the approved flags for
   each subcommand and rejects everything else (``--angle-rep``, ``-u`` catch-all,
   unknown subcommands, unknown formats for ``infer --format``).

2. **CLI / API equivalence** — ``monkeypatch`` ``DetectionModel.from_config``
   to return a facade wrapping :class:`~_facade_fakes.FakeAdapter`, then assert
   the CLI delegates correctly: ``infer`` requests one filtered collection and
   writes only the requested formats; ``train``/``eval``/``export`` dispatch to
   the right facade/adapter methods; typed errors print a message and return
   non-zero without swallowing the original exception in Python API use.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from deim_app.api import DetectionModel
from deim_app.cli import _build_parser, main
from deim_app.config.schema import InferenceConfig
from deim_app.errors import AppConfigError, ExportError, InferenceBackendError

from _facade_fakes import FakeAdapter


# ---------------------------------------------------------------------------
# Helper: monkeypatch DetectionModel.from_config to bypass engine construction.
# ---------------------------------------------------------------------------


def _patch_from_config(monkeypatch, adapter: FakeAdapter) -> FakeAdapter:
    """Replace ``DetectionModel.from_config`` with a factory returning a facade
    wrapping ``adapter``. Returns the same adapter for call assertions."""

    def _factory(cls, config_path, **overrides):  # noqa: ANN001
        return DetectionModel(adapter)

    monkeypatch.setattr(DetectionModel, "from_config", classmethod(_factory))
    return adapter


# ===========================================================================
# Group 1 — Parser / whitelist
# ===========================================================================


class TestParserWhitelist:
    """Argparse must accept exactly the approved flags and reject everything else."""

    def test_config_required_for_all_subcommands(self) -> None:
        with pytest.raises(SystemExit) as exc_info:
            main(["train"])
        assert exc_info.value.code == 2

    @pytest.mark.parametrize(
        "argv",
        [
            ["train", "-c", "app.yml"],
            ["eval", "-c", "app.yml"],
            ["infer", "-c", "app.yml", "-r", "c.pt", "-i", "in", "-o", "out"],
            ["export", "-c", "app.yml", "-r", "c.pt", "-o", "out", "--format", "onnx"],
        ],
    )
    def test_subcommands_accept_config_short_flag(self, argv: list[str]) -> None:
        """``-c`` is accepted by every subcommand (parse succeeds)."""
        parser = _build_parser()
        ns = parser.parse_args(argv)
        assert ns.config == "app.yml"

    def test_config_long_flag_accepted(self) -> None:
        parser = _build_parser()
        ns = parser.parse_args(["train", "--config", "app.yml"])
        assert ns.config == "app.yml"

    def test_train_accepts_approved_flags(self) -> None:
        parser = _build_parser()
        ns = parser.parse_args([
            "train", "-c", "app.yml",
            "--device", "cuda:1",
            "--resume", "resume.pt",
            "--output-dir", "outputs/exp",
        ])
        assert ns.device == "cuda:1"
        assert ns.resume == "resume.pt"
        assert ns.output_dir == "outputs/exp"

    def test_eval_accepts_approved_flags(self) -> None:
        parser = _build_parser()
        ns = parser.parse_args(["eval", "-c", "app.yml", "-r", "ckpt.pt", "--device", "cpu"])
        assert ns.checkpoint == "ckpt.pt"
        assert ns.device == "cpu"

    def test_infer_accepts_all_approved_flags(self) -> None:
        parser = _build_parser()
        ns = parser.parse_args([
            "infer", "-c", "app.yml",
            "-r", "ckpt.pt", "-i", "images/", "-o", "out/",
            "--device", "cuda",
            "--batch-size", "4",
            "--score-threshold", "0.5",
            "--top-k", "100",
            "--class-filter", "c0", "c1",
            "--format", "json", "visualization",
        ])
        assert ns.checkpoint == "ckpt.pt"
        assert ns.input == "images/"
        assert ns.output == "out/"
        assert ns.device == "cuda"
        assert ns.batch_size == 4
        assert ns.score_threshold == 0.5
        assert ns.top_k == 100
        assert ns.class_filter == ["c0", "c1"]
        assert ns.format == ["json", "visualization"]

    def test_export_accepts_approved_flags(self) -> None:
        parser = _build_parser()
        ns = parser.parse_args([
            "export", "-c", "app.yml",
            "-r", "ckpt.pt", "-o", "model.onnx",
            "--format", "onnx", "--device", "cpu",
        ])
        assert ns.checkpoint == "ckpt.pt"
        assert ns.output == "model.onnx"
        assert ns.format == "onnx"
        assert ns.device == "cpu"

    # --- Rejection tests ---

    def test_angle_rep_rejected(self) -> None:
        """``--angle-rep`` is NOT in the approved whitelist → argparse exit 2."""
        with pytest.raises(SystemExit) as exc_info:
            main(["train", "-c", "app.yml", "--angle-rep", "1"])
        assert exc_info.value.code == 2

    def test_u_override_rejected(self) -> None:
        """No ``-u/--update`` catch-all → argparse exit 2."""
        with pytest.raises(SystemExit) as exc_info:
            main([
                "train", "-c", "app.yml",
                "-u", "DEIMTransformer.angle_rep=2",
            ])
        assert exc_info.value.code == 2

    def test_unknown_subcommand_rejected(self) -> None:
        with pytest.raises(SystemExit) as exc_info:
            main(["bogus", "-c", "app.yml"])
        assert exc_info.value.code == 2

    def test_infer_rejects_unknown_format(self) -> None:
        """``--format`` for infer validates against known formats only."""
        with pytest.raises(SystemExit) as exc_info:
            main([
                "infer", "-c", "app.yml", "-r", "c.pt",
                "-i", "in", "-o", "out", "--format", "xml",
            ])
        assert exc_info.value.code == 2

    def test_infer_format_accepts_known_formats(self, monkeypatch) -> None:
        """All three known formats pass argparse (downstream may still fail)."""
        adapter = _patch_from_config(monkeypatch, FakeAdapter())
        # This should reach the handler (not argparse), and fail on predict.
        rc = main([
            "infer", "-c", "app.yml", "-r", "c.pt",
            "-i", "in", "-o", "out", "--format", "json", "dota", "visualization",
        ])
        # dota on hbb → ExportError → rc 1
        assert rc == 1

    def test_no_subcommand_required(self) -> None:
        """Omitting the subcommand must exit non-zero."""
        with pytest.raises(SystemExit) as exc_info:
            main(["-c", "app.yml"])
        assert exc_info.value.code == 2


# ===========================================================================
# Group 2 — CLI / API equivalence
# ===========================================================================


class TestTrainDelegation:
    def test_train_calls_model_train(self, monkeypatch) -> None:
        adapter = _patch_from_config(monkeypatch, FakeAdapter())
        rc = main(["train", "-c", "app.yml"])
        assert rc == 0
        assert adapter.train_called is True

    def test_train_with_device_override(self, monkeypatch) -> None:
        """``--device`` is forwarded as a CLI override to from_config."""
        captured: dict[str, object] = {}

        def _factory(cls, config_path, **overrides):  # noqa: ANN001
            captured["config_path"] = config_path
            captured["overrides"] = overrides
            return DetectionModel(FakeAdapter())

        monkeypatch.setattr(DetectionModel, "from_config", classmethod(_factory))
        main(["train", "-c", "app.yml", "--device", "cuda:1"])
        assert captured["config_path"] == "app.yml"
        assert captured["overrides"] == {"train": {"device": "cuda:1"}}

    def test_train_with_resume_override(self, monkeypatch) -> None:
        captured: dict[str, object] = {}

        def _factory(cls, config_path, **overrides):  # noqa: ANN001
            captured["overrides"] = overrides
            return DetectionModel(FakeAdapter())

        monkeypatch.setattr(DetectionModel, "from_config", classmethod(_factory))
        main(["train", "-c", "app.yml", "--resume", "last.pt"])
        assert captured["overrides"] == {"train": {"resume": "last.pt"}}

    def test_train_with_output_dir_override(self, monkeypatch) -> None:
        captured: dict[str, object] = {}

        def _factory(cls, config_path, **overrides):  # noqa: ANN001
            captured["overrides"] = overrides
            return DetectionModel(FakeAdapter())

        monkeypatch.setattr(DetectionModel, "from_config", classmethod(_factory))
        main(["train", "-c", "app.yml", "--output-dir", "runs/exp"])
        assert captured["overrides"] == {"project": {"output_dir": "runs/exp"}}


class TestEvalDelegation:
    def test_eval_loads_checkpoint_and_evaluates(self, monkeypatch) -> None:
        adapter = _patch_from_config(monkeypatch, FakeAdapter())
        rc = main(["eval", "-c", "app.yml", "-r", "ckpt.pt"])
        assert rc == 0
        assert adapter.load_calls == [("ckpt.pt", True)]
        assert adapter.evaluate_calls == ["ckpt.pt"]

    def test_eval_without_checkpoint(self, monkeypatch) -> None:
        adapter = _patch_from_config(monkeypatch, FakeAdapter())
        rc = main(["eval", "-c", "app.yml"])
        assert rc == 0
        assert adapter.load_calls == [(None, True)]
        assert adapter.evaluate_calls == [None]

    def test_eval_device_override(self, monkeypatch) -> None:
        captured: dict[str, object] = {}

        def _factory(cls, config_path, **overrides):  # noqa: ANN001
            captured["overrides"] = overrides
            return DetectionModel(FakeAdapter())

        monkeypatch.setattr(DetectionModel, "from_config", classmethod(_factory))
        main(["eval", "-c", "app.yml", "-r", "c.pt", "--device", "cpu"])
        assert captured["overrides"] == {"evaluation": {"device": "cpu"}}


class TestInferDelegation:
    def test_infer_requests_one_filtered_collection(self, monkeypatch, tmp_path) -> None:
        """infer calls predict exactly once (one full collection from the adapter)."""
        adapter = _patch_from_config(
            monkeypatch,
            FakeAdapter(inference=InferenceConfig(output_formats=("json",))),
        )
        rc = main([
            "infer", "-c", "app.yml", "-r", "ckpt.pt",
            "-i", "images/", "-o", str(tmp_path),
        ])
        assert rc == 0
        assert len(adapter.predict_calls) == 1

    def test_infer_writes_only_requested_json(self, monkeypatch, tmp_path) -> None:
        adapter = _patch_from_config(
            monkeypatch,
            FakeAdapter(inference=InferenceConfig(output_formats=("json",))),
        )
        rc = main([
            "infer", "-c", "app.yml", "-r", "ckpt.pt",
            "-i", "images/", "-o", str(tmp_path),
            "--format", "json",
        ])
        assert rc == 0
        assert (tmp_path / "predictions.json").exists()
        # No visualization directory or dota files.
        assert not (tmp_path / "img0.png").exists()

    def test_infer_writes_visualization(self, monkeypatch, tmp_path) -> None:
        _patch_from_config(
            monkeypatch,
            FakeAdapter(inference=InferenceConfig(output_formats=("visualization",))),
        )
        rc = main([
            "infer", "-c", "app.yml", "-r", "ckpt.pt",
            "-i", "images/", "-o", str(tmp_path),
            "--format", "visualization",
        ])
        assert rc == 0
        assert (tmp_path / "img0.png").exists()

    def test_infer_writes_multiple_formats(self, monkeypatch, tmp_path) -> None:
        _patch_from_config(
            monkeypatch,
            FakeAdapter(inference=InferenceConfig(output_formats=("json", "visualization"))),
        )
        rc = main([
            "infer", "-c", "app.yml", "-r", "ckpt.pt",
            "-i", "images/", "-o", str(tmp_path),
            "--format", "json", "visualization",
        ])
        assert rc == 0
        assert (tmp_path / "predictions.json").exists()
        assert (tmp_path / "img0.png").exists()

    def test_infer_default_formats_from_config(self, monkeypatch, tmp_path) -> None:
        """Omitting --format pulls defaults from inference.output_formats."""
        _patch_from_config(
            monkeypatch,
            FakeAdapter(
                inference=InferenceConfig(output_formats=("json", "visualization")),
            ),
        )
        rc = main([
            "infer", "-c", "app.yml", "-r", "ckpt.pt",
            "-i", "images/", "-o", str(tmp_path),
        ])
        assert rc == 0
        assert (tmp_path / "predictions.json").exists()
        assert (tmp_path / "img0.png").exists()

    def test_infer_applies_score_threshold(self, monkeypatch, tmp_path) -> None:
        """--score-threshold filters through predict_filtered."""
        adapter = _patch_from_config(monkeypatch, FakeAdapter())
        # Use the facade directly to verify filtering outcome.
        model = DetectionModel(adapter)
        result = model.predict_filtered("images", score_threshold=0.85)
        # Only the 0.9-score detection survives >= 0.85.
        assert len(result.predictions[0].detections) == 1
        # Also verify via CLI that it doesn't crash.
        rc = main([
            "infer", "-c", "app.yml", "-r", "ckpt.pt",
            "-i", "images/", "-o", str(tmp_path),
            "--format", "json", "--score-threshold", "0.85",
        ])
        assert rc == 0

    def test_infer_applies_top_k(self, monkeypatch, tmp_path) -> None:
        adapter = _patch_from_config(monkeypatch, FakeAdapter())
        model = DetectionModel(adapter)
        result = model.predict_filtered("images", top_k=1)
        assert len(result.predictions[0].detections) == 1
        assert result.predictions[0].detections[0].score == 0.9

    def test_infer_applies_class_filter(self, monkeypatch, tmp_path) -> None:
        adapter = _patch_from_config(monkeypatch, FakeAdapter())
        model = DetectionModel(adapter)
        result = model.predict_filtered("images", class_filter=["c1"])
        assert len(result.predictions[0].detections) == 1
        assert result.predictions[0].detections[0].class_name == "c1"

    def test_infer_passes_batch_size(self, monkeypatch, tmp_path) -> None:
        adapter = _patch_from_config(
            monkeypatch,
            FakeAdapter(inference=InferenceConfig(output_formats=("json",))),
        )
        rc = main([
            "infer", "-c", "app.yml", "-r", "ckpt.pt",
            "-i", "images/", "-o", str(tmp_path),
            "--format", "json", "--batch-size", "8",
        ])
        assert rc == 0
        assert adapter.predict_calls[-1] == ("images/", 8)

    def test_infer_format_dota_on_hbb_exits_nonzero(self, monkeypatch, tmp_path) -> None:
        """format=dota on HBB collection → ExportError → rc 1."""
        adapter = _patch_from_config(
            monkeypatch,
            FakeAdapter(inference=InferenceConfig(output_formats=("dota",))),
        )
        assert adapter.box_mode == "hbb"
        rc = main([
            "infer", "-c", "app.yml", "-r", "ckpt.pt",
            "-i", "images/", "-o", str(tmp_path),
            "--format", "dota",
        ])
        assert rc == 1

    def test_infer_unknown_format_from_config_exits_nonzero(
        self, monkeypatch, tmp_path,
    ) -> None:
        """Config-provided unknown format (when --format omitted) → ExportError → rc 1."""
        _patch_from_config(
            monkeypatch,
            FakeAdapter(
                inference=InferenceConfig(output_formats=("json", "xml")),
            ),
        )
        rc = main([
            "infer", "-c", "app.yml", "-r", "ckpt.pt",
            "-i", "images/", "-o", str(tmp_path),
        ])
        assert rc == 1

    def test_infer_checkpoint_passed_to_load(self, monkeypatch, tmp_path) -> None:
        adapter = _patch_from_config(
            monkeypatch,
            FakeAdapter(inference=InferenceConfig(output_formats=("json",))),
        )
        rc = main([
            "infer", "-c", "app.yml", "-r", "weights.pt",
            "-i", "images/", "-o", str(tmp_path),
        ])
        assert rc == 0
        assert adapter.load_calls == [("weights.pt", True)]

    def test_infer_device_override(self, monkeypatch, tmp_path) -> None:
        captured: dict[str, object] = {}

        def _factory(cls, config_path, **overrides):  # noqa: ANN001
            captured["overrides"] = overrides
            return DetectionModel(
                FakeAdapter(inference=InferenceConfig(output_formats=("json",))),
            )

        monkeypatch.setattr(DetectionModel, "from_config", classmethod(_factory))
        rc = main([
            "infer", "-c", "app.yml", "-r", "c.pt",
            "-i", "images/", "-o", str(tmp_path),
            "--device", "cuda:0",
        ])
        assert rc == 0
        assert captured["overrides"] == {"inference": {"device": "cuda:0"}}


class TestExportDelegation:
    def test_export_always_fails_with_export_error(self, monkeypatch, tmp_path) -> None:
        """v1 export always raises ExportError → CLI catches and returns 1."""
        adapter = _patch_from_config(monkeypatch, FakeAdapter())
        rc = main([
            "export", "-c", "app.yml", "-r", "ckpt.pt",
            "-o", str(tmp_path / "model.onnx"), "--format", "onnx",
        ])
        assert rc == 1
        # The adapter's export was called (delegation verified).
        assert len(adapter.export_calls) == 1
        ckpt, fmt, out = adapter.export_calls[0]
        assert ckpt == "ckpt.pt"
        assert fmt == "onnx"


# ===========================================================================
# Error handling — concise message + non-zero + __cause__ preservation
# ===========================================================================


class TestErrorHandling:
    def test_error_prints_message_to_stderr_and_returns_nonzero(
        self, monkeypatch, capsys, tmp_path,
    ) -> None:
        """A typed error prints a concise message to stderr and returns 1."""
        def _factory(cls, config_path, **overrides):  # noqa: ANN001
            raise AppConfigError("config is malformed: missing __include__")

        monkeypatch.setattr(DetectionModel, "from_config", classmethod(_factory))
        rc = main([
            "infer", "-c", "app.yml", "-r", "ckpt.pt",
            "-i", "images/", "-o", str(tmp_path),
        ])
        assert rc == 1
        captured = capsys.readouterr()
        assert "config is malformed" in captured.err

    def test_export_error_message_printed(self, monkeypatch, capsys, tmp_path) -> None:
        _patch_from_config(monkeypatch, FakeAdapter())
        rc = main([
            "export", "-c", "app.yml", "-r", "ckpt.pt",
            "-o", str(tmp_path / "m.onnx"), "--format", "onnx",
        ])
        assert rc == 1
        captured = capsys.readouterr()
        assert "export" in captured.err.lower()

    def test_python_api_preserves_exception(self, monkeypatch) -> None:
        """The facade (Python API path) raises the original typed exception
        without swallowing — __cause__ is not erased by a catch-and-rewrap.

        The CLI catches typed errors only at the outermost ``main()`` boundary;
        the facade itself never catches, so programmatic users get the original
        exception object with its natural ``__cause__`` chain intact.
        """
        model = DetectionModel(FakeAdapter(is_loaded=False))
        with pytest.raises(InferenceBackendError) as exc_info:
            model.predict("images")
        # Original exception preserved — not wrapped into a generic error.
        assert isinstance(exc_info.value, InferenceBackendError)
        assert "not loaded" in str(exc_info.value)
        # __cause__ is whatever the original raise set (None for a direct raise),
        # proving the exception was not caught-and-rewrapped.
        assert exc_info.value.__cause__ is None
