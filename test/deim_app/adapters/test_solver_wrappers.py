"""Tests for Task 8: thin ``train`` / ``evaluate`` / ``export`` wrappers on
``DeimDetectionAdapter``.

These wrappers do NOT call repository scripts via subprocess — they build a
fresh ``YAMLConfig`` and delegate to the existing engine solver via the
``TASKS`` registry. The tests verify dispatch, device propagation, resume
plumbing, and the locked first-version export behavior without constructing
real engine objects.

A fake ``YAMLConfig`` seam (``_SolverStubCfg``) records the kwargs the adapter
forwarded and exposes the settable ``device`` / ``resume`` attributes the
wrapper paths touch. ``TASKS['detection']`` is monkeypatched per test to a
``FakeSolver`` closure so we can assert ``fit()`` vs ``val()`` dispatch.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from deim_app.adapters import DeimDetectionAdapter
from deim_app.errors import ExportError
from engine.solver import TASKS


# ---------------------------------------------------------------------------
# Stub cfg + adapter fixture
# ---------------------------------------------------------------------------


class _SolverStubCfg:
    """Minimal cfg stub for the solver-wrapper paths.

    Exposes ``yaml_cfg['task'] == 'detection'`` for the TASKS lookup plus
    settable ``device`` and ``resume`` attributes. Captures ``(cfg_path,
    kwargs)`` so tests can assert the adapter forwarded them verbatim.
    """

    def __init__(self, cfg_path: str, **kwargs: Any) -> None:
        self.cfg_path = cfg_path
        self.kwargs = dict(kwargs)
        self.yaml_cfg: dict[str, Any] = {"task": "detection"}
        self.device: str = ""
        self.resume: Any = None


@pytest.fixture
def adapter(monkeypatch, canned_resolved, canned_loaded) -> DeimDetectionAdapter:
    """Adapter with the ``YAMLConfig`` seam replaced by ``_SolverStubCfg``.

    ``canned_resolved.config_path`` is ``/synthetic/app.yml`` (no file on
    disk), so ``YAMLConfig`` MUST be stubbed for any wrapper path that
    constructs it. The stub's ``yaml_cfg`` carries ``task='detection'`` so
    the per-test ``TASKS['detection']`` monkeypatch resolves cleanly.
    """
    import deim_app.adapters.deim as deim_mod

    monkeypatch.setattr(deim_mod, "YAMLConfig", _SolverStubCfg)
    return DeimDetectionAdapter(resolved=canned_resolved, loaded=canned_loaded)


# ---------------------------------------------------------------------------
# train()
# ---------------------------------------------------------------------------


def test_train_delegates_to_existing_solver_fit(monkeypatch, adapter) -> None:
    """train() builds TASKS[task](cfg) and calls solver.fit() exactly once."""
    calls: list[str] = []

    class FakeSolver:
        def __init__(self, cfg: Any) -> None:
            self.cfg = cfg

        def fit(self) -> None:
            calls.append("fit")

        def val(self) -> None:
            calls.append("val")

    monkeypatch.setitem(TASKS, "detection", lambda cfg: FakeSolver(cfg))
    adapter.train()
    assert calls == ["fit"]


def test_train_applies_train_device(monkeypatch, adapter, canned_resolved) -> None:
    """train() writes ``app.train.device`` onto the built cfg before solver fit."""
    captured: dict[str, Any] = {}

    class FakeSolver:
        def __init__(self, cfg: Any) -> None:
            captured["cfg"] = cfg

        def fit(self) -> None:
            captured["called"] = "fit"

        def val(self) -> None:
            captured["called"] = "val"

    monkeypatch.setitem(TASKS, "detection", lambda cfg: FakeSolver(cfg))
    adapter.train()
    assert captured["called"] == "fit"
    assert captured["cfg"].device == canned_resolved.app.train.device


def test_train_forwards_resolved_overrides_to_yamlconfig(
    monkeypatch, adapter, canned_resolved
) -> None:
    """train() constructs YAMLConfig(str(config_path), **resolved.overrides)."""
    captured: dict[str, Any] = {}

    class FakeSolver:
        def __init__(self, cfg: Any) -> None:
            captured["cfg_path"] = cfg.cfg_path
            captured["kwargs"] = dict(cfg.kwargs)

        def fit(self) -> None:
            pass

        def val(self) -> None:
            pass

    monkeypatch.setitem(TASKS, "detection", lambda cfg: FakeSolver(cfg))
    adapter.train()
    assert captured["cfg_path"] == str(canned_resolved.config_path)
    assert captured["kwargs"] == canned_resolved.overrides


# ---------------------------------------------------------------------------
# evaluate()
# ---------------------------------------------------------------------------


def test_evaluate_delegates_to_val_with_checkpoint(
    monkeypatch, adapter, tmp_path
) -> None:
    """evaluate(checkpoint) calls solver.val() with cfg.resume == str(checkpoint)."""
    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(b"fixture")
    captured: dict[str, Any] = {}

    class FakeSolver:
        def __init__(self, cfg: Any) -> None:
            captured["cfg"] = cfg

        def val(self) -> None:
            captured["called"] = "val"

        def fit(self) -> None:
            captured["called"] = "fit"

    monkeypatch.setitem(TASKS, "detection", lambda cfg: FakeSolver(cfg))
    adapter.evaluate(checkpoint)
    assert captured["called"] == "val"
    assert captured["cfg"].resume == str(checkpoint)


def test_evaluate_applies_evaluation_device(
    monkeypatch, adapter, canned_resolved
) -> None:
    """evaluate() writes ``app.evaluation.device`` onto the built cfg."""
    captured: dict[str, Any] = {}

    class FakeSolver:
        def __init__(self, cfg: Any) -> None:
            captured["cfg"] = cfg

        def val(self) -> None:
            pass

        def fit(self) -> None:
            pass

    monkeypatch.setitem(TASKS, "detection", lambda cfg: FakeSolver(cfg))
    adapter.evaluate()
    assert captured["cfg"].device == canned_resolved.app.evaluation.device


def test_evaluate_checkpoint_none_preserves_existing_resume(
    monkeypatch, adapter
) -> None:
    """evaluate(checkpoint=None) must NOT overwrite a preset cfg.resume.

    The config/preset-provided resume comes through the YAMLConfig
    constructor; the adapter leaves it untouched when the caller does not
    pass an explicit checkpoint.
    """
    preset_resume = "/preexisting/resume.pth"

    class _StubWithPresetResume:
        def __init__(self, cfg_path: str, **kwargs: Any) -> None:
            self.cfg_path = cfg_path
            self.kwargs = dict(kwargs)
            self.yaml_cfg = {"task": "detection"}
            self.device = ""
            self.resume = preset_resume  # simulate config-provided resume

    import deim_app.adapters.deim as deim_mod

    monkeypatch.setattr(deim_mod, "YAMLConfig", _StubWithPresetResume)

    captured: dict[str, Any] = {}

    class FakeSolver:
        def __init__(self, cfg: Any) -> None:
            captured["cfg"] = cfg

        def val(self) -> None:
            pass

        def fit(self) -> None:
            pass

    monkeypatch.setitem(TASKS, "detection", lambda cfg: FakeSolver(cfg))
    adapter.evaluate()  # checkpoint=None
    assert captured["cfg"].resume == preset_resume


# ---------------------------------------------------------------------------
# supported_export_formats() + export()
# ---------------------------------------------------------------------------


def test_supported_export_formats_empty(adapter) -> None:
    """v1 exposes no export formats."""
    assert adapter.supported_export_formats() == ()


def test_export_raises_and_does_not_create_output(
    adapter, tmp_path
) -> None:
    """export() always raises ExportError in v1 and creates no output file."""
    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(b"fixture")
    output = tmp_path / "out.onnx"
    with pytest.raises(ExportError, match="No export format is enabled"):
        adapter.export(checkpoint=checkpoint, format="onnx", output=output)
    assert not output.exists()


# ---------------------------------------------------------------------------
# Shared _build_engine_cfg() helper
# ---------------------------------------------------------------------------


def test_build_engine_cfg_used_by_load_train_evaluate(
    monkeypatch, canned_resolved, canned_loaded
) -> None:
    """load(), train(), and evaluate() all route engine-object construction
    through the shared ``_build_engine_cfg()`` helper.

    Two independent signals are asserted:

      1. The ``YAMLConfig`` stub records every construction at the class
         level — three calls mean load/train/evaluate each built a cfg.
      2. A spy on ``_build_engine_cfg`` records invocations — three calls
         mean all three entry points route through the helper rather than
         constructing ``YAMLConfig`` inline.

    Each construction must forward ``str(config_path)`` and
    ``resolved.overrides`` verbatim.
    """
    construction_log: list[tuple[str, dict[str, object]]] = []

    class _HelperSpyCfg:
        """Cfg stub supporting the load(), train(), and evaluate() paths."""

        def __init__(self, cfg_path: str, **kwargs: object) -> None:
            self.cfg_path = cfg_path
            self.kwargs = dict(kwargs)
            self.yaml_cfg = {"task": "detection"}
            self.device = ""
            self.resume = None
            # load() deploys model + postprocessor even with checkpoint=None.
            self.model = type("_M", (), {"deploy": lambda self: self})()
            self.postprocessor = type("_P", (), {"deploy": lambda self: self})()
            construction_log.append((cfg_path, dict(kwargs)))

    import deim_app.adapters.deim as deim_mod

    monkeypatch.setattr(deim_mod, "YAMLConfig", _HelperSpyCfg)

    adapter = DeimDetectionAdapter(
        resolved=canned_resolved, loaded=canned_loaded
    )

    helper_calls: list[bool] = []
    real = adapter._build_engine_cfg

    def spy() -> object:
        helper_calls.append(True)
        return real()

    monkeypatch.setattr(adapter, "_build_engine_cfg", spy)

    class _NoopSolver:
        def fit(self) -> None:
            pass

        def val(self) -> None:
            pass

    monkeypatch.setitem(TASKS, "detection", lambda cfg: _NoopSolver())

    adapter.load(checkpoint=None)
    adapter.train()
    adapter.evaluate()

    # Both signals must show exactly three routed constructions.
    assert len(helper_calls) == 3
    assert len(construction_log) == 3
    for cfg_path, kwargs in construction_log:
        assert cfg_path == str(canned_resolved.config_path)
        assert kwargs == canned_resolved.overrides
