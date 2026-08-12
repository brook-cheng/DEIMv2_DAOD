"""Tests for ``deim_app.adapters`` Task 5: checkpoint selection + DEIM adapter.

Two layers are exercised:

  1. ``select_model_state`` (pure-function, no engine) — EMA preference, model
     fallback, ``module.`` prefix stripping, and the missing-key error.
  2. ``DeimDetectionAdapter`` orchestration — engine seams (``YAMLConfig``,
     ``torch.load``, ``select_model_state``) are monkeypatched to stubs so we
     can assert call ORDER and argument plumbing without constructing real
     engine objects (which need GPU + weights).

The class-count compatibility check is driven by feeding the stub model a
``state_dict()`` whose class-head shapes differ from the checkpoint's.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from deim_app.adapters import (
    DeimDetectionAdapter,
    DetectionAdapter,
    select_model_state,
)
from deim_app.errors import CheckpointCompatibilityError, ExportError


# ===========================================================================
# Step 1 — select_model_state
# ===========================================================================


def test_select_model_state_prefers_ema_module() -> None:
    """Exact brief test: prefer_ema=True returns the ema.module state."""
    checkpoint = {
        "model": {"weight": torch.tensor([1.0])},
        "ema": {"module": {"weight": torch.tensor([2.0])}},
    }
    state = select_model_state(checkpoint, prefer_ema=True)
    assert state["weight"].item() == 2.0


def test_select_model_state_falls_back_to_model() -> None:
    """Exact brief test: with no ema, prefer_ema=True still falls back to model."""
    checkpoint = {"model": {"module.weight": torch.tensor([1.0])}}
    state = select_model_state(checkpoint, prefer_ema=True)
    assert set(state) == {"weight"}
    assert torch.equal(state["weight"], torch.tensor([1.0]))


def test_select_model_state_strips_module_prefix() -> None:
    """DDP-trained checkpoints carry 'module.' prefixes; strip them all."""
    checkpoint = {
        "model": {
            "module.backbone.layer1.weight": torch.zeros(2),
            "module.decoder.head.bias": torch.ones(2),
            "decoder.no_prefix.weight": torch.zeros(3),
        }
    }
    state = select_model_state(checkpoint, prefer_ema=False)
    assert set(state) == {
        "backbone.layer1.weight",
        "decoder.head.bias",
        "decoder.no_prefix.weight",
    }


def test_select_model_state_strips_module_prefix_from_ema() -> None:
    """EMA state is also stripped of the module. prefix."""
    checkpoint = {
        "model": {"weight": torch.tensor([1.0])},
        "ema": {"module": {"module.head.weight": torch.tensor([9.0])}},
    }
    state = select_model_state(checkpoint, prefer_ema=True)
    assert set(state) == {"head.weight"}
    assert state["head.weight"].item() == 9.0


def test_select_model_state_raises_when_neither_present() -> None:
    """No 'ema' and no 'model' key → CheckpointCompatibilityError."""
    checkpoint = {"optimizer": {"state": {}}, "last_epoch": 5}
    with pytest.raises(CheckpointCompatibilityError) as exc:
        select_model_state(checkpoint, prefer_ema=True)
    msg = str(exc.value)
    assert "ema" in msg
    assert "model" in msg


def test_select_model_state_raises_when_neither_present_prefer_model() -> None:
    """prefer_ema=False still requires 'model' key."""
    checkpoint = {"ema": {"module": {"weight": torch.tensor([1.0])}}}
    with pytest.raises(CheckpointCompatibilityError):
        select_model_state(checkpoint, prefer_ema=False)


def test_select_model_state_prefer_ema_false_ignores_ema() -> None:
    """prefer_ema=False always takes 'model', even when ema.module exists."""
    checkpoint = {
        "model": {"weight": torch.tensor([1.0])},
        "ema": {"module": {"weight": torch.tensor([2.0])}},
    }
    state = select_model_state(checkpoint, prefer_ema=False)
    assert state["weight"].item() == 1.0


def test_select_model_state_ema_without_module_falls_back_to_model() -> None:
    """An 'ema' dict missing the 'module' key falls back to 'model'."""
    checkpoint = {
        "model": {"weight": torch.tensor([1.0])},
        "ema": {"decay": 0.9999},  # no 'module'
    }
    state = select_model_state(checkpoint, prefer_ema=True)
    assert state["weight"].item() == 1.0


def test_select_model_state_returns_plain_dict() -> None:
    """The returned mapping is a fresh dict (caller may freely mutate it)."""
    checkpoint = {"model": {"w": torch.zeros(1)}}
    state = select_model_state(checkpoint)
    assert isinstance(state, dict)


# ===========================================================================
# Step 2 — DeimDetectionAdapter orchestration (engine seams monkeypatched)
# ===========================================================================


class _FakeTorch:
    """Minimal stand-in whose ``.load`` is a custom callable.

    We swap the adapter's ``torch`` name with this so ``torch.load`` routes to
    the test's fake while keeping everything else the adapter does intact.
    """

    def __init__(self, load_fn) -> None:
        self.load = load_fn


def _build_stub_yaml(
    call_log: list[str],
    *,
    yaml_cfg: dict[str, Any] | None = None,
    model_state: dict[str, torch.Tensor] | None = None,
    cfg_path: str = "/synthetic/app.yml",
    extra_kwargs: dict[str, Any] | None = None,
) -> Any:
    """Construct a single StubYAMLConfig wired to ``call_log``.

    ``cfg_path`` and ``extra_kwargs`` are recorded on the stub so tests can
    assert what the adapter actually forwarded to ``YAMLConfig(...)``.
    """
    from _stubs import StubModel, StubPostprocessor, StubYAMLConfig

    model = StubModel(call_log, state=model_state)
    postprocessor = StubPostprocessor(call_log)
    kwargs: dict[str, Any] = dict(extra_kwargs) if extra_kwargs else {}
    return StubYAMLConfig(
        cfg_path,
        call_log=call_log,
        yaml_cfg=(
            {"HGNetv2": {"pretrained": True}} if yaml_cfg is None else yaml_cfg
        ),
        model=model,
        postprocessor=postprocessor,
        **kwargs,
    )


def _last_stub_yaml() -> Any:
    from _stubs import StubYAMLConfig

    assert StubYAMLConfig.instances, "no StubYAMLConfig was constructed"
    return StubYAMLConfig.instances[-1]


@pytest.fixture
def patched_adapter(monkeypatch, canned_resolved, canned_loaded, call_log):
    """Return an adapter with engine seams patched to call-log-recording stubs.

    Patches (all on the ``deim_app.adapters.deim`` module namespace):
      - ``YAMLConfig`` → a closure that builds a wired ``StubYAMLConfig`` and
        captures the real ``(cfg_path, **kwargs)`` the adapter passed.
      - ``torch`` → a ``_FakeTorch`` whose ``.load`` appends to ``call_log``
        and returns a canned checkpoint dict.
      - ``select_model_state`` → appends to ``call_log`` and returns the
        checkpoint's ema.module state.

    The stub model serves an EMPTY ``state_dict()`` so the class-count
    compatibility check finds no head keys and passes. Tests that need a
    mismatch build their own stub inline (see class-count tests below).
    """
    import deim_app.adapters.deim as deim_mod

    def make_yaml(cfg_path: str, **kwargs: Any) -> Any:
        return _build_stub_yaml(
            call_log, cfg_path=cfg_path, extra_kwargs=kwargs
        )

    def fake_torch_load(path, map_location=None):
        call_log.append("torch.load")
        return {"ema": {"module": {"backbone.w": torch.zeros(2)}}}

    def fake_select(ckpt, prefer_ema=True):
        call_log.append("select_model_state")
        if isinstance(ckpt, dict) and "ema" in ckpt and "module" in ckpt["ema"]:
            return dict(ckpt["ema"]["module"])
        return dict(ckpt.get("model", {}))

    monkeypatch.setattr(deim_mod, "YAMLConfig", make_yaml)
    monkeypatch.setattr(deim_mod, "torch", _FakeTorch(fake_torch_load))
    monkeypatch.setattr(deim_mod, "select_model_state", fake_select)

    return DeimDetectionAdapter(resolved=canned_resolved, loaded=canned_loaded)


# ---- from_config / resolve_config ------------------------------------------------


def test_from_config_calls_loader_and_resolver(
    monkeypatch, canned_loaded, canned_resolved
):
    """from_config delegates to load_app_config + resolve_algorithm_config."""
    # Patch the SOURCE namespace — from_config does a method-local
    # ``from deim_app.config import load_app_config, resolve_algorithm_config``
    # (lazy to avoid a circular import), so the patch must live on
    # ``deim_app.config``, not on ``deim_app.adapters.deim``.
    import deim_app.config as config_mod

    seen_paths: list = []
    seen_overrides: list = []

    def fake_load(p, o=None):
        seen_paths.append(p)
        seen_overrides.append(o)
        return canned_loaded

    monkeypatch.setattr(config_mod, "load_app_config", fake_load)
    monkeypatch.setattr(
        config_mod, "resolve_algorithm_config", lambda loaded: canned_resolved
    )

    adapter = DeimDetectionAdapter.from_config(
        "/some/path.yml", cli_overrides={"train": {"epochs": 5}}
    )

    assert seen_paths == ["/some/path.yml"]
    assert seen_overrides == [{"train": {"epochs": 5}}]
    assert adapter.resolved is canned_resolved
    assert adapter.loaded is canned_loaded
    assert adapter.box_mode == canned_resolved.metadata.box_mode
    assert adapter.metadata is canned_resolved.metadata


def test_resolve_config_re_resolves_from_current_loaded(
    monkeypatch, canned_loaded, canned_resolved
):
    """resolve_config() with no arg re-resolves from self.loaded."""
    import deim_app.config as config_mod

    seen: list = []
    monkeypatch.setattr(
        config_mod,
        "resolve_algorithm_config",
        lambda loaded: (seen.append(loaded), canned_resolved)[1],
    )
    adapter = DeimDetectionAdapter(resolved=canned_resolved, loaded=canned_loaded)

    out = adapter.resolve_config()
    assert seen == [canned_loaded]
    assert out is canned_resolved


def test_resolve_config_with_explicit_loaded(
    monkeypatch, canned_loaded, canned_resolved
):
    """resolve_config(loaded) resolves the provided loaded."""
    import deim_app.config as config_mod

    seen: list = []
    monkeypatch.setattr(
        config_mod,
        "resolve_algorithm_config",
        lambda loaded: (seen.append(loaded), canned_resolved)[1],
    )
    adapter = DeimDetectionAdapter(resolved=canned_resolved, loaded=canned_loaded)

    adapter.resolve_config(canned_loaded)
    assert seen == [canned_loaded]


# ---- load() orchestration -------------------------------------------------------


def test_load_passes_resolved_overrides_to_yamlconfig(
    patched_adapter, canned_resolved
):
    """load() builds YAMLConfig(str(config_path), **resolved.overrides)."""
    patched_adapter.load(checkpoint="/ckpt.pth")

    stub = _last_stub_yaml()
    assert stub.cfg_path == str(canned_resolved.config_path)
    # kwargs must be exactly resolved.overrides — the stub records ALL kwargs
    # the adapter forwarded verbatim.
    assert stub.kwargs == canned_resolved.overrides


def test_load_disables_hgnetv2_pretrained(patched_adapter):
    """The adapter flips HGNetv2.pretrained to False before checkpoint load."""
    patched_adapter.load(checkpoint="/ckpt.pth")

    stub = _last_stub_yaml()
    assert stub.yaml_cfg["HGNetv2"]["pretrained"] is False


def test_load_disables_pretrained_only_when_section_present(
    monkeypatch, canned_resolved, canned_loaded, call_log
):
    """No 'HGNetv2' in yaml_cfg → the adapter must not KeyError."""
    import deim_app.adapters.deim as deim_mod

    stub = _build_stub_yaml(
        call_log, yaml_cfg={"num_classes": 15, "eval_spatial_size": [576, 1024]}
    )
    monkeypatch.setattr(deim_mod, "YAMLConfig", lambda *a, **kw: stub)
    monkeypatch.setattr(deim_mod, "torch", _FakeTorch(lambda p, map_location=None: {"model": {}}))
    monkeypatch.setattr(deim_mod, "select_model_state", lambda c, prefer_ema=True: {})

    adapter = DeimDetectionAdapter(resolved=canned_resolved, loaded=canned_loaded)
    adapter.load(checkpoint="/ckpt.pth")  # must not raise
    assert "HGNetv2" not in stub.yaml_cfg


def test_load_propagates_prefer_ema_to_select_state(
    monkeypatch, canned_resolved, canned_loaded, call_log
):
    """prefer_ema is forwarded to select_model_state."""
    import deim_app.adapters.deim as deim_mod

    stub = _build_stub_yaml(call_log)
    monkeypatch.setattr(deim_mod, "YAMLConfig", lambda *a, **kw: stub)
    monkeypatch.setattr(deim_mod, "torch", _FakeTorch(lambda p, map_location=None: {"model": {}}))
    seen: dict[str, bool] = {}

    def capture(c, prefer_ema=True):
        seen["prefer_ema"] = prefer_ema
        return {}

    monkeypatch.setattr(deim_mod, "select_model_state", capture)

    adapter = DeimDetectionAdapter(resolved=canned_resolved, loaded=canned_loaded)
    adapter.load(checkpoint="/ckpt.pth", prefer_ema=False)
    assert seen == {"prefer_ema": False}


def test_load_call_order(patched_adapter, call_log):
    """Full post-YAML call order:
    torch.load → select_model_state → model.load_state_dict → model.deploy
    → postprocessor.deploy.

    (YAMLConfig is constructed FIRST, before anything else can happen.)
    """
    patched_adapter.load(checkpoint="/ckpt.pth")

    assert call_log == [
        "torch.load",
        "select_model_state",
        "model.load_state_dict",
        "model.deploy",
        "postprocessor.deploy",
    ]


def test_load_calls_load_state_dict_with_strict_false(patched_adapter):
    """model.load_state_dict(state, strict=False) — strict must be False."""
    patched_adapter.load(checkpoint="/ckpt.pth")

    stub = _last_stub_yaml()
    assert stub.model.last_strict is False
    assert stub.model.load_state_dict_calls == 1


def test_load_deploy_called_after_load_state_dict(patched_adapter, call_log):
    """deploy() must follow load_state_dict()."""
    patched_adapter.load(checkpoint="/ckpt.pth")
    stub = _last_stub_yaml()
    assert stub.model.deploy_calls == 1
    assert stub.postprocessor.deploy_calls == 1
    assert call_log.index("model.load_state_dict") < call_log.index("model.deploy")


def test_load_stores_model_and_postprocessor(patched_adapter):
    """After load(), _model and _postprocessor are the deployed objects."""
    patched_adapter.load(checkpoint="/ckpt.pth")
    stub = _last_stub_yaml()
    assert patched_adapter._model is stub.model
    assert patched_adapter._postprocessor is stub.postprocessor
    assert patched_adapter._cfg is stub


def test_load_stores_box_mode_and_metadata(patched_adapter, canned_resolved):
    """box_mode and metadata attributes come from resolved.metadata."""
    patched_adapter.load(checkpoint="/ckpt.pth")
    assert patched_adapter.box_mode == canned_resolved.metadata.box_mode
    assert patched_adapter.metadata is canned_resolved.metadata


def test_load_without_checkpoint_skips_state_loading(
    monkeypatch, canned_resolved, canned_loaded, call_log
):
    """checkpoint=None → no torch.load, no select_model_state, no load_state_dict;
    model + postprocessor are still deployed."""
    import deim_app.adapters.deim as deim_mod

    load_calls: list = []
    fake_torch = _FakeTorch(
        lambda p, map_location=None: load_calls.append(p) or {"model": {}}
    )
    stub = _build_stub_yaml(call_log)
    monkeypatch.setattr(deim_mod, "YAMLConfig", lambda *a, **kw: stub)
    monkeypatch.setattr(deim_mod, "torch", fake_torch)
    select_calls: list = []
    monkeypatch.setattr(
        deim_mod,
        "select_model_state",
        lambda c, prefer_ema=True: select_calls.append(c) or {},
    )

    adapter = DeimDetectionAdapter(resolved=canned_resolved, loaded=canned_loaded)
    adapter.load(checkpoint=None)

    assert load_calls == []  # no torch.load
    assert select_calls == []  # no select_model_state
    assert stub.model.load_state_dict_calls == 0  # no state load
    # But deploy still runs:
    assert stub.model.deploy_calls == 1
    assert stub.postprocessor.deploy_calls == 1


# ---- class-count compatibility check -------------------------------------------


def test_load_raises_on_class_count_mismatch(
    monkeypatch, canned_resolved, canned_loaded, call_log
):
    """Class-head shape mismatch between model and checkpoint → CheckpointCompatibilityError.

    Model head shapes (num_classes=15) differ from checkpoint head shapes
    (num_classes=80) → CheckpointCompatibilityError listing every offending
    key. The check fires BEFORE load_state_dict.
    """
    import deim_app.adapters.deim as deim_mod

    H = 256  # hidden dim, identical on both sides
    model_state = {
        "decoder.enc_score_head.weight": torch.zeros(15, H),
        "decoder.enc_score_head.bias": torch.zeros(15),
        "decoder.dec_score_head.0.weight": torch.zeros(15, H),
        "decoder.dec_score_head.0.bias": torch.zeros(15),
        "decoder.denoising_class_embed.weight": torch.zeros(16, H),  # num_classes+1
        "backbone.block1.weight": torch.zeros(8, 3, 3, 3),  # non-head, ignored
    }
    ckpt_state = {
        "decoder.enc_score_head.weight": torch.zeros(80, H),  # MISMATCH
        "decoder.enc_score_head.bias": torch.zeros(80),  # MISMATCH
        "decoder.dec_score_head.0.weight": torch.zeros(80, H),  # MISMATCH
        "decoder.dec_score_head.0.bias": torch.zeros(80),  # MISMATCH
        "decoder.denoising_class_embed.weight": torch.zeros(81, H),  # MISMATCH
        "backbone.block1.weight": torch.zeros(8, 3, 3, 3),  # matches → not flagged
    }

    stub = _build_stub_yaml(call_log, model_state=model_state)
    monkeypatch.setattr(deim_mod, "YAMLConfig", lambda *a, **kw: stub)
    monkeypatch.setattr(
        deim_mod,
        "torch",
        _FakeTorch(lambda p, map_location=None: {"ema": {"module": ckpt_state}}),
    )
    # select_model_state returns the ckpt_state verbatim.
    monkeypatch.setattr(
        deim_mod, "select_model_state", lambda c, prefer_ema=True: ckpt_state
    )

    adapter = DeimDetectionAdapter(resolved=canned_resolved, loaded=canned_loaded)
    with pytest.raises(CheckpointCompatibilityError) as exc:
        adapter.load(checkpoint="/ckpt.pth")

    msg = str(exc.value)
    for key in (
        "decoder.enc_score_head.weight",
        "decoder.enc_score_head.bias",
        "decoder.dec_score_head.0.weight",
        "decoder.dec_score_head.0.bias",
        "decoder.denoising_class_embed.weight",
    ):
        assert key in msg, f"expected key {key!r} in error message"
    # The matching backbone key must NOT be named.
    assert "backbone.block1.weight" not in msg
    # load_state_dict must NOT have run (check fires first).
    assert stub.model.load_state_dict_calls == 0


def test_load_class_count_check_passes_on_matching_heads(
    monkeypatch, canned_resolved, canned_loaded, call_log
):
    """When model and checkpoint head shapes agree, load_state_dict proceeds."""
    import deim_app.adapters.deim as deim_mod

    H = 256
    matching_state = {
        "decoder.enc_score_head.weight": torch.zeros(15, H),
        "decoder.enc_score_head.bias": torch.zeros(15),
        "decoder.dec_score_head.0.weight": torch.zeros(15, H),
        "decoder.dec_score_head.0.bias": torch.zeros(15),
        "decoder.denoising_class_embed.weight": torch.zeros(16, H),
    }
    stub = _build_stub_yaml(call_log, model_state=matching_state)
    monkeypatch.setattr(deim_mod, "YAMLConfig", lambda *a, **kw: stub)
    monkeypatch.setattr(
        deim_mod,
        "torch",
        _FakeTorch(lambda p, map_location=None: {"ema": {"module": matching_state}}),
    )
    monkeypatch.setattr(
        deim_mod, "select_model_state", lambda c, prefer_ema=True: matching_state
    )

    adapter = DeimDetectionAdapter(resolved=canned_resolved, loaded=canned_loaded)
    adapter.load(checkpoint="/ckpt.pth")  # must NOT raise
    assert stub.model.load_state_dict_calls == 1
    assert stub.model.deploy_calls == 1


def test_load_class_count_check_ignores_keys_only_in_checkpoint(
    monkeypatch, canned_resolved, canned_loaded, call_log
):
    """A class-head key present in checkpoint but absent from the model is NOT a
    class-count mismatch (strict=False load_state_dict handles it). The check
    only flags keys present in BOTH with differing shapes."""
    import deim_app.adapters.deim as deim_mod

    H = 256
    model_state: dict[str, torch.Tensor] = {
        "backbone.w": torch.zeros(4),
    }
    ckpt_state = {
        "backbone.w": torch.zeros(4),
        "decoder.enc_score_head.weight": torch.zeros(80, H),  # not in model
    }
    stub = _build_stub_yaml(call_log, model_state=model_state)
    monkeypatch.setattr(deim_mod, "YAMLConfig", lambda *a, **kw: stub)
    monkeypatch.setattr(
        deim_mod,
        "torch",
        _FakeTorch(lambda p, map_location=None: {"ema": {"module": ckpt_state}}),
    )
    monkeypatch.setattr(
        deim_mod, "select_model_state", lambda c, prefer_ema=True: ckpt_state
    )

    adapter = DeimDetectionAdapter(resolved=canned_resolved, loaded=canned_loaded)
    adapter.load(checkpoint="/ckpt.pth")  # must NOT raise — key is checkpoint-only
    assert stub.model.load_state_dict_calls == 1


# ---- stubs (predict/train/evaluate/export) -------------------------------------


def test_predict_no_longer_raises_not_implemented(patched_adapter):
    """Task 6 implements ``predict``. A loaded adapter now delegates to the
    inference pipeline, so a bogus source surfaces as ``InputSourceError``
    (from ``list_inputs``) rather than ``NotImplementedError``."""
    from deim_app.errors import InputSourceError

    patched_adapter.load(checkpoint="/ckpt.pth")
    with pytest.raises(InputSourceError):
        patched_adapter.predict("nonexistent-source")


def test_train_raises_not_implemented(patched_adapter):
    with pytest.raises(NotImplementedError):
        patched_adapter.train()


def test_evaluate_raises_not_implemented(patched_adapter):
    with pytest.raises(NotImplementedError):
        patched_adapter.evaluate()


def test_supported_export_formats_empty(patched_adapter):
    assert patched_adapter.supported_export_formats() == ()


def test_export_raises_export_error(patched_adapter):
    with pytest.raises(ExportError):
        patched_adapter.export(checkpoint="/ckpt.pth", format="onnx", output="/o")


# ---- protocol ------------------------------------------------------------------


def test_deim_adapter_satisfies_detection_adapter_protocol(patched_adapter):
    """The concrete adapter is structurally compatible with DetectionAdapter."""
    patched_adapter.load(checkpoint="/ckpt.pth")
    assert isinstance(patched_adapter, DetectionAdapter)
