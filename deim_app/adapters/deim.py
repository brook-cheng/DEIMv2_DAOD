"""DEIM detection adapter: the only module that constructs engine objects and
loads checkpoints.

Responsibilities:

  * Build engine objects via ``YAMLConfig(str(config_path), **resolved.overrides)``.
  * Disable backbone pretrained-download (so a checkpoint load is never raced
    by an HTTP fetch for the backbone's ImageNet weights).
  * Load + normalize the checkpoint state through
    :func:`deim_app.adapters.checkpoint.select_model_state`.
  * Verify class-count compatibility between the model's class-prediction
    heads and the checkpoint state BEFORE calling ``load_state_dict``.
  * Deploy the model + postprocessor and expose them for inference (Task 6).

Boundary: this module is one of the few under ``deim_app/adapters/`` permitted
to import ``engine.*``. The dependency guard
(``test/deim_app/test_dependency_boundaries.py``) exempts the whole
``adapters/`` package; non-adapter ``deim_app`` code must route through here.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from engine.core import YAMLConfig

from deim_app.adapters.base import DetectionAdapter
from deim_app.adapters.checkpoint import select_model_state
from deim_app.errors import CheckpointCompatibilityError, ExportError

if TYPE_CHECKING:
    # Imported lazily at runtime inside from_config / resolve_config to avoid a
    # circular import: deim_app.config.loader → deim_app.adapters._engine_yaml
    # → deim_app.adapters (package __init__) → this module.
    from deim_app.config import (
        LoadedAppConfig,
        ResolvedAlgorithmConfig,
        load_app_config,
        resolve_algorithm_config,
    )
    from deim_app.predictions.collection import PredictionCollection

__all__ = ["DeimDetectionAdapter"]


# ===========================================================================
# Class-prediction head key discovery
# ===========================================================================
#
# The compatibility check needs to know which state-dict keys carry the
# class-count dimension, so a checkpoint trained for N classes can be rejected
# when the model expects M. The names below are curated from the DEIM decoder
# source (``engine/deim/deim_decoder.py``):
#
#   * ``decoder.enc_score_head`` — ``nn.Linear(hidden_dim, num_classes)``
#       → state-dict keys ``decoder.enc_score_head.{weight,bias}``
#   * ``decoder.dec_score_head`` — ``nn.ModuleList`` of
#     ``nn.Linear(hidden_dim, num_classes)``
#       → keys ``decoder.dec_score_head.{i}.{weight,bias}`` for each layer
#   * ``decoder.denoising_class_embed`` —
#     ``nn.Embedding(num_classes + 1, hidden_dim, padding_idx=num_classes)``
#       → key ``decoder.denoising_class_embed.weight`` of shape
#       ``(num_classes + 1, hidden_dim)``
#
# The engine solver's own ``_adjust_head_parameters``
# (``engine/solver/_solver.py:293``) lists the same ``enc_score_head`` /
# ``dec_score_head.{i}`` / ``denoising_class_embed`` names, confirming these
# are the canonical class-prediction parameters across DEIM checkpoints.
#
# The brief also mentions the generic head names ``class_head`` and ``fc_cls``
# (from older DETR variants); we include them for forward-compatibility. A key
# is treated as a class-head key if it ENDS with one of the suffixes or if it
# lives under one of the ``ModuleList`` prefixes with a ``.weight``/``.bias``
# leaf — conservative, no false positives on backbone / bbox-head keys.

_CLASS_HEAD_SUFFIXES: tuple[str, ...] = (
    "enc_score_head.weight",
    "enc_score_head.bias",
    "denoising_class_embed.weight",
    # Generic / legacy DETR head names from the task brief.
    "class_head.weight",
    "class_head.bias",
    "fc_cls.weight",
    "fc_cls.bias",
)

#: ``nn.ModuleList`` segment names whose ``<parent>.<name>.{i}.{weight,bias}``
#: leaves are class-prediction heads. Checked as a dotted-path segment (not a
#: substring) so a backbone module named e.g. ``foo_dec_score_head`` is not a
#: false positive.
_CLASS_HEAD_MODULELIST_NAMES: frozenset[str] = frozenset({"dec_score_head"})


def _is_class_head_key(name: str) -> bool:
    """Return ``True`` if ``name`` is a class-prediction head parameter."""
    for suffix in _CLASS_HEAD_SUFFIXES:
        if name.endswith(suffix):
            return True
    # ModuleList heads: <parent>...dec_score_head.{i}.{weight,bias}.
    parts = name.split(".")
    if len(parts) >= 3 and parts[-3] in _CLASS_HEAD_MODULELIST_NAMES:
        leaf = parts[-1]
        if leaf == "weight" or leaf == "bias":
            return True
    return False


def _shape_of(tensor_like: Any) -> tuple[int, ...]:
    """Best-effort shape extraction from a tensor-like object."""
    shape = getattr(tensor_like, "shape", None)
    if shape is None:
        return ()
    return tuple(int(d) for d in shape)


def _verify_class_count_compatibility(
    model_state: Mapping[str, Any],
    ckpt_state: Mapping[str, Any],
) -> None:
    """Raise ``CheckpointCompatibilityError`` if any class-head key present in
    BOTH ``model_state`` and ``ckpt_state`` has a different shape.

    The class-count dimension lives on a different axis depending on the layer
    type (Linear weight → axis 0; Embedding → axis 0; bias → axis 0), so we
    compare the FULL shape: any mismatch on a class-head key signals an
    incompatible checkpoint. Keys present in only one side are NOT flagged
    here — they are handled by the subsequent ``strict=False`` load.
    """
    mismatched: list[str] = []
    for key, model_tensor in model_state.items():
        if not _is_class_head_key(key):
            continue
        if key not in ckpt_state:
            continue
        model_shape = _shape_of(model_tensor)
        ckpt_shape = _shape_of(ckpt_state[key])
        if not model_shape or not ckpt_shape:
            continue
        if ckpt_shape != model_shape:
            mismatched.append(
                f"{key} (checkpoint {list(ckpt_shape)} vs model {list(model_shape)})"
            )
    if mismatched:
        raise CheckpointCompatibilityError(
            "Checkpoint class-count is incompatible with the configured model "
            "for the following class-prediction head keys: "
            + "; ".join(mismatched)
            + ". Re-train the checkpoint for the configured number of classes "
            "or point the adapter at a matching checkpoint."
        )


# ===========================================================================
# DeimDetectionAdapter
# ===========================================================================


class DeimDetectionAdapter(DetectionAdapter):
    """DEIM detection adapter.

    Builds engine objects from a :class:`ResolvedAlgorithmConfig`, loads an
    optional checkpoint (with EMA preference and DDP-prefix normalization),
    verifies class-count compatibility, and exposes the deployed model +
    postprocessor. Inference (``predict``) lands in Task 6; ``train`` /
    ``evaluate`` / ``export`` land in Task 8.
    """

    def __init__(
        self,
        resolved: ResolvedAlgorithmConfig,
        loaded: LoadedAppConfig,
    ) -> None:
        self.resolved: ResolvedAlgorithmConfig = resolved
        self.loaded: LoadedAppConfig = loaded
        self.metadata = resolved.metadata
        self.box_mode: str = resolved.metadata.box_mode
        self._model = None
        self._postprocessor = None
        self._cfg = None

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        path: str | Path,
        cli_overrides: Mapping[str, object] | None = None,
    ) -> "DeimDetectionAdapter":
        """Build an adapter from a user application YAML.

        Delegates to :func:`load_app_config` then
        :func:`resolve_algorithm_config`. The adapter does not call ``load``
        here — callers do that explicitly so a failed checkpoint load never
        aborts construction.
        """
        from deim_app.config import load_app_config, resolve_algorithm_config

        loaded = load_app_config(path, cli_overrides)
        resolved = resolve_algorithm_config(loaded)
        return cls(resolved=resolved, loaded=loaded)

    def resolve_config(
        self, loaded: "LoadedAppConfig | None" = None
    ) -> "ResolvedAlgorithmConfig":
        """Re-resolve the algorithm config from ``loaded`` (default: self.loaded)."""
        from deim_app.config import resolve_algorithm_config

        target = self.loaded if loaded is None else loaded
        return resolve_algorithm_config(target)

    # ------------------------------------------------------------------
    # Engine object construction + checkpoint load
    # ------------------------------------------------------------------

    def load(
        self,
        checkpoint: str | Path | None = None,
        prefer_ema: bool = True,
    ) -> None:
        """Build engine objects and (optionally) load ``checkpoint``.

        Ordering (mirrors ``tools/inference/torch_inf.py`` and
        ``tools/deployment/export_onnx.py``):

          1. ``YAMLConfig(str(config_path), **resolved.overrides)``.
          2. Disable backbone pretrained download (``HGNetv2.pretrained=False``
             when present) so it cannot race the checkpoint load.
          3. If ``checkpoint`` is not ``None``: ``torch.load`` →
             :func:`select_model_state` (EMA preference + module-prefix strip).
          4. Verify class-count compatibility against the model's state_dict().
          5. ``model.load_state_dict(state, strict=False)`` — non-strict so a
             partial match (e.g. a tuning checkpoint) still loads cleanly.
          6. ``model.deploy()`` and ``postprocessor.deploy()``.

        When ``checkpoint`` is ``None`` steps 3–5 are skipped and the model is
        left at its default initialization (used for skeleton ONNX export).
        """
        # 1. Build engine objects.
        cfg = YAMLConfig(str(self.resolved.config_path), **self.resolved.overrides)
        self._cfg = cfg

        # 2. Disable backbone pretrained download. Mirrors
        #    tools/inference/torch_inf.py:117-118 and
        #    tools/deployment/export_onnx.py:28-29 — guarded so configs without
        #    HGNetv2 do not KeyError.
        yaml_cfg = getattr(cfg, "yaml_cfg", None)
        if isinstance(yaml_cfg, dict) and "HGNetv2" in yaml_cfg:
            hgnet = yaml_cfg["HGNetv2"]
            if isinstance(hgnet, dict):
                hgnet["pretrained"] = False

        # 3-5. Checkpoint load (only when a path is provided).
        if checkpoint is not None:
            raw_ckpt = torch.load(checkpoint, map_location="cpu")
            state = select_model_state(raw_ckpt, prefer_ema=prefer_ema)

            # 4. Class-count compatibility check (before load_state_dict).
            model_state = cfg.model.state_dict()
            _verify_class_count_compatibility(model_state, state)

            # 5. Load (non-strict — partial matches are acceptable).
            cfg.model.load_state_dict(state, strict=False)

        # 6. Deploy.
        self._model = cfg.model.deploy()
        self._postprocessor = cfg.postprocessor.deploy()

        # Keep public metadata attributes fresh from the resolved config.
        self.box_mode = self.resolved.metadata.box_mode
        self.metadata = self.resolved.metadata

    # ------------------------------------------------------------------
    # Stubs — implemented in Tasks 6 and 8.
    # ------------------------------------------------------------------

    def predict(
        self,
        source: Any,
        *,
        checkpoint: str | Path | None = None,
        device: str | None = None,
        batch_size: int | None = None,
    ) -> "PredictionCollection":
        """Run inference over ``source`` and return the full collection.

        ``source`` is anything :func:`deim_app.inference.inputs.list_inputs`
        accepts (a single image path, a directory, or an in-memory
        ``PIL.Image.Image``). The returned :class:`PredictionCollection` is
        UNFILTERED — score thresholding, top-k, and class filtering are facade
        / CLI responsibilities.

        ``device`` and ``batch_size`` default to the values in the loaded
        ``app.inference`` section when not supplied. ``checkpoint`` is accepted
        for protocol symmetry but the model must already be loaded via
        :meth:`load`; passing a different checkpoint here raises
        :class:`InferenceBackendError` (re-loading mid-predict is out of scope
        for v1).
        """
        from deim_app.errors import InferenceBackendError
        from deim_app.inference.inputs import list_inputs
        from deim_app.inference.preprocessing import Preprocessor
        from deim_app.inference.torch_backend import TorchBackend

        if checkpoint is not None:
            raise InferenceBackendError(
                "predict(checkpoint=...) is not supported; the model is already "
                "loaded via load(). Pass the checkpoint to load() instead."
            )
        if self._model is None or self._postprocessor is None:
            raise InferenceBackendError(
                "Model is not loaded. Call load(checkpoint) before predict()."
            )

        inference = self.loaded.app.inference
        runtime = self.loaded.app.runtime
        resolved_device = device if device is not None else inference.device
        resolved_batch_size = (
            batch_size if batch_size is not None else inference.batch_size
        )

        inputs = list_inputs(source)
        backend = TorchBackend(
            model=self._model,
            postprocessor=self._postprocessor,
            preprocessor=Preprocessor(runtime.input_size),
            metadata=self.metadata,
            box_mode=self.box_mode,
            device=resolved_device,
        )
        return backend.predict(inputs, batch_size=resolved_batch_size)

    def train(self) -> None:
        """Launch a training run. Implemented in Task 8."""
        raise NotImplementedError("Task 8 implements train")

    def evaluate(self, checkpoint: str | Path | None = None) -> None:
        """Run evaluation. Implemented in Task 8."""
        raise NotImplementedError("Task 8 implements evaluate")

    def supported_export_formats(self) -> tuple[str, ...]:
        """Return the export formats this adapter supports.

        Empty in the first application-layer version; ``export`` always raises
        :class:`ExportError` until Task 8 wires a real exporter.
        """
        return ()

    def export(
        self,
        checkpoint: str | Path,
        format: str,
        output: str | Path,
    ) -> Path:
        """Export the model. Implemented in Task 8."""
        raise ExportError(
            "No export format is enabled in the first application-layer version"
        )
