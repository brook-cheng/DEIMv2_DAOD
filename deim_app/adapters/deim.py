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
from engine.solver import TASKS
from engine.solver._solver import (
    assert_checkpoint_compat,
    classify_checkpoint_kind,
)

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
    """Raise ``CheckpointCompatibilityError`` when any class-head key the model
    requires is absent from the checkpoint or has a different shape.

    Two failure modes are flagged, both BEFORE ``load_state_dict`` runs:

      * A class-head key present in ``model_state`` but absent from
        ``ckpt_state``. Without this check the subsequent ``strict=False``
        load would silently leave the model's class heads at their random
        initialization — the model would run but its predictions would be
        garbage, which is the exact failure mode this adapter exists to make
        impossible.
      * A class-head key present in BOTH mappings whose shapes differ. The
        class-count dimension lives on a different axis depending on the
        layer type (Linear weight → axis 0; Embedding → axis 0; bias →
        axis 0), so the FULL shape is compared.

    Keys present ONLY in ``ckpt_state`` (checkpoint-only keys) are NOT
    flagged here — they are handled by the subsequent ``strict=False`` load
    and do not affect the model's class predictions.
    """
    missing: list[str] = []
    mismatched: list[str] = []
    for key, model_tensor in model_state.items():
        if not _is_class_head_key(key):
            continue
        if key not in ckpt_state:
            missing.append(key)
            continue
        model_shape = _shape_of(model_tensor)
        ckpt_shape = _shape_of(ckpt_state[key])
        if not model_shape or not ckpt_shape:
            continue
        if ckpt_shape != model_shape:
            mismatched.append(
                f"{key} (checkpoint {list(ckpt_shape)} vs model {list(model_shape)})"
            )
    problems: list[str] = []
    if missing:
        problems.append(
            "missing class-prediction head keys (present in model, absent from "
            "checkpoint): " + ", ".join(missing)
        )
    if mismatched:
        problems.append(
            "shape-incompatible class-prediction head keys: "
            + "; ".join(mismatched)
        )
    if problems:
        raise CheckpointCompatibilityError(
            "Checkpoint class-count is incompatible with the configured model "
            "for the following class-prediction head keys: "
            + "; ".join(problems)
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

    def _build_engine_cfg(self) -> YAMLConfig:
        """Construct a fresh ``YAMLConfig`` from the resolved config.

        Shared by :meth:`load`, :meth:`train`, and :meth:`evaluate` so they
        all build the engine config the same way:
        ``YAMLConfig(str(config_path), **resolved.overrides)``. Keeping the
        construction in one place guarantees the three entry points forward
        the same path and overrides — only their post-construction mutations
        (``device`` / ``resume``) and solver dispatch differ.
        """
        return YAMLConfig(
            str(self.resolved.config_path), **self.resolved.overrides
        )

    def load(
        self,
        checkpoint: str | Path | None = None,
        prefer_ema: bool = True,
    ) -> None:
        """Build engine objects and (optionally) load ``checkpoint``.

        Ordering (mirrors ``tools/inference/torch_inf.py`` and
        ``tools/deployment/export_onnx.py``):

          1. ``YAMLConfig(str(config_path), **resolved.overrides)`` via
             :meth:`_build_engine_cfg`.
          2. Disable backbone pretrained download (``HGNetv2.pretrained=False``
             when present) so it cannot race the checkpoint load.
          3. If ``checkpoint`` is not ``None``: ``torch.load`` →
             :func:`select_model_state` (EMA preference + module-prefix strip).
          4. OBB shifted_v1 contract gate (OBB app configs only): when
             ``resolved.metadata.box_mode == 'obb'``, classify the RAW
             checkpoint and — unless it is identifiable 4-D HBB pretraining —
             require ``meta.obb_angle_contract = "shifted_v1"``. Runs on the
             raw checkpoint (before :func:`select_model_state` strips ``meta``)
             and reuses the engine solver's helpers so marker semantics are
             defined in exactly one place. ``CheckpointIncompatibleError``
             propagates explicitly; HBB app configs skip the gate entirely.
          5. Verify class-count compatibility against the model's state_dict().
          6. ``model.load_state_dict(state, strict=False)`` — non-strict so a
             partial match (e.g. a tuning checkpoint) still loads cleanly.
          7. ``model.deploy()`` and ``postprocessor.deploy()``.
          8. ``model.to(inference.device)`` and ``postprocessor.to(inference.device)``
             — moves both deployed modules to the configured inference device.

        When ``checkpoint`` is ``None`` steps 3–6 are skipped and the model is
        left at its default initialization (used for skeleton ONNX export).
        """
        # 1. Build engine objects.
        cfg = self._build_engine_cfg()
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

        # 3-6. Checkpoint load (only when a path is provided).
        if checkpoint is not None:
            # weights_only=True: checkpoints are data, not code (CWE-502); the
            # engine format is tensors + plain primitives, so this never needs pickle.
            raw_ckpt = torch.load(
                checkpoint, map_location="cpu", weights_only=True
            )

            # 4. OBB shifted_v1 contract gate — only for OBB app configs, on the
            #    RAW checkpoint so ``meta`` is retained. 4-D HBB pretraining is
            #    always an acceptable OBB tuning source; any other head needs
            #    the marker. HBB app configs never enforce the OBB marker.
            if self.resolved.metadata.box_mode == "obb":
                if classify_checkpoint_kind(raw_ckpt) != "hbb":
                    assert_checkpoint_compat(raw_ckpt)

            state = select_model_state(raw_ckpt, prefer_ema=prefer_ema)

            # 5. Class-count compatibility check (before load_state_dict).
            model_state = cfg.model.state_dict()
            _verify_class_count_compatibility(model_state, state)

            # 6. Load (non-strict — partial matches are acceptable).
            cfg.model.load_state_dict(state, strict=False)

        # 6. Deploy.
        self._model = cfg.model.deploy()
        self._postprocessor = cfg.postprocessor.deploy()

        # 7. Move deployed modules to the configured inference device. Runs
        #    AFTER deploy() because deploy() returns self — the move must act
        #    on the deployed module identity. Mirrors
        #    tools/inference/torch_inf.py placing ``model.to(device)`` once the
        #    model is built; without this a CUDA-configured app silently runs
        #    on CPU (the engine default).
        inference_device = self.loaded.app.inference.device
        self._model.to(inference_device)
        self._postprocessor.to(inference_device)

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
        """Build the solver and run one full training cycle via ``solver.fit()``.

        Does NOT call subprocess. Builds the solver from a freshly constructed
        :class:`~engine.core.YAMLConfig` (via :meth:`_build_engine_cfg`),
        registers it through ``TASKS[cfg.yaml_cfg['task']]``, applies
        ``app.train.device`` so ``BaseSolver._setup`` places the model on the
        configured device, then calls ``fit()`` — which internally runs
        ``train()`` (setup + resume-from-``cfg.resume``) and the epoch loop.
        """
        cfg = self._build_engine_cfg()
        cfg.device = self.loaded.app.train.device
        solver = TASKS[cfg.yaml_cfg["task"]](cfg)
        solver.fit()

    def evaluate(self, checkpoint: str | Path | None = None) -> None:
        """Build the solver and run one evaluation pass via ``solver.val()``.

        When ``checkpoint`` is provided, set ``cfg.resume = str(checkpoint)``
        so ``BaseSolver.eval`` loads it. When ``checkpoint`` is ``None`` the
        adapter leaves any preset / config-provided ``cfg.resume`` untouched
        (preserving the resume behavior the engine already implements).
        Applies ``app.evaluation.device`` regardless.
        """
        cfg = self._build_engine_cfg()
        if checkpoint is not None:
            cfg.resume = str(checkpoint)
        cfg.device = self.loaded.app.evaluation.device
        solver = TASKS[cfg.yaml_cfg["task"]](cfg)
        solver.val()

    def supported_export_formats(self) -> tuple[str, ...]:
        """Return the export formats this adapter supports.

        No export format is enabled in the first application-layer version;
        :meth:`export` always raises :class:`ExportError`.
        """
        return ()

    def export(
        self,
        checkpoint: str | Path,
        format: str,
        output: str | Path,
    ) -> Path:
        """Export the model. v1 always raises — no export format is enabled."""
        raise ExportError(
            "No export format is enabled in the first application-layer version"
        )
