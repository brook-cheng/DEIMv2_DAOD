"""Batched PyTorch inference backend.

:class:`TorchBackend` runs a deployed DEIM model + postprocessor over a
sequence of :class:`~deim_app.inference.inputs.InputImage` inputs and returns a
fully-populated :class:`~deim_app.predictions.collection.PredictionCollection`.
No score thresholding, top-k, or class filtering happens here — those are
facade / CLI responsibilities (Tasks 7 and 9).

Postprocessor contract (verified against
``engine/deim/postprocessor.py:PostProcessor.forward``):

  * ``orig_target_sizes`` is shape ``(N, 2)`` interpreted as ``[width, height]``
    per image. For HBB it does ``bbox_pred *= orig_target_sizes.repeat(1, 2)``
    (multiplying xyxy by ``[w, h, w, h]``); for OBB it indexes
    ``orig_target_sizes[:, 0:1]`` as ``img_w`` and ``[:, 1:2]`` as ``img_h``.
  * In ``deploy_mode`` the postprocessor returns ``(labels, boxes, scores)``
    where boxes are ALREADY in original-image pixel coordinates. The backend
    MUST NOT rescale HBB or OBB boxes after postprocessing.

Class-name lookup: always uses ``metadata.class_names_by_label`` keyed by the
raw model label (``0..N-1``). In ``deploy_mode`` the postprocessor returns at
its ``deploy_mode`` guard *before* the ``remap_mscoco_category`` branch, so the
emitted labels are always contiguous model labels — never dataset-native
category ids. ``output_names_by_id`` is intentionally NOT consulted here.

Boundary: this module imports only from ``deim_app``, ``torch``, and the
standard library — never from ``engine``.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Literal

import torch
import torch.nn as nn

from deim_app.config.metadata import DatasetMetadata
from deim_app.inference.inputs import InputImage
from deim_app.inference.preprocessing import Preprocessor
from deim_app.predictions.collection import PredictionCollection
from deim_app.predictions.types import (
    Detection,
    HBBDetection,
    ImagePrediction,
    OBBDetection,
    Timings,
)

__all__ = ["TorchBackend"]


BoxMode = Literal["hbb", "obb"]


class TorchBackend:
    """Batched PyTorch inference over a deployed DEIM model.

    Args:
        model: A deployed DEIM ``nn.Module`` whose ``forward`` accepts a
            ``(B, 3, H, W)`` float tensor and returns a dict with
            ``pred_logits`` and ``pred_boxes``.
        postprocessor: A deployed ``PostProcessor`` whose ``forward`` accepts
            ``(outputs, orig_target_sizes)`` and returns the deploy tuple
            ``(labels, boxes, scores)``.
        preprocessor: The canonical :class:`Preprocessor` used to prepare each
            input.
        metadata: Dataset metadata providing the class-name mappings.
        box_mode: ``"hbb"`` or ``"obb"`` — selects the detection dataclass and
            the expected box cardinality (4 for HBB xyxy, 5 for OBB cxcywhθ).
        device: Device string for the model input tensor (default ``"cuda"``).
    """

    def __init__(
        self,
        model: nn.Module,
        postprocessor: nn.Module,
        preprocessor: Preprocessor,
        metadata: DatasetMetadata,
        box_mode: str,
        device: str = "cuda",
    ) -> None:
        if box_mode not in ("hbb", "obb"):
            raise ValueError(
                f"box_mode must be 'hbb' or 'obb', got {box_mode!r}"
            )
        self._model = model
        self._postprocessor = postprocessor
        self._preprocessor = preprocessor
        self._metadata = metadata
        self._box_mode: BoxMode = box_mode  # validated above
        self._device = device

    @property
    def box_mode(self) -> BoxMode:
        return self._box_mode

    def predict(
        self,
        inputs: Sequence[InputImage],
        batch_size: int = 1,
    ) -> PredictionCollection:
        """Run batched inference and return the FULL unfiltered collection.

        Inputs are chunked into batches of ``batch_size`` (the final batch may
        be smaller). Within each batch:

          1. Each input is preprocessed (timed per image).
          2. Tensors are stacked into ``(B, 3, H, W)`` and moved to ``device``.
          3. ``orig_target_sizes`` is built as ``(B, 2)`` using ``[width,
             height]`` per image.
          4. The model is run (timed once per batch).
          5. The postprocessor is run (timed once per batch) →
             ``(labels, boxes, scores)``.
          6. Per-image detections are built from the postprocessor output.

        No result filtering or file writing occurs here.
        """
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")

        predictions: list[ImagePrediction] = []
        for batch in _chunked(inputs, batch_size):
            predictions.extend(self._run_batch(batch))
        return PredictionCollection(
            box_mode=self._box_mode,
            predictions=tuple(predictions),
        )

    # ------------------------------------------------------------------
    # Per-batch execution
    # ------------------------------------------------------------------

    def _run_batch(self, batch: tuple[InputImage, ...]) -> tuple[ImagePrediction, ...]:
        # 1. Preprocess (per-image timing).
        prepared = []
        preprocess_times: list[float] = []
        for inp in batch:
            start = time.perf_counter()
            prepared.append(self._preprocessor(inp))
            preprocess_times.append(time.perf_counter() - start)

        # 2. Stack into (B, 3, H, W) and move to device.
        batch_tensor = torch.stack(
            [p.tensor.squeeze(0) for p in prepared]
        ).to(self._device)

        # 3. orig_target_sizes: (B, 2) as [width, height] per image.
        #    PreparedImage.original_size_hw is (height, width) → swap.
        orig_target_sizes = torch.tensor(
            [[p.original_size_hw[1], p.original_size_hw[0]] for p in prepared],
            dtype=torch.float32,
            device=self._device,
        )

        # 4. Model forward (per-batch timing) under inference_mode so no
        #    autograd graph is built for the deployed pass.
        inf_start = time.perf_counter()
        with torch.inference_mode():
            outputs = self._model(batch_tensor)
        inference_s = time.perf_counter() - inf_start

        # 5. Postprocessor (per-batch timing) → deploy tuple. Also under
        #    inference_mode — its tensor ops would otherwise retain history
        #    anchored to the model outputs.
        post_start = time.perf_counter()
        with torch.inference_mode():
            post_out = self._postprocessor(outputs, orig_target_sizes)
        postprocess_s = time.perf_counter() - post_start
        labels, boxes, scores = post_out

        # 6. Build per-image detections.
        image_predictions: list[ImagePrediction] = []
        for i, prep in enumerate(prepared):
            detections = self._build_detections(
                labels[i], boxes[i], scores[i]
            )
            image_predictions.append(
                ImagePrediction(
                    image_id=prep.image_id,
                    source=prep.source,
                    original_image=prep.original_image,
                    original_size=prep.original_size_hw,
                    detections=detections,
                    timings=Timings(
                        preprocess_s=preprocess_times[i],
                        inference_s=inference_s,
                        postprocess_s=postprocess_s,
                    ),
                )
            )
        return tuple(image_predictions)

    # ------------------------------------------------------------------
    # Detection construction + class-name lookup
    # ------------------------------------------------------------------

    def _build_detections(
        self,
        image_labels: torch.Tensor,
        image_boxes: torch.Tensor,
        image_scores: torch.Tensor,
    ) -> tuple[Detection, ...]:
        """Convert one image's postprocessor output into typed detections.

        HBB → :class:`HBBDetection` (xyxy, 4 values).
        OBB → :class:`OBBDetection` (cxcywhθ, 5 values).

        Boxes are copied verbatim from the postprocessor output — no rescale.
        """
        num = int(image_labels.shape[0])
        detections: list[Detection] = []
        for k in range(num):
            label_id = int(image_labels[k].item())
            score = float(image_scores[k].item())
            class_name = self._class_name_for_label(label_id)
            coords = [float(v) for v in image_boxes[k].tolist()]
            if self._box_mode == "hbb":
                detections.append(
                    HBBDetection(
                        class_id=label_id,
                        class_name=class_name,
                        score=score,
                        xyxy=(
                            coords[0],
                            coords[1],
                            coords[2],
                            coords[3],
                        ),
                    )
                )
            else:
                detections.append(
                    OBBDetection(
                        class_id=label_id,
                        class_name=class_name,
                        score=score,
                        xywhr=(
                            coords[0],
                            coords[1],
                            coords[2],
                            coords[3],
                            coords[4],
                        ),
                    )
                )
        return tuple(detections)

    def _class_name_for_label(self, label_id: int) -> str:
        """Resolve a deployed label id to its class name.

        The deployed :class:`PostProcessor` returns raw model labels
        (``0..N-1``) — the ``remap_mscoco_category`` branch does not execute in
        ``deploy_mode`` (see ``engine/deim/postprocessor.py``: it returns at the
        ``if self.deploy_mode`` guard, *before* the remap block). Therefore
        ``class_names_by_label`` — which keys on contiguous model labels — is
        the correct mapping for every case, and ``output_names_by_id`` (which
        keys on dataset-native category ids, e.g. remapped MS COCO ids) must
        NEVER be consulted here. Using ``output_names_by_id`` would silently
        misname every label for a deployed MS COCO HBB model (e.g. label 1 →
        ``"person"`` instead of ``"bicycle"``).
        """
        return self._metadata.class_names_by_label[label_id]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chunked(seq: Sequence[InputImage], size: int) -> tuple[tuple[InputImage, ...], ...]:
    """Split ``seq`` into contiguous tuples of length ``size`` (last may be shorter).

    Deterministic: 3 inputs / size 2 → ``((0, 1), (2,))``. Empty input → ``()``.
    """
    return tuple(tuple(seq[i : i + size]) for i in range(0, len(seq), size))
