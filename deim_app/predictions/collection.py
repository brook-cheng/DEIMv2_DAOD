"""Immutable collection over ``ImagePrediction`` with non-mutating helpers.

``PredictionCollection.filter`` and ``PredictionCollection.top_k`` ALWAYS return
new collections; they never mutate the receiver. Empty per-image detections are
preserved on ``filter`` so caller indices stay stable. Exporters
(``export_json`` / ``export_dota`` / ``save_images``) import their writer inside
the method body to avoid an import cycle (writers import this module's types).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

from deim_app.errors import ExportError
from deim_app.predictions.types import Detection, ImagePrediction


@dataclass(frozen=True, slots=True)
class PredictionCollection:
    """Frozen tuple-of-``ImagePrediction`` plus the box mode shared by all entries."""

    box_mode: Literal["hbb", "obb"]
    predictions: tuple[ImagePrediction, ...]

    def __post_init__(self) -> None:
        if self.box_mode not in ("hbb", "obb"):
            raise ValueError(f"box_mode must be 'hbb' or 'obb', got {self.box_mode!r}")

    # ------------------------------------------------------------------
    # Non-mutating filters
    # ------------------------------------------------------------------

    def filter(
        self,
        *,
        score_threshold: float | None = None,
        class_names: Iterable[str] | None = None,
        class_ids: Iterable[int] | None = None,
    ) -> "PredictionCollection":
        """Return a NEW collection whose detections match every supplied filter.

        - ``score_threshold``: keep detections with ``score >= threshold``.
        - ``class_names``: keep detections whose ``class_name`` is in the set.
        - ``class_ids``: keep detections whose ``class_id`` is in the set.

        Images with zero surviving detections remain present with an empty
        ``detections`` tuple (so caller-visible indices do not shift). The
        receiver is never mutated.
        """
        name_set = set(class_names) if class_names is not None else None
        id_set = set(class_ids) if class_ids is not None else None

        new_predictions: list[ImagePrediction] = []
        for pred in self.predictions:
            kept: tuple[Detection, ...] = tuple(
                d
                for d in pred.detections
                if _passes(d, score_threshold, name_set, id_set)
            )
            new_predictions.append(
                ImagePrediction(
                    image_id=pred.image_id,
                    source=pred.source,
                    original_image=pred.original_image,
                    original_size=pred.original_size,
                    detections=kept,
                    timings=pred.timings,
                )
            )
        return PredictionCollection(box_mode=self.box_mode, predictions=tuple(new_predictions))

    def top_k(self, k: int) -> "PredictionCollection":
        """Return a NEW collection with at most ``k`` detections per image.

        Detections are sorted by score descending (stable on equal scores) and
        truncated to ``k`` per image. Image order is preserved. ``k < 0`` raises
        ``ValueError``; ``k == 0`` yields empty detections per image.
        """
        if k < 0:
            raise ValueError(f"k must be non-negative, got {k}")

        new_predictions: list[ImagePrediction] = []
        for pred in self.predictions:
            ordered = sorted(pred.detections, key=lambda d: d.score, reverse=True)
            kept = tuple(ordered[:k])
            new_predictions.append(
                ImagePrediction(
                    image_id=pred.image_id,
                    source=pred.source,
                    original_image=pred.original_image,
                    original_size=pred.original_size,
                    detections=kept,
                    timings=pred.timings,
                )
            )
        return PredictionCollection(box_mode=self.box_mode, predictions=tuple(new_predictions))

    # ------------------------------------------------------------------
    # Exports — writers imported lazily inside method bodies (avoid cycles).
    # ------------------------------------------------------------------

    def export_json(self, path: str | Path) -> Path:
        from deim_app.writers.json_writer import write_json

        return write_json(self, Path(path))

    def export_dota(self, path: str | Path) -> Path:
        if self.box_mode != "obb":
            raise ExportError(
                f"DOTA export requires box_mode='obb'; got {self.box_mode!r}"
            )
        from deim_app.writers.dota_writer import write_dota

        return write_dota(self, Path(path))

    def save_images(
        self,
        output_dir: str | Path,
        *,
        color: tuple = (255, 0, 0),
        line_width: int = 2,
        alpha: float = 0.3,
        score_threshold: float = 0.0,
    ) -> Path:
        from deim_app.writers.visualization import write_visualization

        return write_visualization(
            self,
            Path(output_dir),
            color=color,
            line_width=line_width,
            alpha=alpha,
            score_threshold=score_threshold,
        )


def _passes(
    det: Detection,
    score_threshold: float | None,
    name_set: set[str] | None,
    id_set: set[int] | None,
) -> bool:
    if score_threshold is not None and det.score < score_threshold:
        return False
    if name_set is not None and det.class_name not in name_set:
        return False
    if id_set is not None and det.class_id not in id_set:
        return False
    return True


__all__ = ["PredictionCollection"]
