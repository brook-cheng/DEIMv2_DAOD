"""Shared fixtures for ``deim_app.predictions`` tests.

Provides the ``make_obb_collection`` / ``make_hbb_collection`` helpers used by
both ``test_collection.py`` and ``test_writers.py``. The helpers build small,
deterministic collections backed by a synthetic 8x8 RGB PIL image so writers
can be exercised end-to-end without touching the filesystem beyond ``tmp_path``.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import pytest
from PIL import Image

from deim_app.predictions.collection import PredictionCollection
from deim_app.predictions.types import (
    HBBDetection,
    ImagePrediction,
    OBBDetection,
    Timings,
)


def _blank_image(size: tuple[int, int] = (8, 8)) -> Image.Image:
    """Return a fresh RGB PIL image (so per-image mutations stay isolated)."""
    return Image.new("RGB", size, (0, 0, 0))


def _make_image_prediction(
    box_mode: str,
    scores: Sequence[float],
    class_names: Sequence[str] | None = None,
    image_id: str = "img0",
) -> ImagePrediction:
    if class_names is None:
        class_names = tuple(f"c{i}" for i in range(len(scores)))
    detections = []
    for i, s in enumerate(scores):
        name = class_names[i] if i < len(class_names) else f"c{i}"
        if box_mode == "hbb":
            detections.append(
                HBBDetection(
                    class_id=i,
                    class_name=name,
                    score=float(s),
                    xyxy=(float(i), float(i), float(i + 1), float(i + 1)),
                )
            )
        else:
            # xywhr: cx, cy, w, h, theta in [0, pi)
            detections.append(
                OBBDetection(
                    class_id=i,
                    class_name=name,
                    score=float(s),
                    xywhr=(
                        float(i + 1) * 0.5,
                        float(i + 1) * 0.5,
                        1.0,
                        1.0,
                        0.0,
                    ),
                )
            )
    return ImagePrediction(
        image_id=image_id,
        source=f"/synthetic/{image_id}.png",
        original_image=_blank_image(),
        original_size=(8, 8),
        detections=tuple(detections),
        timings=Timings(preprocess_s=0.01, inference_s=0.02, postprocess_s=0.03),
    )


def make_obb_collection(
    scores: Iterable[float] = (0.9, 0.2),
    class_names: Sequence[str] | None = None,
    image_ids: Sequence[str] | None = None,
) -> PredictionCollection:
    """Build a small OBB ``PredictionCollection``.

    Each entry in ``scores`` becomes one ``OBBDetection`` inside the first image
    (when ``image_ids`` has one entry) or one image (when ``image_ids`` matches
    the length of ``scores``).
    """
    scores_t = tuple(float(s) for s in scores)
    if image_ids is not None and len(image_ids) == len(scores_t):
        predictions = tuple(
            _make_image_prediction(
                "obb",
                (s,),
                class_names=(class_names[i],) if class_names else None,
                image_id=image_ids[i],
            )
            for i, s in enumerate(scores_t)
        )
    else:
        predictions = (
            _make_image_prediction("obb", scores_t, class_names=class_names),
        )
    return PredictionCollection(box_mode="obb", predictions=predictions)


def make_hbb_collection(
    scores: Iterable[float] = (0.3, 0.9, 0.7),
    class_names: Sequence[str] | None = None,
    image_ids: Sequence[str] | None = None,
) -> PredictionCollection:
    """Build a small HBB ``PredictionCollection`` (see ``make_obb_collection``)."""
    scores_t = tuple(float(s) for s in scores)
    if image_ids is not None and len(image_ids) == len(scores_t):
        predictions = tuple(
            _make_image_prediction(
                "hbb",
                (s,),
                class_names=(class_names[i],) if class_names else None,
                image_id=image_ids[i],
            )
            for i, s in enumerate(scores_t)
        )
    else:
        predictions = (
            _make_image_prediction("hbb", scores_t, class_names=class_names),
        )
    return PredictionCollection(box_mode="hbb", predictions=predictions)


@pytest.fixture
def tmp_image_path(tmp_path) -> str:
    """Return a writable image-file path under ``tmp_path`` (no file created)."""
    return str(tmp_path / "synthetic.png")
