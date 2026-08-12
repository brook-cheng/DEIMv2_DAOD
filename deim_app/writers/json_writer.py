"""JSON writer: serialise a ``PredictionCollection`` to a per-image JSON array.

No pixel data is emitted. Each detection carries an explicit ``box_mode``
(``"hbb"`` or ``"obb"``); HBB detections write ``xyxy``, OBB detections write
``xywhr``. This module imports ONLY from ``deim_app`` — never from ``engine``
or ``tools.model_compare`` (enforced by the dep-guard test).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from deim_app.predictions.collection import PredictionCollection
from deim_app.predictions.types import HBBDetection, OBBDetection


def _detection_to_dict(det: Any) -> dict[str, Any]:
    if isinstance(det, HBBDetection):
        return {
            "box_mode": "hbb",
            "class_id": det.class_id,
            "class_name": det.class_name,
            "score": float(det.score),
            "xyxy": list(det.xyxy),
        }
    if isinstance(det, OBBDetection):
        return {
            "box_mode": "obb",
            "class_id": det.class_id,
            "class_name": det.class_name,
            "score": float(det.score),
            "xywhr": list(det.xywhr),
        }
    raise TypeError(f"Unsupported detection type: {type(det).__name__}")


def _image_to_dict(pred: Any) -> dict[str, Any]:
    return {
        "image_id": pred.image_id,
        "source": pred.source,
        "original_size": list(pred.original_size),
        "detections": [_detection_to_dict(d) for d in pred.detections],
        "timings": {
            "preprocess_s": float(pred.timings.preprocess_s),
            "inference_s": float(pred.timings.inference_s),
            "postprocess_s": float(pred.timings.postprocess_s),
        },
    }


def write_json(collection: PredictionCollection, path: Path) -> Path:
    """Serialise ``collection`` to ``path`` as a JSON array (one entry per image)."""
    payload = [_image_to_dict(p) for p in collection.predictions]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
