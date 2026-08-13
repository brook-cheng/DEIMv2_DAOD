"""Visualization writer: draw detections onto per-image copies and save to disk.

OBB drawing flows through ``deim_app.adapters.geometry.draw_obb_detections``
(never ``engine.*`` / ``tools.model_compare.*`` directly). HBB drawing is inline
``PIL.ImageDraw.rectangle`` — HBB has no engine geometry dependency, so no
adapter wrapper is needed.

The ``score_threshold`` is applied AT DRAW TIME only — the underlying collection
is never mutated (every detection stays in place for subsequent JSON / DOTA
exports).
"""

from __future__ import annotations

import os
from pathlib import Path

from PIL import Image, ImageDraw

from deim_app.adapters.geometry import draw_obb_detections
from deim_app.predictions.types import (
    HBBDetection,
    ImagePrediction,
    OBBDetection,
    PredictionCollectionLike,
)


def _ext_for(image_id: str) -> str:
    base = os.path.basename(image_id)
    if "." in base:
        return "." + base.rsplit(".", 1)[1]
    return ".png"


def _draw_hbb_inline(
    image: Image.Image,
    pred: ImagePrediction,
    color: tuple[int, int, int],
    line_width: int,
    alpha: float,
    score_threshold: float,
) -> Image.Image:
    """Draw HBB rectangles with PIL directly (no engine import needed)."""
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas, "RGBA")
    fill_color = None if alpha == 0 else (color[0], color[1], color[2], int(255 * alpha))
    for det in pred.detections:
        if not isinstance(det, HBBDetection):
            continue
        if det.score < score_threshold:
            continue
        x1, y1, x2, y2 = det.xyxy
        draw.rectangle([x1, y1, x2, y2], outline=color, fill=fill_color, width=line_width)
        label = f"{det.class_name} {det.score:.2f}"
        draw.text((x1, max(0, y1 - 12)), label, fill=color)
    return canvas


def write_visualization(
    collection: PredictionCollectionLike,
    output_dir: Path,
    *,
    color: tuple[int, int, int] = (255, 0, 0),
    line_width: int = 2,
    alpha: float = 0.3,
    score_threshold: float = 0.0,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    for pred in collection.predictions:
        ext = _ext_for(pred.image_id)
        out_path = output_dir / f"{os.path.splitext(os.path.basename(pred.image_id))[0]}{ext}"

        if collection.box_mode == "obb":
            obb_dets = [d for d in pred.detections if isinstance(d, OBBDetection)]
            canvas = draw_obb_detections(
                pred.original_image,
                obb_dets,
                color=color,
                line_width=line_width,
                alpha=alpha,
                score_threshold=score_threshold,
            )
        else:
            canvas = _draw_hbb_inline(
                pred.original_image,
                pred,
                color=color,
                line_width=line_width,
                alpha=alpha,
                score_threshold=score_threshold,
            )
        canvas.convert("RGB").save(out_path)

    return output_dir
