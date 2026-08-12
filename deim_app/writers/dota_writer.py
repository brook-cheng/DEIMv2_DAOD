"""DOTA writer: emit per-image OBB ``.txt`` files in DOTA 1.0 polygon format.

Each output file is named ``<image_id_stem>.txt`` and contains one line per
detection::

    x1 y1 x2 y2 x3 y3 x4 y4 class_name score

Polygon conversion flows through ``deim_app.adapters.geometry.obb_to_polygon``
(never ``engine.*`` directly). HBB collections are rejected with ``ExportError``;
callers wanting HBB → DOTA must convert geometries upstream (out of scope here).
"""

from __future__ import annotations

import os
from pathlib import Path

from deim_app.errors import ExportError
from deim_app.predictions.collection import PredictionCollection
from deim_app.predictions.types import OBBDetection

# Geometry must come from the adapter, never from the engine directly.
from deim_app.adapters.geometry import obb_to_polygon


def write_dota(collection: PredictionCollection, path: Path) -> Path:
    if collection.box_mode != "obb":
        raise ExportError(
            f"DOTA export requires box_mode='obb'; got {collection.box_mode!r}"
        )

    path.mkdir(parents=True, exist_ok=True)

    for pred in collection.predictions:
        stem = os.path.splitext(os.path.basename(pred.image_id))[0]
        out_path = path / f"{stem}.txt"
        lines: list[str] = []
        for det in pred.detections:
            if not isinstance(det, OBBDetection):
                # Mixed collections are a programmer error; DOTA is OBB-only.
                raise ExportError(
                    f"DOTA export got non-OBB detection in OBB collection: "
                    f"{type(det).__name__}"
                )
            poly = obb_to_polygon(det.xywhr)
            coords = " ".join(f"{v:.6f}" for v in poly)
            lines.append(f"{coords} {det.class_name} {det.score:.6f}")
        out_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    return path
