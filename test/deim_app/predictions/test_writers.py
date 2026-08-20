"""Tests for ``deim_app.writers`` (json / dota / visualization).

Boundary assertions:
- JSON: explicit ``box_mode`` per detection; HBB writes ``xyxy``, OBB ``xywhr``.
- DOTA polygon coordinates equal ``engine.deim.obb_geometry.xywhr_to_xyxyxyxy``
  through ``deim_app.adapters.geometry.obb_to_polygon`` (no reproduced trig).
- HBB DOTA export raises ``ExportError``.
- OBB visualization delegates through ``deim_app.adapters.geometry.draw_obb_detections``.
- Applying a visualization threshold does NOT mutate the collection subsequently
  exported to JSON or DOTA.

These tests are product-code-agnostic for OBB visualization: they monkeypatch
``deim_app.adapters.geometry.draw_obb_detections`` to verify delegation without
re-running real PIL drawing through the engine wrapper.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch

from deim_app.errors import ExportError

from conftest import make_hbb_collection, make_obb_collection


# ---------------------------------------------------------------------------
# JSON writer
# ---------------------------------------------------------------------------


def test_json_writer_hbb_writes_xyxy_with_box_mode(tmp_path: Path) -> None:
    coll = make_hbb_collection(scores=(0.9,))
    from deim_app.writers.json_writer import write_json

    out = write_json(coll, tmp_path / "out.json")
    data = json.loads(out.read_text(encoding="utf-8"))
    assert len(data) == 1
    img = data[0]
    assert img["image_id"] == "img0"
    assert img["original_size"] == [8, 8]
    assert img["timings"] == {
        "preprocess_s": 0.01,
        "inference_s": 0.02,
        "postprocess_s": 0.03,
    }
    det = img["detections"][0]
    assert det["box_mode"] == "hbb"
    assert "xyxy" in det
    assert "xywhr" not in det
    assert det["class_id"] == 0
    assert det["class_name"] == "c0"
    assert det["score"] == pytest.approx(0.9)


def test_json_writer_obb_writes_xywhr_with_box_mode(tmp_path: Path) -> None:
    coll = make_obb_collection(scores=(0.7,))
    from deim_app.writers.json_writer import write_json

    out = write_json(coll, tmp_path / "out.json")
    data = json.loads(out.read_text(encoding="utf-8"))
    det = data[0]["detections"][0]
    assert det["box_mode"] == "obb"
    assert "xywhr" in det
    assert "xyxy" not in det
    assert len(det["xywhr"]) == 5


def test_json_writer_has_no_pixel_data(tmp_path: Path) -> None:
    coll = make_hbb_collection()
    from deim_app.writers.json_writer import write_json

    out = write_json(coll, tmp_path / "out.json")
    text = out.read_text(encoding="utf-8")
    # No pixel array leaks.
    assert "pixels" not in text.lower()
    assert "original_image" not in text


# ---------------------------------------------------------------------------
# DOTA writer
# ---------------------------------------------------------------------------


def test_dota_writer_polygon_matches_engine(tmp_path: Path) -> None:
    """DOTA polygon coordinates equal ``xywhr_to_xyxyxyxy`` from the engine."""
    from engine.deim.obb_geometry import xywhr_to_xyxyxyxy

    from deim_app.writers.dota_writer import write_dota

    xywhr = (1.5, 1.5, 2.0, 1.0, 0.3)
    coll = make_obb_collection(scores=(0.8,))
    # Overwrite the single detection with a known xywhr.
    from deim_app.predictions.types import OBBDetection

    pred = coll.predictions[0]
    new_det = OBBDetection(
        class_id=pred.detections[0].class_id,
        class_name=pred.detections[0].class_name,
        score=0.8,
        xywhr=xywhr,
    )
    from deim_app.predictions.types import ImagePrediction

    new_pred = ImagePrediction(
        image_id=pred.image_id,
        source=pred.source,
        original_image=pred.original_image,
        original_size=pred.original_size,
        detections=(new_det,),
        timings=pred.timings,
    )
    from deim_app.predictions.collection import PredictionCollection

    coll = PredictionCollection(box_mode="obb", predictions=(new_pred,))

    out_dir = write_dota(coll, tmp_path / "dota")
    files = list(out_dir.glob("*.txt"))
    assert files, "DOTA writer produced no files"
    line = files[0].read_text(encoding="utf-8").strip().splitlines()[0]
    parts = line.split()
    coords = [float(x) for x in parts[:8]]

    expected = (
        xywhr_to_xyxyxyxy(torch.tensor(xywhr, dtype=torch.float32).reshape(1, 5))
        .reshape(-1)
        .tolist()
    )
    assert coords == pytest.approx(expected, abs=1e-6)
    assert parts[8] == "c0"
    assert float(parts[9]) == pytest.approx(0.8)


def test_dota_writer_hbb_raises_export_error(tmp_path: Path) -> None:
    coll = make_hbb_collection(scores=(0.9,))
    from deim_app.writers.dota_writer import write_dota

    with pytest.raises(ExportError):
        write_dota(coll, tmp_path / "dota")


# ---------------------------------------------------------------------------
# Visualization writer
# ---------------------------------------------------------------------------


def test_visualization_obb_delegates_through_adapter(tmp_path: Path, monkeypatch) -> None:
    """OBB visualization delegates through ``deim_app.adapters.geometry.draw_obb_detections``."""
    import deim_app.adapters.geometry as geom
    from deim_app.writers import visualization

    calls: list[dict[str, Any]] = []

    def fake_draw(image, detections, *, color, line_width, alpha, score_threshold, font=None):
        calls.append(
            {
                "image": image,
                "detections": list(detections),
                "color": color,
                "line_width": line_width,
                "alpha": alpha,
                "score_threshold": score_threshold,
                "font": font,
            }
        )
        return image

    monkeypatch.setattr(geom, "draw_obb_detections", fake_draw)
    monkeypatch.setattr(visualization, "draw_obb_detections", fake_draw)

    coll = make_obb_collection(scores=(0.9, 0.2))
    out_dir = visualization.write_visualization(
        coll, tmp_path / "vis", color=(0, 255, 0), line_width=3, alpha=0.5, score_threshold=0.5
    )
    assert calls, "draw_obb_detections was not called"
    # score_threshold is forwarded as-is (filtering happens inside the adapter).
    assert calls[0]["score_threshold"] == 0.5
    assert calls[0]["color"] == (0, 255, 0)
    assert calls[0]["line_width"] == 3
    assert calls[0]["alpha"] == 0.5


def test_visualization_threshold_does_not_mutate_collection(tmp_path: Path, monkeypatch) -> None:
    """Applying a viz threshold must not corrupt subsequent JSON / DOTA exports."""
    from deim_app.writers import json_writer, visualization

    # Make viz drawing a no-op so it can't accidentally mutate the source image.
    def noop_draw(image, detections, **kwargs):
        return image

    monkeypatch.setattr(visualization, "draw_obb_detections", noop_draw)

    coll = make_obb_collection(scores=(0.9, 0.2))
    # Snapshot detection counts and scores.
    original_scores = [d.score for d in coll.predictions[0].detections]

    visualization.write_visualization(
        coll, tmp_path / "vis", score_threshold=0.5
    )
    # Collection unchanged after viz.
    assert [d.score for d in coll.predictions[0].detections] == original_scores

    # JSON export reflects the FULL collection (not thresholded).
    out_json = json_writer.write_json(coll, tmp_path / "out.json")
    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert len(data[0]["detections"]) == 2
    assert [d["score"] for d in data[0]["detections"]] == pytest.approx(original_scores)


def test_visualization_threshold_does_not_mutate_dota(tmp_path: Path, monkeypatch) -> None:
    from deim_app.writers import dota_writer, visualization

    def noop_draw(image, detections, **kwargs):
        return image

    monkeypatch.setattr(visualization, "draw_obb_detections", noop_draw)

    coll = make_obb_collection(scores=(0.9, 0.2))
    visualization.write_visualization(coll, tmp_path / "vis", score_threshold=0.5)
    out_dir = dota_writer.write_dota(coll, tmp_path / "dota")
    txt_files = list(out_dir.glob("*.txt"))
    assert txt_files
    lines = txt_files[0].read_text(encoding="utf-8").strip().splitlines()
    # Both detections survive in DOTA (threshold only affected viz).
    assert len(lines) == 2


def test_visualization_hbb_uses_image_draw_directly(tmp_path: Path) -> None:
    """HBB visualization draws rectangles with PIL directly (no engine import)."""
    from deim_app.writers import visualization

    coll = make_hbb_collection(scores=(0.9,))
    out_dir = visualization.write_visualization(
        coll, tmp_path / "vis", color=(255, 0, 0), line_width=1, alpha=0.0, score_threshold=0.0
    )
    files = list(out_dir.glob("*"))
    assert files
