"""Tests for ``deim_app.predictions.collection``: immutable filtering and top-K.

Asserts:
- ``filter`` / ``top_k`` return NEW collections and never mutate the input.
- ``filter`` keeps empty per-image detections tuples (indices stay stable).
- ``top_k`` is per-image, score-descending, preserves image order.
- Mode-mismatched DOTA export (HBB collection) raises ``ExportError``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from deim_app.errors import ExportError
from deim_app.predictions.collection import PredictionCollection
from deim_app.predictions.types import HBBDetection, OBBDetection

from conftest import make_hbb_collection, make_obb_collection


# ---------------------------------------------------------------------------
# Immutability / non-mutation
# ---------------------------------------------------------------------------


def test_filter_returns_new_collection_without_mutating_full_predictions() -> None:
    full = make_obb_collection(scores=(0.9, 0.2))
    before = full.predictions[0].detections
    filtered = full.filter(score_threshold=0.5)
    # Original is untouched.
    assert len(full.predictions[0].detections) == 2
    assert len(filtered.predictions[0].detections) == 1
    # Identity differs (a NEW collection was returned).
    assert filtered is not full
    assert filtered.predictions is not full.predictions
    assert full.predictions[0].detections == before


def test_top_k_does_not_mutate_input() -> None:
    full = make_hbb_collection(scores=(0.3, 0.9, 0.7))
    limited = full.top_k(2)
    assert len(full.predictions[0].detections) == 3
    assert len(limited.predictions[0].detections) == 2
    assert limited is not full


def test_filtered_detections_tuple_is_a_new_object() -> None:
    full = make_obb_collection(scores=(0.9, 0.2))
    filtered = full.filter(score_threshold=0.5)
    # New tuple instance on the surviving image even though we kept a subset.
    assert filtered.predictions[0].detections is not full.predictions[0].detections
    # And the per-image prediction is a new frozen dataclass.
    assert filtered.predictions[0] is not full.predictions[0]


# ---------------------------------------------------------------------------
# Score / class-name / class-id filtering
# ---------------------------------------------------------------------------


def test_filter_by_score_threshold_keeps_geq() -> None:
    full = make_obb_collection(scores=(0.1, 0.5, 0.9))
    out = full.filter(score_threshold=0.5)
    kept = [d.score for d in out.predictions[0].detections]
    # 0.5 and 0.9 survive (>= threshold).
    assert kept == [0.5, 0.9]


def test_filter_by_class_names() -> None:
    full = make_hbb_collection(
        scores=(0.9, 0.9, 0.9),
        class_names=("cat", "dog", "bird"),
    )
    out = full.filter(class_names={"dog", "bird"})
    names = [d.class_name for d in out.predictions[0].detections]
    assert sorted(names) == ["bird", "dog"]


def test_filter_by_class_ids() -> None:
    full = make_hbb_collection(scores=(0.9, 0.9, 0.9))
    # class_id mirrors the score index.
    out = full.filter(class_ids={0, 2})
    ids = [d.class_id for d in out.predictions[0].detections]
    assert ids == [0, 2]


def test_filter_combines_score_and_class() -> None:
    full = make_obb_collection(
        scores=(0.1, 0.9, 0.9),
        class_names=("a", "b", "c"),
    )
    out = full.filter(score_threshold=0.5, class_names={"b"})
    kept = [(d.class_name, d.score) for d in out.predictions[0].detections]
    assert kept == [("b", 0.9)]


def test_filter_preserves_image_count_with_empty_survivors() -> None:
    full = make_obb_collection(scores=(0.1, 0.2))
    out = full.filter(score_threshold=0.9)
    assert len(out.predictions) == len(full.predictions)
    assert out.predictions[0].detections == ()


# ---------------------------------------------------------------------------
# top_k semantics
# ---------------------------------------------------------------------------


def test_top_k_is_per_image_and_score_ordered() -> None:
    full = make_hbb_collection(scores=(0.3, 0.9, 0.7))
    limited = full.top_k(2)
    assert [d.score for d in limited.predictions[0].detections] == [0.9, 0.7]


def test_top_k_preserves_image_order() -> None:
    full = make_hbb_collection(
        scores=(0.9, 0.1),
        image_ids=("img0", "img1"),
    )
    out = full.top_k(1)
    assert [p.image_id for p in out.predictions] == ["img0", "img1"]
    # Per-image top-1 keeps the high-score detection only.
    assert [d.score for d in out.predictions[0].detections] == [0.9]
    assert [d.score for d in out.predictions[1].detections] == [0.1]


def test_top_k_with_k_larger_than_detections_returns_all_sorted() -> None:
    full = make_hbb_collection(scores=(0.1, 0.5, 0.3))
    out = full.top_k(10)
    assert [d.score for d in out.predictions[0].detections] == [0.5, 0.3, 0.1]


def test_top_k_zero_returns_empty_per_image() -> None:
    full = make_obb_collection(scores=(0.9, 0.2))
    out = full.top_k(0)
    assert len(out.predictions) == 1
    assert out.predictions[0].detections == ()


# ---------------------------------------------------------------------------
# Type / immutability of detection dataclasses
# ---------------------------------------------------------------------------


def test_detections_are_frozen() -> None:
    det = HBBDetection(
        class_id=0, class_name="a", score=0.5, xyxy=(0.0, 0.0, 1.0, 1.0)
    )
    with pytest.raises(Exception):
        det.class_id = 1  # type: ignore[misc]


def test_collection_is_frozen() -> None:
    coll = make_obb_collection()
    with pytest.raises(Exception):
        coll.box_mode = "hbb"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# box_mode property
# ---------------------------------------------------------------------------


def test_box_mode_property_on_detections() -> None:
    h = HBBDetection(0, "a", 0.5, (0.0, 0.0, 1.0, 1.0))
    o = OBBDetection(0, "a", 0.5, (0.0, 0.0, 1.0, 1.0, 0.0))
    assert h.box_mode == "hbb"
    assert o.box_mode == "obb"


# ---------------------------------------------------------------------------
# Mode-mismatched export
# ---------------------------------------------------------------------------


def test_export_dota_on_hbb_raises_export_error(tmp_path: Path) -> None:
    coll = make_hbb_collection(scores=(0.9,))
    with pytest.raises(ExportError):
        coll.export_dota(tmp_path / "out")


def test_export_json_roundtrip_smoke(tmp_path: Path) -> None:
    coll = make_obb_collection(scores=(0.9, 0.2))
    out = coll.export_json(tmp_path / "out.json")
    assert out.exists()
    data = json.loads(out.read_text(encoding="utf-8"))
    assert isinstance(data, list)
    assert data[0]["image_id"] == "img0"
    assert data[0]["detections"][0]["box_mode"] == "obb"
