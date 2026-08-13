"""Dataset metadata loaders for the application layer.

Reads COCO JSON and OBB ``classes.txt`` annotation metadata using only the
Python standard library (``json`` for COCO, plain text for OBB).  This module
MUST NOT import ``engine.*`` or ``pycocotools`` — the dependency boundary
(``test/deim_app/test_dependency_boundaries.py``) permits engine imports only
in ``deim_app/adapters/``.

The MS COCO 1..90 → 0..79 category mapping is inlined below from
``engine/data/dataset/coco_dataset.py`` (``mscoco_category2name``) to avoid
the engine import.  The source is documented in the constant docstring.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

from deim_app.errors import AppConfigError

__all__ = [
    "DatasetMetadata",
    "load_coco_metadata",
    "load_obb_metadata",
]


# ---------------------------------------------------------------------------
# Frozen result container
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DatasetMetadata:
    """Immutable dataset metadata derived from on-disk annotation files.

    Attributes:
        box_mode: ``"hbb"`` for horizontal bounding boxes (COCO), ``"obb"``
            for oriented bounding boxes (DOTA / YOLO-OBB).
        num_classes: Number of foreground classes the model must predict
            (excluding any background index).  Drives the top-level
            ``num_classes`` engine field.
        class_names_by_label: Contiguous zero-based model label → human-readable
            class name.  This is the mapping the model's classification head
            uses internally.
        output_names_by_id: Emitted category ID → class name.  After the
            postprocessor remaps model labels to dataset-native category IDs
            (only when ``remap_mscoco_category=True``), this mapping gives the
            name for each emitted ID.  For contiguous (non-remapped) datasets
            this is identical to ``class_names_by_label``.
    """

    box_mode: str
    num_classes: int
    class_names_by_label: Mapping[int, str]
    output_names_by_id: Mapping[int, str]


# ---------------------------------------------------------------------------
# OBB metadata (classes.txt)
# ---------------------------------------------------------------------------


def load_obb_metadata(classes_file: Path) -> DatasetMetadata:
    """Read an OBB ``classes.txt`` file and derive :class:`DatasetMetadata`.

    The file must contain one class name per non-empty line (whitespace
    stripped).  Line order defines the zero-based label assignment: the first
    non-empty line is label 0, the second is label 1, etc.

    Raises:
        AppConfigError: if ``classes_file`` does not exist.
    """
    classes_file = Path(classes_file)
    if not classes_file.exists():
        raise AppConfigError(
            f"data.classes_file '{classes_file}' does not exist; "
            f"cannot derive OBB metadata"
        )

    raw_lines = classes_file.read_text(encoding="utf-8").splitlines()
    names = [line.strip() for line in raw_lines if line.strip()]
    class_names_by_label: dict[int, str] = {i: name for i, name in enumerate(names)}

    return DatasetMetadata(
        box_mode="obb",
        num_classes=len(names),
        class_names_by_label=class_names_by_label,
        output_names_by_id=dict(class_names_by_label),
    )


# ---------------------------------------------------------------------------
# COCO metadata (instances JSON)
# ---------------------------------------------------------------------------

#: MS COCO 1..90 (with gaps) category-id → name mapping.
#:
#: Inlined verbatim from ``engine/data/dataset/coco_dataset.py:180-261``
#: (``mscoco_category2name``) to avoid importing ``engine.*`` from the
#: application layer (dependency-boundary rule).  The canonical source is
#: the COCO dataset's official ``instances_val2017.json`` ``categories`` list;
#: the engine copy is itself derived from torchvision's COCO utilities.
_MSCOCO_CATEGORY2NAME: dict[int, str] = {
    1: "person",
    2: "bicycle",
    3: "car",
    4: "motorcycle",
    5: "airplane",
    6: "bus",
    7: "train",
    8: "truck",
    9: "boat",
    10: "traffic light",
    11: "fire hydrant",
    13: "stop sign",
    14: "parking meter",
    15: "bench",
    16: "bird",
    17: "cat",
    18: "dog",
    19: "horse",
    20: "sheep",
    21: "cow",
    22: "elephant",
    23: "bear",
    24: "zebra",
    25: "giraffe",
    27: "backpack",
    28: "umbrella",
    31: "handbag",
    32: "tie",
    33: "suitcase",
    34: "frisbee",
    35: "skis",
    36: "snowboard",
    37: "sports ball",
    38: "kite",
    39: "baseball bat",
    40: "baseball glove",
    41: "skateboard",
    42: "surfboard",
    43: "tennis racket",
    44: "bottle",
    46: "wine glass",
    47: "cup",
    48: "fork",
    49: "knife",
    50: "spoon",
    51: "bowl",
    52: "banana",
    53: "apple",
    54: "sandwich",
    55: "orange",
    56: "broccoli",
    57: "carrot",
    58: "hot dog",
    59: "pizza",
    60: "donut",
    61: "cake",
    62: "chair",
    63: "couch",
    64: "potted plant",
    65: "bed",
    67: "dining table",
    70: "toilet",
    72: "tv",
    73: "laptop",
    74: "mouse",
    75: "remote",
    76: "keyboard",
    77: "cell phone",
    78: "microwave",
    79: "oven",
    80: "toaster",
    81: "sink",
    82: "refrigerator",
    84: "book",
    85: "clock",
    86: "vase",
    87: "scissors",
    88: "teddy bear",
    89: "hair drier",
    90: "toothbrush",
}

#: MS COCO contiguous-label mapping: ``{category_id: contiguous_label}``.
#: Derived identically to ``engine/data/dataset/coco_dataset.py:263``
#: (``mscoco_category2label = {k: i for i, k in enumerate(mscoco_category2name.keys())}``).
_MSCOCO_CATEGORY2LABEL: dict[int, int] = {
    cat_id: label for label, cat_id in enumerate(_MSCOCO_CATEGORY2NAME)
}

#: The 80 MS COCO category names as a set, for auto-detection.
#: Derived from ``_MSCOCO_CATEGORY2NAME.values()``.
_MSCOCO_CATEGORY_NAMES: frozenset[str] = frozenset(_MSCOCO_CATEGORY2NAME.values())

#: The standard MS COCO 1..90 category IDs (with gaps) as a set.
_MSCOCO_CATEGORY_IDS: frozenset[int] = frozenset(_MSCOCO_CATEGORY2NAME)


def load_coco_metadata(
    annotation_path: Path,
    remap_mscoco_category: bool | None = None,
) -> DatasetMetadata:
    """Read a COCO-format JSON and derive :class:`DatasetMetadata`.

    Args:
        annotation_path: Path to a COCO ``instances*.json`` file.
        remap_mscoco_category: Controls how category IDs are interpreted.

            * ``True`` (explicit): the JSON must contain exactly the standard
              MS COCO 80 category names and 1..90 IDs. If the names do not
              match, raises :class:`AppConfigError` suggesting
              ``remap_mscoco_category=False``.
            * ``False`` (explicit): category IDs must be contiguous zero-based
              (``0, 1, ..., N-1``); any gap raises :class:`AppConfigError`
              with ``"contiguous"`` in the message.
            * ``None`` (default — auto-detect): if the category names and IDs
              exactly match the MS COCO 80-class standard, remaps as ``True``.
              Otherwise requires contiguous zero-based IDs (non-contiguous
              raises ``AppConfigError``). This eliminates silent label
              corruption for custom datasets whose IDs overlap 1..90.

    Raises:
        AppConfigError: if the file does not exist, is not valid JSON,
            category IDs are non-contiguous (in contiguous mode), or
            category names don't match MS COCO (in explicit remap mode).
    """
    annotation_path = Path(annotation_path)
    if not annotation_path.exists():
        raise AppConfigError(
            f"data annotation file '{annotation_path}' does not exist; "
            f"cannot derive COCO metadata"
        )

    try:
        raw = json.loads(annotation_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AppConfigError(
            f"data annotation file '{annotation_path}' is not valid JSON: {exc}"
        ) from exc

    if not isinstance(raw, dict) or "categories" not in raw:
        raise AppConfigError(
            f"data annotation file '{annotation_path}' is missing a 'categories' list"
        )

    categories = raw["categories"]
    if not isinstance(categories, list):
        raise AppConfigError(
            f"data annotation file '{annotation_path}' 'categories' must be a list"
        )

    json_names = frozenset(str(cat.get("name", "")).strip() for cat in categories)
    json_ids = frozenset(cat["id"] for cat in categories)
    is_mscoco = (
        json_names == _MSCOCO_CATEGORY_NAMES
        and json_ids == _MSCOCO_CATEGORY_IDS
    )

    if remap_mscoco_category is True:
        if not is_mscoco:
            _raise_remap_name_mismatch(json_names, annotation_path)
        return _build_remapped_metadata()

    if remap_mscoco_category is False:
        return _build_contiguous_metadata(categories, annotation_path)

    # remap_mscoco_category is None — auto-detect
    if is_mscoco:
        return _build_remapped_metadata()
    return _build_contiguous_metadata(categories, annotation_path)


def _raise_remap_name_mismatch(
    json_names: frozenset[str], annotation_path: Path
) -> None:
    """Raise an actionable error for explicit ``True`` with non-MS-COCO names."""
    missing = _MSCOCO_CATEGORY_NAMES - json_names
    extra = json_names - _MSCOCO_CATEGORY_NAMES
    details: list[str] = []
    if missing:
        details.append(f"missing {len(missing)} MS COCO names (e.g. {sorted(missing)[:3]})")
    if extra:
        details.append(f"found {len(extra)} non-MS-COCO names (e.g. {sorted(extra)[:3]})")
    raise AppConfigError(
        f"remap_mscoco_category=True but categories in '{annotation_path}' "
        f"do not match the standard MS COCO 80 names ({'; '.join(details)}). "
        f"Set remap_mscoco_category=False (or omit it for auto-detection) "
        f"for custom datasets with contiguous zero-based IDs."
    )


class _CocoCategory(TypedDict):
    """COCO-format category entry as read from ``instances_*.json``."""

    id: int
    name: str


def _build_contiguous_metadata(
    categories: list[_CocoCategory], annotation_path: Path
) -> DatasetMetadata:
    sorted_cats = sorted(categories, key=lambda c: c["id"])
    ids = [c["id"] for c in sorted_cats]

    expected = list(range(len(ids)))
    if ids != expected:
        raise AppConfigError(
            f"data annotation file '{annotation_path}' has non-contiguous "
            f"category IDs {ids}; IDs must be contiguous zero-based (0..N-1). "
            f"If this is the standard MS COCO dataset, ensure all 80 category "
            f"names and 1..90 IDs are present for auto-detection."
        )

    class_names_by_label = {i: cat["name"] for i, cat in enumerate(sorted_cats)}
    return DatasetMetadata(
        box_mode="hbb",
        num_classes=len(sorted_cats),
        class_names_by_label=class_names_by_label,
        output_names_by_id=dict(class_names_by_label),
    )


def _build_remapped_metadata() -> DatasetMetadata:
    class_names_by_label: dict[int, str] = {
        _MSCOCO_CATEGORY2LABEL[cat_id]: _MSCOCO_CATEGORY2NAME[cat_id]
        for cat_id in _MSCOCO_CATEGORY2NAME
    }
    output_names_by_id = dict(_MSCOCO_CATEGORY2NAME)

    return DatasetMetadata(
        box_mode="hbb",
        num_classes=len(_MSCOCO_CATEGORY2NAME),
        class_names_by_label=class_names_by_label,
        output_names_by_id=output_names_by_id,
    )
