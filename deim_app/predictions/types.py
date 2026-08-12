"""Immutable prediction types for the DEIMv2 application layer.

All public surfaces here are frozen dataclasses. ``HBBDetection`` and
``OBBDetection`` carry box geometry as plain Python tuples (no torch tensors on
the public API), keeping these objects trivially serialisable and side-effect
free. ``Detection`` is a Union alias, not a base class — callers must
discriminate via ``box_mode`` or ``isinstance``.

``ImagePrediction`` holds a reference to a PIL image so it can be redrawn by
the visualization writer. It is ``frozen=True`` WITHOUT ``slots=True``: the
underlying ``PIL.Image.Image`` carries dynamic state and is not slottable in a
way that interacts cleanly with frozen slot dataclasses; we trade the minor
memory overhead for stable, predictable immutability semantics on the public
fields. JSON and DOTA writers omit the image entirely (only the metadata fields
are serialised).

Note on storage: ``xyxy`` is ``(x1, y1, x2, y2)`` axis-aligned; ``xywhr`` is
``(cx, cy, w, h, theta)`` with ``theta in [0, pi)`` — the same convention as
``engine.deim.obb_geometry.xywhr_to_xyxyxyxy``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

from PIL import Image


# ---------------------------------------------------------------------------
# Timings
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Timings:
    """Per-image inference timings in seconds. All entries must be >= 0."""

    preprocess_s: float
    inference_s: float
    postprocess_s: float

    def __post_init__(self) -> None:
        for name in ("preprocess_s", "inference_s", "postprocess_s"):
            value = getattr(self, name)
            if value < 0:
                raise ValueError(f"{name} must be non-negative, got {value!r}")


# ---------------------------------------------------------------------------
# Detections
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class HBBDetection:
    """Axis-aligned detection: ``xyxy = (x1, y1, x2, y2)``."""

    class_id: int
    class_name: str
    score: float
    xyxy: tuple[float, float, float, float]

    @property
    def box_mode(self) -> str:
        return "hbb"


@dataclass(frozen=True, slots=True)
class OBBDetection:
    """Oriented detection: ``xywhr = (cx, cy, w, h, theta)``, theta in [0, pi)."""

    class_id: int
    class_name: str
    score: float
    xywhr: tuple[float, float, float, float, float]

    @property
    def box_mode(self) -> str:
        return "obb"


#: Union alias for either detection kind. Callers discriminate via ``box_mode``
#: or ``isinstance``; there is no shared base class to subclass.
Detection = Union[HBBDetection, OBBDetection]


# ---------------------------------------------------------------------------
# Per-image prediction
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ImagePrediction:
    """All detections for a single image plus metadata for downstream export.

    ``original_image`` is held by reference for the visualization writer; the
    JSON and DOTA writers MUST NOT serialise pixel data — they emit only the
    metadata fields (``image_id``, ``source``, ``original_size``, detections,
    timings). The stored image should be treated as read-only: writers copy it
    before any in-place draw operation.

    ``frozen=True`` is set without ``slots=True`` because ``PIL.Image.Image``
    does not play well with slotted frozen dataclasses (its instances carry
    dynamic state that breaks slot binding). The public fields remain
    immutable from the caller's perspective.
    """

    image_id: str
    source: str
    original_image: Image.Image
    original_size: tuple[int, int]
    detections: tuple[Detection, ...]
    timings: Timings
