"""Adapter-owned geometry wrappers for the DEIMv2 application layer.

This is the ONLY module under ``deim_app/`` permitted to import
``engine.deim.obb_geometry`` or ``tools.model_compare.obb_utils``. The
boundary is enforced by ``test/deim_app/test_dependency_boundaries.py``.

All non-adapter code (writers, predictions, future inference code) MUST route
geometry through these wrappers — never import the engine or ``tools.model_compare``
modules directly.

Why an adapter layer:
- Keeps the engine side-effect surface small (one well-audited import point).
- Lets writers / tests stay pure-Python (no torch dependency at import time on
  the writer side; torch loads lazily inside ``obb_to_polygon``).
- Provides a single seam to mock in writer tests.
"""

from __future__ import annotations

from typing import Sequence

from PIL import Image

# Engine / tools imports — ONLY allowed here. These names resolve through the
# project's existing sys.path setup (engine and tools live at the repo root).
from engine.deim.obb_geometry import xywhr_to_xyxyxyxy
from tools.model_compare.obb_utils import (
    _load_vis_font,
    draw_obb_polygons,
)


def _to_xywhr_tensor(xywhr: Sequence[float] | object) -> "object":
    """Coerce a single xywhr input to a ``(1, 5)`` float32 torch tensor."""
    import torch

    # Already a tensor?
    if hasattr(xywhr, "shape") and hasattr(xywhr, "dim"):
        t = xywhr
        if t.dim() == 1:
            if t.shape[0] != 5:
                raise ValueError(f"xywhr tensor must have 5 elements, got shape {tuple(t.shape)}")
            return t.reshape(1, 5).to(torch.float32)
        if t.dim() == 2 and t.shape[-1] == 5:
            if t.shape[0] != 1:
                raise ValueError(
                    f"obb_to_polygon expects a single OBB; got shape {tuple(t.shape)}"
                )
            return t.to(torch.float32)
        if t.dim() == 2 and t.shape[0] == 5 and t.shape[1] == 1:
            return t.reshape(1, 5).to(torch.float32)
        raise ValueError(f"xywhr tensor has unexpected shape {tuple(t.shape)}")
    # Treat as a sequence.
    seq = list(xywhr)
    if len(seq) != 5:
        raise ValueError(f"xywhr sequence must have 5 elements, got {len(seq)}")
    return torch.tensor(seq, dtype=torch.float32).reshape(1, 5)


def obb_to_polygon(
    xywhr: Sequence[float] | object,
) -> tuple[float, float, float, float, float, float, float, float]:
    """Convert a single OBB ``(cx, cy, w, h, theta)`` to 8 polygon floats.

    Thin wrapper over ``engine.deim.obb_geometry.xywhr_to_xyxyxyxy`` — never
    reimplements the trigonometry, so behavioural parity with the engine is
    guaranteed by construction.

    Accepts:
      * a Python sequence of length 5, or
      * a torch tensor of shape ``(5,)`` / ``(1, 5)``.

    Always returns a Python tuple of 8 floats ``(x1, y1, x2, y2, x3, y3, x4, y4)``.
    """
    t = _to_xywhr_tensor(xywhr)
    corners = xywhr_to_xyxyxyxy(t)  # shape (1, 4, 2)
    flat = corners.reshape(-1).tolist()
    return tuple(float(v) for v in flat)  # type: ignore[return-value]


def draw_obb_detections(
    image: Image.Image,
    detections: Sequence["object"],
    *,
    color: tuple = (255, 0, 0),
    line_width: int = 2,
    alpha: float = 0.3,
    score_threshold: float = 0.0,
    font=None,
) -> Image.Image:
    """Draw OBB detections onto a copy of ``image`` and return the copy.

    Builds the ``annotations`` list of ``{'poly': [...8 floats], 'label': str,
    'score': float}`` dicts that ``draw_obb_polygons`` expects, then delegates.
    The input image is COPIED first because ``ImageDraw`` mutates in place —
    callers may pass a shared source image safely.

    Detections whose ``score < score_threshold`` are skipped (this is the
    visualisation-time decision and does NOT mutate the caller's collection).
    """
    annotations = []
    for det in detections:
        if det.score < score_threshold:
            continue
        poly = list(obb_to_polygon(det.xywhr))
        annotations.append({"poly": poly, "label": det.class_name, "score": float(det.score)})

    canvas = image.copy()
    return draw_obb_polygons(
        canvas,
        annotations,
        color=color,
        line_width=line_width,
        alpha=alpha,
        font=font,
    )


def load_visualization_font(size: int = 24):
    """Cross-platform TTF font loader with PIL fallback (wraps ``_load_vis_font``)."""
    return _load_vis_font(size=size)
