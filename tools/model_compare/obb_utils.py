"""
OBB model comparison utilities — format converters and shared helpers.
"""

import os
import json
import numpy as np
from typing import Dict, List, Optional, Union
from PIL import Image, ImageDraw


# ---------------------------------------------------------------------------
#  Format converters
# ---------------------------------------------------------------------------

def _rbox_to_poly(rbox: list) -> list:
    """Convert YOLO-OBB rbox [cx, cy, w, h, theta_rad] to 8-coord polygon."""
    cx, cy, w, h, t = rbox
    cos_t, sin_t = np.cos(t), np.sin(t)
    dx = np.array([-w / 2, w / 2, w / 2, -w / 2])
    dy = np.array([-h / 2, -h / 2, h / 2, h / 2])
    x = cx + dx * cos_t - dy * sin_t
    y = cy + dx * sin_t + dy * cos_t
    poly = []
    for i in range(4):
        poly.extend([float(x[i]), float(y[i])])
    return poly


def ultralytics_obb_json_to_dota(
    json_path: str,
    output_dir: str,
    category_map: Dict[int, str],
    score_threshold: float = 0.0,
) -> List[str]:
    """Convert Ultralytics OBB val JSON to per-image DOTA txt files.

    Args:
        json_path:  path to ultralytics val results JSON.
        output_dir: directory to write per-image .txt files.
        category_map: {category_id: class_name} mapping.
        score_threshold: minimum confidence to keep (default 0.0 = keep all).

    Returns:
        list of image names (basenames) that had predictions written.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    os.makedirs(output_dir, exist_ok=True)

    # group by image_id
    by_image: Dict[str, list] = {}
    for item in data:
        score = item.get("score", 0.0)
        if score < score_threshold:
            continue
        cls_id = item.get("category_id")
        cls_name = category_map.get(cls_id, f"class_{cls_id}")
        poly = item.get("poly")
        if poly is None and item.get("rbox"):
            poly = _rbox_to_poly(item["rbox"])
        if poly is None:
            continue
        img_id = item["image_id"]
        by_image.setdefault(img_id, []).append(
            " ".join([f"{x:.6f}" for x in poly]) + f" {cls_name} {score:.6f}"
        )

    written = []
    for img_id, lines in by_image.items():
        safe_name = img_id.replace("/", "_").replace("\\", "_")
        out_path = os.path.join(output_dir, f"{safe_name}.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        written.append(safe_name)
    return sorted(written)


def merge_dota_files_to_per_image(
    input_paths: List[str],
    output_dir: str,
    image_name: str = None,
) -> List[str]:
    """Merge one or more single-file DOTA-format predictions into per-image txt files.

    Use when each model produces a single merged file (like DEIMv2-OBB demo format).
    If image_name is None, all lines go into a single file named "merged".

    Args:
        input_paths: list of DOTA-format txt files to merge.
        output_dir:  directory to write per-image .txt files.
        image_name:  image basename for the output file (default "merged").

    Returns:
        list of output file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    name = image_name or "merged"
    all_lines = []
    for path in input_paths:
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    all_lines.append(line)
    out_path = os.path.join(output_dir, f"{name}.txt")
    with open(out_path, "w") as f:
        f.write("\n".join(all_lines) + "\n")
    return [out_path]


def deimv2_obb_outputs_to_dota(
    outputs_dict: dict,
    output_dir: str,
    labels_map: dict,
    score_threshold: float = 0.0,
):
    """Convert DEIMv2-OBB inference outputs to per-image DOTA txt files.

    Args:
        outputs_dict: per-image predictions in the format::

            {
                "image_name.png": {
                    "labels": [0, 1, ...],
                    "boxes": [[cx, cy, w, h, theta], ...],   # xywhr format
                    "scores": [0.9, 0.8, ...],
                },
                ...
            }
        output_dir:    directory to write per-image .txt files.
        labels_map:    {label_id: class_name} mapping.
        score_threshold: min confidence to keep.
    """
    from engine.deim.obb_geometry import xywhr_to_xyxyxyxy
    import torch

    os.makedirs(output_dir, exist_ok=True)

    for img_name, pred in outputs_dict.items():
        labels = pred.get("labels", [])
        boxes = pred.get("boxes", [])
        scores = pred.get("scores", [])

        lines = []
        for box, score, label_id in zip(boxes, scores, labels):
            if score < score_threshold:
                continue
            cls_name = labels_map.get(int(label_id), f"class_{label_id}")
            # xywhr → 8-coord polygon
            t = torch.tensor(box, dtype=torch.float32).reshape(1, 5)
            poly = xywhr_to_xyxyxyxy(t).numpy().flatten()
            lines.append(
                " ".join([f"{x:.6f}" for x in poly])
                + f" {cls_name} {score:.6f}"
            )

        safe_name = img_name.replace("/", "_").replace("\\", "_")
        out_path = os.path.join(output_dir, f"{os.path.splitext(safe_name)[0]}.txt")
        with open(out_path, "w") as f:
            f.write("\n".join(lines) + "\n")

    print(f"Saved {len(outputs_dict)} images to {output_dir}")


# ---------------------------------------------------------------------------
#  OBB drawing
# ---------------------------------------------------------------------------

def draw_obb_polygons(
    image: Image.Image,
    annotations: List[dict],
    color: tuple,
    line_width: int = 2,
    alpha: float = 0.3,
) -> Image.Image:
    """Draw oriented bounding boxes on a PIL image.

    Args:
        image:       PIL image (RGB).
        annotations: list of dicts with keys 'poly' (8 floats) and optionally
                     'label' (str) and 'score' (float).
        color:       (R, G, B) tuple for outline color.
        line_width:  outline thickness.
        alpha:       fill alpha (0-1).

    Returns:
        PIL image with OBB polygons drawn.
    """
    draw = ImageDraw.Draw(image, "RGBA")
    fill_color = (*color, int(255 * alpha))

    for ann in annotations:
        poly = ann["poly"]
        pts = [(poly[i], poly[i + 1]) for i in range(0, len(poly), 2)]
        draw.polygon(pts, outline=color, fill=fill_color, width=line_width)

        label = ann.get("label", "")
        score = ann.get("score")
        if score is not None:
            label = f"{label} {score:.2f}"
        if label and len(pts) > 0:
            draw.text((pts[0][0], pts[0][1] - 12), label, fill=color)

    return image


def parse_dota_line(line: str) -> Optional[dict]:
    """Parse a single DOTA-format line into a dict with 'poly' and 'score'.

    Format: x1 y1 x2 y2 x3 y3 x4 y4 class_name score [difficulty]

    Returns None if parse fails.
    """
    parts = line.strip().split()
    if len(parts) < 10:
        return None
    try:
        poly = [float(x) for x in parts[:8]]
        return {
            "poly": poly,
            "label": parts[8],
            "score": float(parts[9]),
        }
    except (ValueError, IndexError):
        return None
