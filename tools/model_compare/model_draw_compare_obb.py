"""
Draw OBB model comparison — GT and multiple model predictions overlaid on images.

Usage (script):
    python tools/model_compare/model_draw_compare_obb.py

Usage (API):
    from tools.model_compare.model_draw_compare_obb import draw_obb_compare
    draw_obb_compare(img_dir, gt_dir, det_dirs, output_dir, model_names=...)
"""

import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import Dict, List, Optional

from tools.model_compare.obb_utils import draw_obb_polygons, parse_dota_line


def _load_image_annotations(
    gt_dir: str, det_dirs: List[str], img_name: str
) -> tuple:
    """Load GT and all model predictions for a single image.

    Returns:
        (gt_anns, model_anns_list) where model_anns_list is list[list[dict]].
    """
    gt_anns = []
    gt_path = os.path.join(gt_dir, f"{img_name}.txt")
    if os.path.exists(gt_path):
        with open(gt_path, "r") as f:
            for line in f:
                ann = parse_dota_line(line)
                if ann:
                    gt_anns.append(ann)

    model_anns_list = []
    for det_dir in det_dirs:
        anns = []
        det_path = os.path.join(det_dir, f"{img_name}.txt")
        if os.path.exists(det_path):
            with open(det_path, "r") as f:
                for line in f:
                    ann = parse_dota_line(line)
                    if ann:
                        anns.append(ann)
        model_anns_list.append(anns)

    return gt_anns, model_anns_list


def _discover_images(gt_dir, det_dirs, image_list):
    """Discover image names from directories or given list."""
    if image_list:
        return image_list
    names = set()
    for d in [gt_dir] + det_dirs:
        if os.path.isdir(d):
            for f in os.listdir(d):
                if f.endswith(".txt"):
                    names.add(os.path.splitext(f)[0])
    return sorted(names)


def draw_obb_compare(
    img_dir: str,
    gt_dir: str,
    det_dirs: List[str],
    output_dir: str,
    image_list: Optional[List[str]] = None,
    model_names: Optional[List[str]] = None,
    score_threshold: float = 0.3,
    max_images: Optional[int] = None,
):
    """Draw OBB comparison images with color-coded GT and model predictions.

    Args:
        img_dir:      directory of source images.
        gt_dir:       directory of per-image ground truth .txt files.
        det_dirs:     list of per-image prediction directories (one per model).
        output_dir:   directory to save comparison images.
        image_list:   optional list of image basenames (auto-discover if None).
        model_names:  list of model name strings (auto-name if None).
        score_threshold: min confidence to draw.
        max_images:   limit number of images processed (None = all).
    """
    os.makedirs(output_dir, exist_ok=True)
    num_models = len(det_dirs)

    if model_names is None:
        model_names = [f"Model_{i+1}" for i in range(num_models)]
    assert len(model_names) == num_models

    # color palette: GT = green, models = rainbow
    colors = plt.cm.tab10(np.linspace(0, 1, max(num_models + 1, 10)))
    gt_color = (0, 200, 0)  # green for GT
    model_colors = [
        tuple(int(c * 255) for c in colors[i + 1][:3]) for i in range(num_models)
    ]

    # discover images
    img_names = _discover_images(gt_dir, det_dirs, image_list)
    if max_images:
        img_names = img_names[:max_images]

    print(f"Processing {len(img_names)} images, {num_models} model(s)")

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24
        )
    except Exception:
        font = ImageFont.load_default()

    for idx, img_name in enumerate(img_names):
        # find image file
        img_path = None
        for ext in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"):
            candidate = os.path.join(img_dir, f"{img_name}{ext}")
            if os.path.exists(candidate):
                img_path = candidate
                break
        if img_path is None:
            candidate = os.path.join(img_dir, img_name)
            if os.path.exists(candidate):
                img_path = candidate

        if img_path is None:
            print(f"  [{idx+1}/{len(img_names)}] WARN: image not found for {img_name}")
            continue

        image = Image.open(img_path).convert("RGBA")
        gt_anns, model_anns_list = _load_image_annotations(gt_dir, det_dirs, img_name)

        # filter by score
        model_anns_list = [
            [a for a in anns if a.get("score", 1.0) >= score_threshold]
            for anns in model_anns_list
        ]

        # draw GT
        if gt_anns:
            image = draw_obb_polygons(image, gt_anns, gt_color, line_width=3, alpha=0.15)

        # draw models
        for anns, color in zip(model_anns_list, model_colors):
            if anns:
                image = draw_obb_polygons(image, anns, color, line_width=2, alpha=0.1)

        # legend
        draw = ImageDraw.Draw(image)
        x, y = 10, 10
        box_h = 28
        gap = box_h + 4

        legend_items = [("GT", gt_color)] + list(zip(model_names, model_colors))
        for name, color in legend_items:
            draw.rectangle([x, y, x + 40, y + box_h], fill=color, outline=(255, 255, 255))
            draw.text((x + 46, y + 2), name, fill=(255, 255, 255), font=font)
            y += gap

        # convert to RGB for save
        bg = Image.new("RGB", image.size, (255, 255, 255))
        bg.paste(image, mask=image.split()[3])
        out_path = os.path.join(output_dir, f"{img_name}_compare.jpg")
        bg.save(out_path)
        print(f"  [{idx+1}/{len(img_names)}] {img_name} -> {out_path}")

    print(f"\nDone. {len(img_names)} images saved to {output_dir}")


if __name__ == "__main__":
    # Example usage with demo data
    demo_dir = os.path.join(os.path.dirname(__file__), "annotation_demo")
    gt_dir = os.path.join(demo_dir, "gt")
    os.makedirs(gt_dir, exist_ok=True)

    # Write demo GT file
    with open(os.path.join(gt_dir, "demo.txt"), "w") as f:
        f.write(
            "503.998464 416.002048 497.999872 294.001664 "
            "622.001152 298.000384 626.00192 421.997568 plane 0\n"
        )
        f.write(
            "573.99808 288.001024 442.000384 252.002304 "
            "465.999872 146.00192 595.99872 177.997824 plane 0\n"
        )

    # Use demo DOTA file directly as det
    det_dir = os.path.join(demo_dir, "det")
    os.makedirs(det_dir, exist_ok=True)
    import shutil
    shutil.copy(
        os.path.join(demo_dir, "dota_annotation_obb_demo.txt"),
        os.path.join(det_dir, "demo.txt"),
    )

    draw_obb_compare(
        img_dir=demo_dir,   # no real images, will warn
        gt_dir=gt_dir,
        det_dirs=[det_dir],
        output_dir=os.path.join(demo_dir, "output_draw"),
        model_names=["DEIMv2-OBB"],
    )
