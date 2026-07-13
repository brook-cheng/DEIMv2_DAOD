"""
dlzdt OBB model comparison — DEIMv2-OBB vs YOLO-OBB on pj_dlzdt dataset.

Converts YOLO GT labels (YOLO-OBB xywhr format → DOTA 8-coord),
converts YOLO predictions (JSON → DOTA per-image),
then runs metrics + visualization comparison.
"""

import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import json
import shutil
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from engine.deim.obb_geometry import xywhr_to_xyxyxyxy
from tools.model_compare.obb_utils import (
    ultralytics_obb_json_to_dota,
    parse_dota_line,
)
from tools.model_compare.model_metrics_compare_obb import compare_obb_models
from tools.model_compare.model_draw_compare_obb import draw_obb_compare

# ── Paths ──
BASE_DATA = "/mnt/d/project_data/model_test/deimv2_obb_train_data/dlzdt_obb_val"
IMG_DIR = os.path.join(BASE_DATA, "images", "val")
CLASSES_TXT = os.path.join(BASE_DATA, "classes.txt")

GT_YOLO_DIR = os.path.join(BASE_DATA, "labels", "val")  # YOLO-OBB format labels
YOLO_PRED_JSON = "/mnt/d/cx/thired/ultralytics_update/runs/dlazdt_obb_val/yolo_train_1280_2026_5_31/val2/predictions.json"

OUTPUT_ROOT = "./test/data/outputs/dlzdt_obb_compare"
GT_DOTA_DIR = os.path.join(OUTPUT_ROOT, "gt_dota")
YOLO_DOTA_DIR = os.path.join(OUTPUT_ROOT, "yolo_dota")
VISUAL_DIR = os.path.join(OUTPUT_ROOT, "comparison_images")

DET_DIRS = [
    YOLO_DOTA_DIR,
    "./test/data/outputs/dlzdt_sp_rep0",
    "./test/data/outputs/dlzdt_sp_rep1",
]
MODEL_NAMES = ["YOLO-OBB", "DEIMv2-OBB-SP-Rep0", "DEIMv2-OBB-SP-Rep1"]
VIS_IMAGE_NUM = 259


def yolo_gt_to_dota(
    gt_yolo_dir: str, img_dir: str, output_dota_dir: str, class_names: list
):
    """Convert YOLO-OBB GT labels (xywhr normalized) to DOTA 8-coord per-image txt.

    YOLO label format: class_id cx cy w h angle  (all normalized to [0,1], angle in radians)
    DOTA output:       x1 y1 x2 y2 x3 y3 x4 y4 class_name 0
    """
    os.makedirs(output_dota_dir, exist_ok=True)

    # build image name → dimensions
    img_dims = {}
    for fname in os.listdir(img_dir):
        if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            base = os.path.splitext(fname)[0]
            try:
                img = Image.open(os.path.join(img_dir, fname))
                img_dims[base] = (img.width, img.height)
            except Exception:
                img_dims[base] = (640, 640)

    converted = 0
    for fname in os.listdir(gt_yolo_dir):
        if not fname.endswith(".txt"):
            continue
        base = os.path.splitext(fname)[0]
        if base not in img_dims:
            continue
        img_w, img_h = img_dims[base]

        lines = []
        with open(os.path.join(gt_yolo_dir, fname), "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 6:
                    continue
                cls_id = int(parts[0])
                # cx = float(parts[1]) * img_w
                # cy = float(parts[2]) * img_h
                # w = float(parts[3]) * img_w
                # h = float(parts[4]) * img_h
                # theta = float(parts[5])  # radians, already correct

                # # xywhr → 8-coord
                # t = torch.tensor([[cx, cy, w, h, theta]], dtype=torch.float32)
                # poly = xywhr_to_xyxyxyxy(t).numpy().flatten()
                poly = np.array(
                    [
                        float(parts[1]) * img_w,
                        float(parts[2]) * img_h,
                        float(parts[3]) * img_w,
                        float(parts[4]) * img_h,
                        float(parts[5]) * img_w,
                        float(parts[6]) * img_h,
                        float(parts[7]) * img_w,
                        float(parts[8]) * img_h,
                    ]
                )

                cls_name = (
                    class_names[cls_id]
                    if cls_id < len(class_names)
                    else f"class_{cls_id}"
                )
                lines.append(" ".join([f"{x:.6f}" for x in poly]) + f" {cls_name} 0")

        if lines:
            with open(os.path.join(output_dota_dir, f"{base}.txt"), "w") as f:
                f.write("\n".join(lines) + "\n")
            converted += 1

    print(f"  GT converted: {converted} images → {output_dota_dir}")


def main():
    # ── load class names ──
    with open(CLASSES_TXT, "r") as f:
        class_names = [line.strip() for line in f if line.strip()]
    print(f"Classes: {class_names}")

    # ── Step 1: YOLO GT → DOTA format ──
    print("\n" + "=" * 60)
    print("Step 1: Converting YOLO GT labels → DOTA format")
    print("=" * 60)
    if os.path.exists(GT_DOTA_DIR):
        shutil.rmtree(GT_DOTA_DIR)
    yolo_gt_to_dota(GT_YOLO_DIR, IMG_DIR, GT_DOTA_DIR, class_names)

    # ── Step 2: YOLO predictions → DOTA format ──
    print("\n" + "=" * 60)
    print("Step 2: Converting YOLO predictions → DOTA format")
    print("=" * 60)
    if os.path.exists(YOLO_DOTA_DIR):
        shutil.rmtree(YOLO_DOTA_DIR)
    category_map = {
        i + 1: name for i, name in enumerate(class_names)
    }  # YOLO uses 1-indexed
    written = ultralytics_obb_json_to_dota(
        YOLO_PRED_JSON, YOLO_DOTA_DIR, category_map, score_threshold=0.01
    )
    print(f"  YOLO predictions: {len(written)} images → {YOLO_DOTA_DIR}")

    # ── Step 3: Metrics comparison ──
    print("\n" + "=" * 60)
    print("Step 3: Metrics comparison")
    print("=" * 60)
    compare_obb_models(
        gt_dir=GT_DOTA_DIR,
        det_dirs=DET_DIRS,
        classes_file=CLASSES_TXT,
        model_names=MODEL_NAMES,
        iouv=np.array([0.3]),
    )

    # ── Step 4: Visualization ──
    print("\n" + "=" * 60)
    print("Step 4: Visualization comparison")
    print("=" * 60)
    if os.path.exists(VISUAL_DIR):
        shutil.rmtree(VISUAL_DIR)
    draw_obb_compare(
        img_dir=IMG_DIR,
        gt_dir=GT_DOTA_DIR,
        det_dirs=DET_DIRS,
        output_dir=VISUAL_DIR,
        model_names=MODEL_NAMES,
        score_threshold=0.25,
        max_images=VIS_IMAGE_NUM,
    )

    print("\n" + "=" * 60)
    print("Done. All outputs in:", OUTPUT_ROOT)
    print("=" * 60)


if __name__ == "__main__":
    main()
