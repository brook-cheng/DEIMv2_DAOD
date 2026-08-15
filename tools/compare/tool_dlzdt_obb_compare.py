#!/usr/bin/env python3
"""
dlzdt OBB 模型对比工具 — DEIMv2-OBB vs YOLO-OBB
==================================================

Overview
--------
Multi-model comparison pipeline for OBB detection on the dlzdt dataset:

1. Converts YOLO-OBB GT labels (xywhr normalized) → DOTA 8-coord format
2. Converts YOLO-OBB predictions (JSON) → DOTA format
3. Runs metric comparison (mAP, IoU distribution, PR curves)
4. Generates side-by-side visualization overlays

Designed for comparing DEIMv2-OBB variants against YOLO-OBB baselines.

Entry Points
------------
    python tools/compare/tool_dlzdt_obb_compare.py

Configuration (edit in-file)
-----------------------------
BASE_DATA       : str   — root dir with images/val/ and labels/val/
CLASSES_TXT     : str   — path to classes.txt
YOLO_PRED_JSON  : str   — YOLO predictions.json path
OUTPUT_ROOT     : str   — output directory
DET_DIRS        : list  — model prediction directories (YOLO + DEIMv2 variants)
MODEL_NAMES     : list  — display names matching DET_DIRS order
IOUV            : ndarray — IoU thresholds for evaluation (None = default [0.5:0.95:0.05])
VIS_IMAGE_NUM   : int   — max images for visualization output

Output Structure
----------------
OUTPUT_ROOT/
├── gt_dota/                       # converted GT in DOTA format
├── yolo_dota/                     # converted YOLO predictions in DOTA format
├── comparison_images_train/       # side-by-side visualization PNGs
└── (metrics printed to stdout)

Usage
-----
    1. Set paths at the top of the script
    2. python tools/compare/tool_dlzdt_obb_compare.py
"""

import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
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
GT_LABLE_DIR = os.path.join(BASE_DATA, "labels", "val")  # YOLO-OBB format labels
CLASSES_TXT = os.path.join(BASE_DATA, "classes.txt")

YOLO_PRED_JSON = "/mnt/d/cx/thired/ultralytics_update/runs/dlazdt_obb_val/yolo_train_1280_2026_5_31/val2/predictions.json"

OUTPUT_ROOT = "./test/data/outputs/dlzdt_obb_compare_val"
OUTPUT_GT_DOTA_DIR = os.path.join(OUTPUT_ROOT, "gt_dota")
OUTPUT_YOLO_DOTA_DIR = os.path.join(OUTPUT_ROOT, "yolo_dota")
OUTPUT_VISUAL_DIR = os.path.join(OUTPUT_ROOT, "comparison_images_train")

DET_DIRS = [
    OUTPUT_YOLO_DOTA_DIR,
    "./test/data/outputs/dlzdt_res/dlzdt_ablation/abl_rep0",
    "./test/data/outputs/dlzdt_res/dlzdt_ablation/abl_rep0_shifted",
    "./test/data/outputs/dlzdt_res/dlzdt_ablation/abl_rep0_mangle",
    "./test/data/outputs/dlzdt_res/dlzdt_ablation/abl_rep0_offset_post",
    "./test/data/outputs/dlzdt_res/dlzdt_ablation/abl_rep1",
    "./test/data/outputs/dlzdt_res/dlzdt_ablation/abl_rep3",
    "./test/data/outputs/dlzdt_res/dlzdt_ablation/abl_rep3_afp",
    "./test/data/outputs/dlzdt_res/dlzdt_ablation/abl_rep3_fused",
]
MODEL_NAMES = [
    "YOLO-OBB",
    "abl_rep0",
    "abl_rep0_shifted",
    "abl_rep0_mangle",
    "abl_rep0_offset_post",
    "abl_rep1",
    "abl_rep3",
    "abl_rep3_afp",
    "abl_rep3_fused",
]
# DET_DIRS = [
#     "./test/data/outputs/dlzdt_sp_rep0_train",
#     "./test/data/outputs/dlzdt_sp_rep1_train",
# ]
# MODEL_NAMES = ["DEIMv2-OBB-SP-Rep0", "DEIMv2-OBB-SP-Rep1"]
VIS_IMAGE_NUM = 250

IOUV = None
# IOUV = None


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
    if os.path.exists(OUTPUT_GT_DOTA_DIR):
        shutil.rmtree(OUTPUT_GT_DOTA_DIR)
    yolo_gt_to_dota(GT_LABLE_DIR, IMG_DIR, OUTPUT_GT_DOTA_DIR, class_names)

    # ── Step 2: YOLO predictions → DOTA format ──
    print("\n" + "=" * 60)
    print("Step 2: Converting YOLO predictions → DOTA format")
    print("=" * 60)
    if os.path.exists(OUTPUT_YOLO_DOTA_DIR):
        shutil.rmtree(OUTPUT_YOLO_DOTA_DIR)
    category_map = {
        i + 1: name for i, name in enumerate(class_names)
    }  # YOLO uses 1-indexed
    written = ultralytics_obb_json_to_dota(
        YOLO_PRED_JSON, OUTPUT_YOLO_DOTA_DIR, category_map, score_threshold=0.01
    )
    print(f"  YOLO predictions: {len(written)} images → {OUTPUT_YOLO_DOTA_DIR}")

    # ── Step 3: Metrics comparison ──
    print("\n" + "=" * 60)
    print("Step 3: Metrics comparison")
    print("=" * 60)
    compare_obb_models(
        gt_dir=OUTPUT_GT_DOTA_DIR,
        det_dirs=DET_DIRS,
        classes_file=CLASSES_TXT,
        model_names=MODEL_NAMES,
        iouv=IOUV,
    )

    # ── Step 4: Visualization ──
    print("\n" + "=" * 60)
    print("Step 4: Visualization comparison")
    print("=" * 60)
    if os.path.exists(OUTPUT_VISUAL_DIR):
        shutil.rmtree(OUTPUT_VISUAL_DIR)
    draw_obb_compare(
        img_dir=IMG_DIR,
        gt_dir=OUTPUT_GT_DOTA_DIR,
        det_dirs=DET_DIRS,
        output_dir=OUTPUT_VISUAL_DIR,
        model_names=MODEL_NAMES,
        score_threshold=0.25,
        max_images=VIS_IMAGE_NUM,
    )

    print("\n" + "=" * 60)
    print("Done. All outputs in:", OUTPUT_ROOT)
    print("=" * 60)


if __name__ == "__main__":
    main()
