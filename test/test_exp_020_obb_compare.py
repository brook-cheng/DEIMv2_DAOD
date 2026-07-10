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


def main():
    CLASSES_TXT = "/mnt/d/project_data/model_test/deimv2_obb_train_data/synthetic_ellipse/classes.txt"
    GT_DOTA_DIR = "/mnt/d/project_data/model_test/deimv2_obb_train_data/synthetic_ellipse/density_020/val"
    IMG_DIR = "/mnt/d/project_data/model_test/deimv2_obb_train_data/synthetic_ellipse/density_020/val"
    COMPARE_LIST = [
        f"./test/data/outputs/exp_020_anrep{angle_rep}" for angle_rep in range(0, 4)
    ]
    model_names = [f"angle_rep{angle_rep}" for angle_rep in range(0, 4)]
    VISUAL_DIR = "./test/data/outputs/exp_020_anrep/visual"
    # ── load class names ──
    with open(CLASSES_TXT, "r") as f:
        class_names = [line.strip() for line in f if line.strip()]
    print(f"Classes: {class_names}")

    # ── Step 3: Metrics comparison ──
    print("\n" + "=" * 60)
    print("Step 3: Metrics comparison")
    print("=" * 60)
    compare_obb_models(
        gt_dir=GT_DOTA_DIR,
        det_dirs=COMPARE_LIST,
        classes_file=CLASSES_TXT,
        model_names=model_names,
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
        det_dirs=COMPARE_LIST,
        output_dir=VISUAL_DIR,
        model_names=model_names,
        score_threshold=0.1,
        max_images=20,
    )

    print("\n" + "=" * 60)
    print("Done. All outputs in:", VISUAL_DIR)
    print("=" * 60)


if __name__ == "__main__":
    main()
