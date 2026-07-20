#!/usr/bin/env python3
"""
DEIMv2-OBB 模型推理与 DOTA 导出工具
=====================================

Overview
--------
加载 DEIMv2-OBB 模型 checkpoint，对图像目录批量推理，将 OBB 检测结果保存为
per-image DOTA 格式（8 坐标 + 类别名 + 置信度），供后续模型比对和可视化使用。

Each image gets a .txt file with one detection per line::

    x1 y1 x2 y2 x3 y3 x4 y4 class_name confidence

Maintains a list of model variants and iterates over them, so a single run can
export predictions for multiple checkpoints/configs in batch.

Entry Points
------------
Programmatic:
    ``infer_obb_and_export(img_dir, ckpt, config, output_dir, classes_txt, ...)``
    — run inference on one model variant, export to DOTA dir.

Batch via ``__main__``:
    Edit the ``infoes`` list and run::

        python test/tool_deimv2_obb_infer.py

Parameters (per model variant)
------------------------------
img_dir         : str   — directory of input images
ckpt            : str   — path to .pth checkpoint
config          : str   — training YAML config path
output_dir      : str   — directory for per-image .txt outputs
classes_txt     : str   — path to classes.txt (one class name per line)
imgsz           : tuple — (H, W) inference size, default (640, 640)
max_det         : int   — maximum number of detections per image (num_queries)
score_threshold : float — confidence threshold for filtering (default 0.2)
device          : str   — "cuda:0" or "cpu"

Output Structure
----------------
output_dir/
├── img_001.txt    # OBB detections in DOTA format
├── img_002.txt
└── ...

Each line::

    x1 y1 x2 y2 x3 y3 x4 y4 class_name confidence

Usage
-----
    python test/tool_deimv2_obb_infer.py
"""

import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np

from engine.backbone import DINOv3STAsResAtten
from engine.deim import HybridEncoder, DEIMTransformer
from engine.deim.postprocessor import PostProcessor
from engine.data.transforms import ConvertPILImage
from tools.model_compare.obb_utils import deimv2_obb_outputs_to_dota


class DEIMv2OBB(nn.Module):
    """DEIMv2-OBB model wrapper for inference."""

    def __init__(self, config: dict, device: str = "cpu"):
        super().__init__()
        self.backbone = DINOv3STAsResAtten(**config["DINOv3STAsResAtten"]).to(device)
        self.encoder = HybridEncoder(**config["HybridEncoder"]).to(device)
        self.decoder = DEIMTransformer(**config["DEIMTransformer"]).to(device)
        self.postprocessor = PostProcessor(**config["PostProcessor"]).to(device)

    def forward(self, x, orig_target_sizes):
        x1 = self.backbone(x)
        x2 = self.encoder(x1)
        x3 = self.decoder(x2)
        x4 = self.postprocessor(x3, orig_target_sizes)
        return x4


def load_checkpoint(model: nn.Module, ckpt_path: str, map_location: str = "cpu"):
    """Load checkpoint with standard key remapping (ema.module → model)."""
    state = torch.load(ckpt_path, weights_only=True, map_location=map_location)
    # common checkpoint structure: ckpt["ema"]["module"]
    if "ema" in state:
        state = state["ema"]
        if "module" in state:
            state = state["module"]
    elif "model" in state:
        state = state["model"]
    # strip DDP "module." prefix if present
    new_state = {}
    for k, v in state.items():
        new_state[k.replace("module.", "")] = v
    model.load_state_dict(new_state, strict=False)
    print(f"Loaded checkpoint from {ckpt_path}")
    return model


def infer_obb_and_export(
    img_dir: str,
    ckpt: str,
    config: str,
    output_dir: str,
    classes_txt: str,
    imgsz: tuple = (640, 640),
    max_det: int = 300,
    score_threshold: float = 0.0,
    device: str = "cuda:0",
):
    """Run OBB inference on all images and export to DOTA per-image format.

    num_classes is auto-detected from classes_txt.
    """
    os.makedirs(output_dir, exist_ok=True)

    # ── load class names ──
    with open(classes_txt, "r") as f:
        class_names = [line.strip() for line in f if line.strip()]
    num_classes = len(class_names)
    labels_map = {i: name for i, name in enumerate(class_names)}
    print(f"Classes ({num_classes}): {labels_map}")

    # ── build model config from training YAML ──
    from engine.core.yaml_utils import load_config

    config = load_config(config)
    model_cfg = {
        "DINOv3STAsResAtten": config["DINOv3STAsResAtten"],
        "HybridEncoder": config["HybridEncoder"],
        "DEIMTransformer": {
            **config["DEIMTransformer"],
            "num_classes": num_classes,
            "num_queries": max_det,
        },
        "PostProcessor": {
            **config["PostProcessor"],
            "num_classes": num_classes,
            "num_top_queries": max_det,
        },
    }
    # Path supplied by the caller overrides whatever the training config says
    model_cfg["DINOv3STAsResAtten"]["weights_path"] = ckpt

    model = DEIMv2OBB(model_cfg, device)
    load_checkpoint(model, ckpt)
    model.eval()

    transform = transforms.Compose(
        [
            transforms.Resize(imgsz),
            ConvertPILImage(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    img_list = [
        f
        for f in os.listdir(img_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    ]
    print(f"Found {len(img_list)} images")

    outputs_dict = {}

    for img_name in tqdm(img_list, desc="Inference"):
        img_path = os.path.join(img_dir, img_name)
        try:
            image = Image.open(img_path).convert("RGB")
            orig_w, orig_h = image.size
            input_tensor = transform(image).unsqueeze(0).to(device)
            dst_sz = torch.tensor([imgsz[0], imgsz[1]], device=device)[None, :]
            src_sz = torch.tensor([orig_h, orig_w], device=device)

            with torch.no_grad():
                results = model(input_tensor, orig_target_sizes=dst_sz)

            for output in results:
                labels = output["labels"].cpu().numpy()
                boxes = output["boxes"].cpu().numpy()
                scores = output["scores"].cpu().numpy()

                if len(labels) == 0:
                    continue

                # rescale OBB boxes: (cx, cy, w, h, θ)
                # cx, cy, w, h 乘缩放因子；θ 不变
                scale_y = orig_h / imgsz[0]
                scale_x = orig_w / imgsz[1]
                boxes[:, 0] *= scale_x  # cx
                boxes[:, 1] *= scale_y  # cy
                boxes[:, 2] *= scale_x  # w
                boxes[:, 3] *= scale_y  # h
                # boxes[:, 4] (θ) unchanged

                filtered_labels = []
                filtered_boxes = []
                filtered_scores = []
                for lbl, box, sc in zip(labels, boxes, scores):
                    if sc >= score_threshold:
                        filtered_labels.append(int(lbl))
                        filtered_boxes.append(box.tolist())
                        filtered_scores.append(float(sc))

                if filtered_labels:
                    outputs_dict[img_name] = {
                        "labels": filtered_labels,
                        "boxes": filtered_boxes,
                        "scores": filtered_scores,
                    }

        except Exception as e:
            print(f"Error processing {img_name}: {e}")
            continue

    print(f"\nInference completed: {len(outputs_dict)} images with detections")

    # ── export to DOTA format ──
    print(f"\nExporting to DOTA format → {output_dir}")
    deimv2_obb_outputs_to_dota(outputs_dict, output_dir, labels_map, score_threshold)

    print(f"\nDone. Predictions saved to {output_dir}/")


if __name__ == "__main__":
    img_dir = (
        "/mnt/d/project_data/model_test/deimv2_obb_train_data/dlzdt_obb_val/images/val"
    )
    classes_txt = (
        "/mnt/d/project_data/model_test/deimv2_obb_train_data/dlzdt_obb_val/classes.txt"
    )
    imgsz = (640, 640)
    max_det = 300
    score_threshold = 0.01
    device = "cuda:0"

    infoes = [
        {
            "config": "configs/custom_obb/dlzdt/hp_fz_rep0.yml",
            "ckpt": "outputs/hp_fz_rep0_0717.pth",
            "output_dir": "./test/data/outputs/dlzdt_res/hp_fz_rep0_0717_val",
        },
        # {
        #     "config": "configs/custom_obb/dlzdt/hp_fz_rep3.yml",
        #     "ckpt": "outputs/hp_fz_rep3_0717.pth",
        #     "output_dir": "./test/data/outputs/dlzdt_res/sp_ft_rep1_0715_val",
        # },
    ]

    for info in infoes:
        infer_obb_and_export(
            img_dir=img_dir,
            ckpt=info["ckpt"],
            config=info["config"],
            output_dir=info["output_dir"],
            classes_txt=classes_txt,
            imgsz=imgsz,
            max_det=max_det,
            score_threshold=score_threshold,
            device=device,
        )
