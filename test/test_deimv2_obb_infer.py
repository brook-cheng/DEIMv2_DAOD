"""
DEIMv2-OBB 模型推理与结果导出脚本

将 OBB 检测结果保存为 per-image DOTA 格式，支持后续模型比对。

Usage:
    python test/test_deimv2_obb_infer.py
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
    model_weight: str,
    output_dir: str,
    classes_txt: str,
    imgsz: tuple = (640, 640),
    max_det: int = 300,
    score_threshold: float = 0.0,
    device: str = "cuda:0",
    angle_rep=0,
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

    # ── build model config (must match training config) ──
    config = {
        "DINOv3STAsResAtten": {
            "name": "dinov3_vits16plus",
            "weights_path": model_weight,
            "interaction_indexes": [],
            "finetune": False,
            "conv_inplane": 64,
            "hidden_dim": 256,
        },
        "HybridEncoder": {
            "in_channels": [
                256,
                256,
                256,
            ],  # DINOv3-STAs-ResAtten projects all layers to hidden_dim
            "feat_strides": [8, 16, 32],
            "hidden_dim": 256,
            "use_encoder_idx": [2],
            "num_encoder_layers": 1,
            "nhead": 8,
            "dim_feedforward": 1024,
            "dropout": 0.0,
            "enc_act": "gelu",
            "expansion": 1.25,
            "depth_mult": 1.37,
            "act": "silu",
            "version": "deim",
            "csp_type": "csp2",
            "fuse_op": "sum",
        },
        "DEIMTransformer": {
            "box_mode": "obb",
            "angle_rep": angle_rep,
            "feat_channels": [256, 256, 256],
            "feat_strides": [8, 16, 32],
            "hidden_dim": 256,
            "dim_feedforward": 2048,
            "num_levels": 3,
            "num_layers": 6,
            "eval_idx": -1,
            "num_queries": max_det,
            "num_classes": num_classes,
            "reg_max": 32,
            "reg_scale": 4,
            "num_points": [3, 6, 3],
            "cross_attn_method": "default",
            "query_select_method": "default",
            "activation": "silu",
            "mlp_act": "silu",
        },
        "PostProcessor": {
            "box_mode": "obb",
            "num_top_queries": max_det,
            "num_classes": num_classes,
            "use_focal_loss": True,
        },
    }

    print(f"Building DEIMv2-OBB model (backbone=DINOv3-STAs-ResAtten, imgsz={imgsz})")
    model = DEIMv2OBB(config, device)
    load_checkpoint(model, model_weight)
    model.eval()

    transform = transforms.Compose(
        [
            transforms.Resize(imgsz),
            transforms.ToTensor(),
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
    img_dir = "/mnt/d/project_data/model_test/deimv2_obb_train_data/synthetic_ellipse/density_020/val"
    classes_txt = "/mnt/d/project_data/model_test/deimv2_obb_train_data/synthetic_ellipse/classes.txt"
    angle_rep = 3
    output_dir = f"./test/data/outputs/exp_020_anrep{angle_rep}"
    model_weight = f"/home/cx/win_dir/thired/DEIMv2_DAOD/outputs/synthetic_exp_020_anrep{angle_rep}_offset_per/last.pth"
    imgsz = (640, 640)
    max_det = 50
    score_threshold = 0.2
    device = "cuda:0"

    infer_obb_and_export(
        img_dir=img_dir,
        model_weight=model_weight,
        output_dir=output_dir,
        classes_txt=classes_txt,
        imgsz=imgsz,
        max_det=max_det,
        score_threshold=score_threshold,
        device=device,
        angle_rep=angle_rep,
    )
