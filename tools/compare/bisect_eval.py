#!/usr/bin/env python3
"""
Bisect: compare obb_evaluate results between training-style and inference-style
model loading, using the same checkpoint and val dataloader.

Usage:
    python tools/compare/bisect_eval.py
    python tools/compare/bisect_eval.py -c configs/custom_obb/dlzdt/sp_fz_common.yml -r outputs/dlzdt_ablation/abl_rep0.pth -d cuda:0

Paths compared:
    A  Training-style — YAMLConfig.model + EMA weights → obb_evaluate
    B  Inference-style — DEIMv2OBB + load_checkpoint → obb_evaluate
    B2 Inference-style without weights_path override → obb_evaluate

If all three produce identical results, the loading path is NOT the source of
metric discrepancy.  If any path differs, the corresponding loading code has a bug.
"""

import argparse
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import torch
import torch.nn as nn

from engine.core import YAMLConfig
from engine.core.yaml_utils import load_config as load_yaml_config
from engine.eval.obb_eval import obb_evaluate

DEFAULT_DEVICE = "cuda:0"


def _load_ema_weights(ckpt_path, model):
    """Load EMA-smoothed weights from training checkpoint into model."""
    state = torch.load(ckpt_path, map_location="cpu")
    if "ema" in state:
        weight_source = state["ema"]["module"]
        label = "ema.module"
    elif "model" in state:
        weight_source = state["model"]
        label = "model"
    else:
        weight_source = state
        label = "bare"
    print(f"  weight source: {label} ({len(weight_source)} keys)")

    cleaned = {k.replace("module.", ""): v for k, v in weight_source.items()}
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        print(f"  missing keys: {len(missing)}")
    if unexpected:
        print(f"  unexpected keys: {len(unexpected)}")
    if not missing and not unexpected:
        print("  all keys matched")
    return model


class DEIMv2OBB(nn.Module):
    """Inference-only model wrapper (matching tool_deimv2_obb_infer.py)."""

    def __init__(self, cfg, device):
        super().__init__()
        from engine.backbone import DINOv3STAsResAtten
        from engine.deim import HybridEncoder, DEIMTransformer
        from engine.deim.postprocessor import PostProcessor

        self.backbone = DINOv3STAsResAtten(**cfg["DINOv3STAsResAtten"]).to(device)
        self.encoder = HybridEncoder(**cfg["HybridEncoder"]).to(device)
        self.decoder = DEIMTransformer(**cfg["DEIMTransformer"]).to(device)
        self.postprocessor = PostProcessor(**cfg["PostProcessor"]).to(device)

    def forward(self, x, targets=None):
        x1 = self.backbone(x)
        x2 = self.encoder(x1)
        x3 = self.decoder(x2, targets)
        return x3


def _build_infer_model_cfg(yml_config, num_classes):
    return {
        "DINOv3STAsResAtten": yml_config["DINOv3STAsResAtten"],
        "HybridEncoder": yml_config["HybridEncoder"],
        "DEIMTransformer": {
            **yml_config["DEIMTransformer"],
            "num_classes": num_classes,
            "num_queries": 300,
        },
        "PostProcessor": {
            **yml_config["PostProcessor"],
            "num_classes": num_classes,
            "num_top_queries": 300,
        },
    }


# ─────────────────────────────────────────────────────────────────
#  Path A: Training-style (YAMLConfig + EMA weights)
# ─────────────────────────────────────────────────────────────────
def eval_path_a(config_path, ckpt_path, device, data_overrides):
    print("\n" + "=" * 60)
    print("Path A: Training-style (YAMLConfig.model + EMA)")
    print("=" * 60)

    cfg = YAMLConfig(config_path, val_dataloader=data_overrides)
    model = cfg.model
    postprocessor = cfg.postprocessor
    val_loader = cfg.val_dataloader

    model.to(device)
    postprocessor.to(device)
    _load_ema_weights(ckpt_path, model)
    model.eval()
    postprocessor.eval()

    results = obb_evaluate(
        model, postprocessor, val_loader, device,
        num_classes=postprocessor.num_classes,
    )
    _print_result("Path A", results)
    return results


# ─────────────────────────────────────────────────────────────────
#  Path B: Inference-style (DEIMv2OBB + load_checkpoint)
# ─────────────────────────────────────────────────────────────────
def eval_path_b(config_path, ckpt_path, device, data_overrides):
    print("\n" + "=" * 60)
    print("Path B: Inference-style (DEIMv2OBB + load_checkpoint)")
    print("=" * 60)

    yml_cfg = load_yaml_config(config_path)
    num_classes = yml_cfg.get("num_classes", 1)
    model_cfg = _build_infer_model_cfg(yml_cfg, num_classes)
    model_cfg["DINOv3STAsResAtten"]["weights_path"] = ckpt_path

    model = DEIMv2OBB(model_cfg, device)

    state = torch.load(ckpt_path, weights_only=True, map_location="cpu")
    if "ema" in state:
        state = state["ema"]
        if "module" in state:
            state = state["module"]
    elif "model" in state:
        state = state["model"]
    new_state = {k.replace("module.", ""): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(new_state, strict=False)
    print(f"  missing={len(missing)}, unexpected={len(unexpected)}")
    model.eval()

    cfg = YAMLConfig(config_path, val_dataloader=data_overrides)
    postprocessor = cfg.postprocessor.to(device)
    val_loader = cfg.val_dataloader

    results = obb_evaluate(
        model, postprocessor, val_loader, device,
        num_classes=num_classes,
    )
    _print_result("Path B", results)
    return results


# ─────────────────────────────────────────────────────────────────
#  Path B2: Inference-style WITHOUT weights_path override
# ─────────────────────────────────────────────────────────────────
def eval_path_b2(config_path, ckpt_path, device, data_overrides):
    print("\n" + "=" * 60)
    print("Path B2: Inference-style (no weights_path override)")
    print("=" * 60)

    yml_cfg = load_yaml_config(config_path)
    num_classes = yml_cfg.get("num_classes", 1)
    model_cfg = _build_infer_model_cfg(yml_cfg, num_classes)
    # DO NOT override weights_path

    model = DEIMv2OBB(model_cfg, device)

    state = torch.load(ckpt_path, weights_only=True, map_location="cpu")
    if "ema" in state:
        state = state["ema"]
        if "module" in state:
            state = state["module"]
    elif "model" in state:
        state = state["model"]
    new_state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(new_state, strict=False)
    model.eval()

    cfg = YAMLConfig(config_path, val_dataloader=data_overrides)
    postprocessor = cfg.postprocessor.to(device)
    val_loader = cfg.val_dataloader

    results = obb_evaluate(
        model, postprocessor, val_loader, device,
        num_classes=num_classes,
    )
    _print_result("Path B2", results)
    return results


def _print_result(label, results):
    print(f"\n{label}: mAP50={results['mAP50']:.4f}  "
          f"mAP50_95={results['mAP50_95']:.4f}  "
          f"recall={results['recall']:.4f}  "
          f"precision={results['precision']:.4f}")


# ─────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bisect eval paths for OBB model loading")
    parser.add_argument("-c", "--config", default="configs/custom_obb/dlzdt/sp_fz_common.yml")
    parser.add_argument("-r", "--ckpt", default="outputs/dlzdt_ablation/abl_rep0.pth")
    parser.add_argument("-d", "--device", default=DEFAULT_DEVICE)
    parser.add_argument("--data-base",
        default="/mnt/d/project_data/model_test/deimv2_obb_train_data/dlzdt_obb_val")
    parser.add_argument("--paths", nargs="+", choices=["a", "b", "b2"], default=["a", "b", "b2"],
        help="which paths to run (default: all)")
    args = parser.parse_args()

    data_overrides = {
        "dataset": {
            "img_folder": os.path.join(args.data_base, "images", "val"),
            "ann_folder": os.path.join(args.data_base, "labels", "val"),
            "classes_file": os.path.join(args.data_base, "classes.txt"),
        }
    }

    results = {}
    if "a" in args.paths:
        results["A"] = eval_path_a(args.config, args.ckpt, args.device, data_overrides)
    if "b" in args.paths:
        results["B"] = eval_path_b(args.config, args.ckpt, args.device, data_overrides)
    if "b2" in args.paths:
        results["B2"] = eval_path_b2(args.config, args.ckpt, args.device, data_overrides)

    if len(results) > 1:
        print("\n" + "=" * 60)
        print("COMPARISON")
        print("=" * 60)
        keys = ["mAP50", "mAP50_95", "recall", "precision", "AP50", "AP75"]
        header = f"{'Metric':<14}" + "".join(f"  {k:<14}" for k in results)
        print(header)
        print("-" * len(header))
        for k in keys:
            vals = "".join(f"  {results[r].get(k, 0):<14.4f}" for r in results)
            print(f"{k:<14}{vals}")

        first = list(results.values())[0]
        all_same = all(
            abs(r.get("mAP50", 0) - first.get("mAP50", 0)) < 0.002
            for r in results.values()
        )
        if all_same:
            print("\n✓ All paths produce identical results — model loading is NOT the issue.")
        else:
            print("\n✗ Paths differ! Investigate the divergent loading code.")
