import os
import sys
import torch
import torch.nn as nn
from torchvision import transforms

from engine.backbone import HGNetv2, DINOv3STAs
from engine.deim import HybridEncoder, LiteEncoder
from engine.deim import DFINETransformer, DEIMTransformer
from engine.deim.postprocessor import PostProcessor

DEIMV2_X_CFG = {
    "DINOv3STAs": {
        "name": "dinov3_vits16plus",
        "weights_path": "",
        "embed_dim": 256,
        "interaction_indexes": [5, 8, 11],
        "num_heads": None,
        "conv_inplane": 64,
        "hidden_dim": 256,
    },
    "HybridEncoder": {
        "in_channels": [256, 256, 256],
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
        "num_classes": "",
        "feat_channels": [256, 256, 256],
        "feat_strides": [8, 16, 32],
        "hidden_dim": 256,
        "num_levels": 3,
        "num_layers": 6,
        "eval_idx": -1,
        "num_queries": 300,
        "num_denoising": 100,
        "label_noise_ratio": 0.5,
        "box_noise_scale": 1.0,
        "reg_max": 32,
        "reg_scale": 4,
        "layer_scale": 1,
        "num_points": [3, 6, 3],
        "cross_attn_method": "default",
        "query_select_method": "default",
        "activation": "silu",
        "mlp_act": "silu",
        "dim_feedforward": 2048,
        "eval_spatial_size": [640, 640],
    },
    "PostProcessor": {"num_classes": "", "num_top_queries": 50},
}


class DEIMv2(nn.Module):
    def __init__(self, config, device="cpu"):
        super().__init__()
        if "DINOv3STAs" in config:
            self.backbone = DINOv3STAs(**config["DINOv3STAs"]).to(device)
        else:
            self.backbone = HGNetv2(**config["HGNetv2"]).to(device)
        if "LiteEncoder" in config:
            self.encoder = LiteEncoder(**config["LiteEncoder"]).to(device)
        else:
            self.encoder = HybridEncoder(**config["HybridEncoder"]).to(device)
        if "DEIMTransformer" in config:
            self.decoder = DEIMTransformer(**config["DEIMTransformer"]).to(device)
        else:
            self.decoder = DFINETransformer(**config["DFINETransformer"]).to(device)
        self.postprocessor = PostProcessor(**config["PostProcessor"]).to(device)

    def forward(self, x, orig_target_sizes):
        x1 = self.backbone(x)
        x2 = self.encoder(x1)
        x3 = self.decoder(x2)
        x4 = self.postprocessor(x3, orig_target_sizes)

        return x4


class DEIMv2Det:
    def __init__(self, model_weight, num_classes, imgsz, max_det, device="cpu"):
        self.config = DEIMV2_X_CFG
        self.config["DINOv3STAs"].update({"weights_path": model_weight})
        self.config["DEIMTransformer"].update({"num_classes": num_classes})
        self.config["DEIMTransformer"].update({"num_queries": max_det})
        self.config["PostProcessor"].update({"num_classes": num_classes})
        self.dimv2 = DEIMv2(self.config, device)
        self.model_weight = torch.load(model_weight, weights_only=True)["ema"]["module"]
        self.dimv2.load_state_dict(self.model_weight)
        self.dimv2.eval()
        self.transform = transforms.Compose(
            [
                transforms.Resize(imgsz),
                transforms.ToTensor(),
            ]
        )

    def infer(
        self,
        inputs,
        score,
    ):
        inputs = self.transform(inputs).unsqueeze(0)
        with torch.no_grad():
            outputs = self.dimv2(inputs)

        labels_batch = []
        bboxes_batch = []
        scores_batch = []
        for output in outputs:
            all_labels = output["labels"]
            all_bboxes = outputs["boxes"]
            all_scores = outputs["scores"]
            tmp_labels, tmp_bboxes, tmp_scores = [], [], []

            for label, box, score in zip(all_labels, all_bboxes, all_scores):
                if score > score:
                    tmp_labels.append(label)
                    tmp_bboxes.append(box)
                    tmp_scores.append(score)
            labels_batch.append(tmp_labels)
            bboxes_batch.append(tmp_bboxes)
            scores_batch.append(tmp_scores)

        return {
            "labels": labels_batch,
            "boxes": bboxes_batch,
            "scores": scores_batch,
        }
