import os
import sys
import torch
import torch.nn as nn
from torchvision import transforms

from engine.backbone import HGNetv2, DINOv3STAs
from engine.deim import HybridEncoder, LiteEncoder
from engine.deim import DFINETransformer, DEIMTransformer
from engine.deim.postprocessor import PostProcessor
from .deimv2_model_config import DEIMV2_X_CFG, DEIMV2_B_CFG


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

    def __init__(
        self,
        model_weight,
        num_classes,
        imgsz,
        max_det,
        config=DEIMV2_X_CFG,
        device="cpu",
    ):
        self.config = config
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
