import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if "ROOT_DIR" in globals():
    sys.path.append(ROOT_DIR)

import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
from PIL import Image

from engine.backbone import HGNetv2, DINOv3STAs, DINOv3STAsResAtten
from engine.deim import HybridEncoder, LiteEncoder
from engine.deim import DFINETransformer, DEIMTransformer
from engine.deim.postprocessor import PostProcessor
from deim_wapper.deimv2_model_config import (
    DEIMV2_X_CFG,
    DEIMV2_B_CFG,
    DEIMV2_VITL16P_CFG,
)


class DEIMv2(nn.Module):
    def __init__(self, config, device="cpu"):
        super().__init__()
        if "DINOv3STAs" in config:
            self.backbone = DINOv3STAs(**config["DINOv3STAs"]).to(device)
        elif "DINOv3STAsResAtten" in config:
            self.backbone = DINOv3STAsResAtten(**config["DINOv3STAsResAtten"]).to(
                device
            )
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
        model_weight: str,
        num_classes: int,
        imgsz: tuple = (640, 640),
        max_det: int = 50,
        config=DEIMV2_X_CFG,
        device="cpu",
    ):
        self.config = config
        if "DINOv3STAsResAtten" in self.config:
            self.config["DINOv3STAsResAtten"].update({"weights_path": model_weight})
        elif "DINOv3STAs" in self.config:
            self.config["DINOv3STAs"].update({"weights_path": model_weight})
        self.config["DEIMTransformer"].update({"num_classes": num_classes})
        self.config["DEIMTransformer"].update({"num_queries": max_det})
        self.config["PostProcessor"].update({"num_classes": num_classes})
        self.imgsz = imgsz
        self.device = device
        self.dimv2 = DEIMv2(self.config, device)
        self.model_weight = torch.load(
            model_weight, weights_only=True, map_location="cpu"
        )["ema"]["module"]
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
        input_tensor = self.transform(inputs).unsqueeze(0).to(self.device)
        src_sz = torch.tensor(inputs.size).to(self.device)
        dst_sz = torch.Tensor(self.imgsz).to(self.device)
        with torch.no_grad():
            outputs = self.dimv2(input_tensor, orig_target_sizes=dst_sz)

        labels_batch = []
        bboxes_batch = []
        scores_batch = []
        dst_sz_repeated = dst_sz.repeat(1, 2).unsqueeze(1)
        src_sz_repeated = src_sz.repeat(1, 2).unsqueeze(1)
        for output in outputs:
            all_labels = output["labels"].to("cpu").numpy()
            all_bboxes = (
                (output["boxes"] / dst_sz_repeated * src_sz_repeated)
                .to("cpu")
                .numpy()
                .astype(np.int64)[0]
            )
            all_scores = output["scores"].to("cpu").numpy()
            tmp_labels, tmp_bboxes, tmp_scores = [], [], []

            for label, box, cu_score in zip(all_labels, all_bboxes, all_scores):
                if cu_score > score:
                    tmp_labels.append(label.item())
                    tmp_bboxes.append(box.tolist())
                    tmp_scores.append(cu_score.item())
            labels_batch.append(tmp_labels)
            bboxes_batch.append(tmp_bboxes)
            scores_batch.append(tmp_scores)

        return {
            "labels": labels_batch,
            "boxes": bboxes_batch,
            "scores": scores_batch,
        }
