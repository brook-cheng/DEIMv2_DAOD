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
        self.config["DINOv3STAsResAtten"].update({"weights_path": model_weight})
        self.config["DEIMTransformer"].update({"num_classes": num_classes})
        self.config["DEIMTransformer"].update({"num_queries": max_det})
        self.config["PostProcessor"].update({"num_classes": num_classes})
        self.imgsz = imgsz
        self.device = device
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


def visulize_outputs(img, outputs, labels_map=None, score_threshold=0.5):
    from PIL import ImageDraw, ImageFont
    import matplotlib.pyplot as plt

    img = img.copy()
    draw = ImageDraw.Draw(img)

    img_width, img_height = img.size
    reference_size = 640
    scale_factor = max(img_width, img_height) / reference_size

    line_width = max(2, int(3 * scale_factor))
    font_size = max(12, int(16 * scale_factor))

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size
        )
    except:
        font = ImageFont.load_default()

    if labels_map is None:
        labels_map = {i: f"class_{i}" for i in range(10)}

    num_classes = len(labels_map)
    colors = plt.cm.rainbow(np.linspace(0, 1, num_classes))
    color_map = {
        label: tuple([int(c * 255) for c in colors[i]])
        for i, label in enumerate(labels_map.keys())
    }

    batch_labels = outputs["labels"]
    batch_boxes = outputs["boxes"]
    batch_scores = outputs["scores"]

    for img_idx in range(len(batch_labels)):
        labels = batch_labels[img_idx]
        boxes = batch_boxes[img_idx]
        scores = batch_scores[img_idx]

        for label, box, score in zip(labels, boxes, scores):
            if score < score_threshold:
                continue

            x_min, y_min, x_max, y_max = box
            color = color_map.get(label, (255, 0, 0))

            draw.rectangle(
                [x_min, y_min, x_max, y_max], outline=color, width=line_width
            )

            label_name = labels_map.get(label, f"class_{label}")
            text = f"{label_name}: {score:.2f}"

            text_bbox = draw.textbbox((x_min, y_min), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]

            if y_min - text_height >= 0:
                text_y = y_min - text_height
            else:
                text_y = y_min

            draw.rectangle(
                [text_bbox[0], text_y, text_bbox[2], text_y + text_height], fill=color
            )

            draw.text((x_min, text_y), text, fill="white", font=font)

    return img


def __test():
    from tqdm import tqdm

    img_dir = (
        "/home/cx/cx_dir/data/deimv2_train_data/dlzdt_dataset_20260331_hbb/images/val"
    )
    output_img_dir = "./test/outputs"

    model_weight = "outputs/dlzdt_vitl16_freeze_extend/best_stg2.pth"
    num_classes = 3
    imgsz = (640, 640)
    max_det = 20
    config = DEIMV2_VITL16P_CFG
    model = DEIMv2Det(
        model_weight, num_classes, imgsz, max_det, DEIMV2_VITL16P_CFG, "cuda:1"
    )

    total_cost = 0
    for img_name in tqdm(os.listdir(img_dir)):
        if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        img_path = os.path.join(img_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        outputs = model.infer(image, 0.5)
        image = visulize_outputs(
            image,
            outputs,
            labels_map={0: "background", 1: "dlzdt", 2: "null"},
            score_threshold=0.5,
        )
        # print(outputs)
        image.save(os.path.join(output_img_dir, img_name))


if __name__ == "__main__":
    __test()
