import os
import sys

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin

from engine.backbone import HGNetv2, DINOv3STAs
from engine.deim import HybridEncoder, LiteEncoder
from engine.deim import DFINETransformer, DEIMTransformer
from engine.deim.postprocessor import PostProcessor

import colorsys
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont

# There is an example in the end!
os.environ.update({"CUDA_VISIBLE_DEVICES": "0"})


class DEIMv2(nn.Module, PyTorchModelHubMixin):
    def __init__(self, config):
        super().__init__()
        if "DINOv3STAs" in config:
            self.backbone = DINOv3STAs(**config["DINOv3STAs"])
        else:
            self.backbone = HGNetv2(**config["HGNetv2"])
        if "LiteEncoder" in config:
            self.encoder = LiteEncoder(**config["LiteEncoder"])
        else:
            self.encoder = HybridEncoder(**config["HybridEncoder"])
        if "DEIMTransformer" in config:
            self.decoder = DEIMTransformer(**config["DEIMTransformer"])
        else:
            self.decoder = DFINETransformer(**config["DFINETransformer"])
        self.postprocessor = PostProcessor(**config["PostProcessor"])

    def forward(self, x, orig_target_sizes):
        x1 = self.backbone(x)
        x2 = self.encoder(x1)
        x3 = self.decoder(x2)
        x4 = self.postprocessor(x3, orig_target_sizes)

        return x1, x2, x3, x4


COCO_WEIGHTS_PATH = "./ckpts/down_stream_task_models/deimv2_dinov3_x_coco.pth"
CUSTOM_WEIGHTS_PATH = (
    "./outputs/deimv2_dinov3_x_custom/best_stg2_freeze_1109_e186_mAP67.pth"
)
CUSTOM_VAL_WEIGHTS_PATH = "./outputs/model_weight_init_test/model_val_weigth_saved.pth"

CUSTOM_NUM_CLASSES = 2

deimv2_x_config = {
    "DINOv3STAs": {
        "name": "dinov3_vits16plus",
        "weights_path": CUSTOM_WEIGHTS_PATH,
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
        "num_classes": CUSTOM_NUM_CLASSES,
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
    "PostProcessor": {"num_classes": CUSTOM_NUM_CLASSES, "num_top_queries": 50},
}

deimv2_x = DEIMv2(deimv2_x_config)
model_wight = torch.load(CUSTOM_WEIGHTS_PATH, weights_only=True)["ema"]["module"]
coco_model_wight = torch.load(COCO_WEIGHTS_PATH, weights_only=True)
model_wight_val = torch.load(CUSTOM_VAL_WEIGHTS_PATH, weights_only=True)

for (x_k, x_v), (val_x_k, val_x_v) in zip(model_wight.items(), model_wight_val.items()):
    if x_k != val_x_k:
        print("key_diff", x_k, val_x_k)
    if x_v.data.all() != val_x_v.data.all():
        diff = x_v - val_x_v
        print(f"val_diff:{x_k} cus:{x_v} val:{val_x_v}")


deimv2_x.load_state_dict(model_wight)


MODEL = deimv2_x

IMAGE_SIZE = (640, 640)  # resolution settings of model, you can find in config
# IMAGE_PATH = "test/pose_test.jpeg"  # path of input image
IMAGE_PATH = "./test/DJI_20250821061356_0001_S_frame_000123_20251009.jpg"
CONFIDENCE_THRESHOLD = 0.2

font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
font_size = 25
font = ImageFont.truetype(font_path, font_size)

label_map = {1: "bh_guagoutou"}

num_classes = len(label_map)
category_colors = {}
for i, label in enumerate(label_map.values()):
    hue = i / num_classes
    saturation = 0.8
    value = 0.9
    rgb = colorsys.hsv_to_rgb(hue, saturation, value)
    rgb = tuple(int(c * 255) for c in rgb)
    category_colors[label] = rgb

image = Image.open(IMAGE_PATH).convert("RGB")

transform = transforms.Compose(
    [
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
    ]
)

input_tensor = transform(image).unsqueeze(0)

MODEL.eval()
with torch.no_grad():
    hf_x1, hf_x2, hf_x3, outputs = MODEL(
        input_tensor, orig_target_sizes=torch.tensor([IMAGE_SIZE])
    )


class_labels, bboxes, scores = (
    outputs[0]["labels"],
    outputs[0]["boxes"],
    outputs[0]["scores"],
)

detections = []
for label, bbox, score in zip(class_labels, bboxes, scores):
    if score.item() >= CONFIDENCE_THRESHOLD:
        label_id = label.item() + 1
        label_id = 1
        detection = {
            "label": label_map[label_id],
            "bounding_box": [
                bbox[0] / IMAGE_SIZE[0],
                bbox[1] / IMAGE_SIZE[1],
                (bbox[2] - bbox[0]) / IMAGE_SIZE[0],
                (bbox[3] - bbox[1]) / IMAGE_SIZE[1],
            ],
            "confidence": score.item(),
        }
        detections.append(detection)

draw = ImageDraw.Draw(image)
for detection in detections:
    label = detection["label"]
    confidence = detection["confidence"]
    bbox = detection["bounding_box"]

    x_min = bbox[0] * image.width
    y_min = bbox[1] * image.height
    x_max = (bbox[0] + bbox[2]) * image.width
    y_max = (bbox[1] + bbox[3]) * image.height

    color = category_colors[label]

    draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=2)
    # draw.text((x_min, y_min - 13), f"{label}: {confidence:.1f}", fill="red", font=font)
    # confidence = 0.25
    draw.text((x_min, y_min - 13), f"{label}: {confidence:.3f}", fill="red", font=font)

# image.show()
image.save("./outputs/x_custom_test/output.jpg")
print("Inference completed and output image saved.")
