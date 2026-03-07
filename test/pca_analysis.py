import os
import sys

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin
import torch
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont

from sklearn.decomposition import PCA
import umap
import numpy as np

from engine.backbone import HGNetv2, DINOv3STAs
from engine.deim import HybridEncoder, LiteEncoder
from engine.deim import DFINETransformer, DEIMTransformer
from engine.deim.postprocessor import PostProcessor

# There is an example in the end!


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


CUSTOM_WEIGHTS_PATH = (
    "./outputs/deimv2_dinov3_x_custom/best_stg2_freeze_1109_e186_mAP67.pth"
)
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
    "PostProcessor": {"num_top_queries": 300},
}

deimv2_x = DEIMv2(deimv2_x_config)
model_wight = torch.load(CUSTOM_WEIGHTS_PATH, weights_only=True)["ema"]["module"]
deimv2_x.load_state_dict(model_wight)

MODEL = deimv2_x
IMAGE_SIZE = (640, 640)  # resolution settings of model, you can find in config
# IMAGE_PATH = "test/pose_test.jpeg"  # path of input image
IMAGE_PATH = "/home/cx/cx_dir/data/deimv2_train_data/safetyhook_detection/test_dataset/images/test/DJI_20250821054840_0001_S_frame_000000_20250928.jpg"
CONFIDENCE_THRESHOLD = 0.01

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
    x1, x2, x3, outputs = MODEL(
        input_tensor, orig_target_sizes=torch.tensor([IMAGE_SIZE])
    )

n_components = 2
for features in x1:
    features = features.cpu().numpy()[0][0]
    print("Feature Map Shape:", features.shape)
    pca = PCA(n_components=n_components).fit(features)
    pca_descriptors = pca.transform(features)
    print("PCA Components Shape:", pca_descriptors.shape)

    reducer = umap.UMAP(n_components=2, random_state=42)
    umap_embeddings = reducer.fit_transform(features)
    print()
