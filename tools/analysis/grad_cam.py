import torch
import torchvision
import torch.nn as nn
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        self.target_layer = target_layer
        self.feature_maps = None
        self.gradients = None

        # 注册前向和反向钩子
        self.target_layer.register_forward_hook(self._save_feature_maps)
        self.target_layer.register_backward_hook(self._save_gradients)

    def _save_feature_maps(self, module, input, output):
        self.feature_maps = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def gen_heatmap(self, input_image):
        output = self.model(input_image)

        target_class = output.argmax().item()

        self.model.zero_grad()
        output[0, target_class].backward()

        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])

        for i in range(self.feature_maps.shape[1]):
            self.feature_maps[:, i, :, :] *= pooled_gradients[i]

        heatmap = torch.mean(self.feature_maps, dim=1).squeeze()
        heatmap = torch.nn.functional.relu(heatmap)

        heatmap = heatmap.cpu().numpy()
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)

        return heatmap

    def visualize(self, input_image, original_image, target_class=None, alpha=0.4):
        heatmap = self.gen_heatmap(input_image, target_class)
        heatmap = cv2.resize(
            heatmap, (original_image.shape[1], original_image.shape[0])
        )
        heatmap_colored = cv2.applyColorMap(
            np.uint8(255 * heatmap), cv2.COLORMAP_VIRIDIS
        )
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(original_image, 1 - alpha, heatmap_colored, alpha, 0)

        return overlay, heatmap


def _test():
    import os
    import sys

    ROOT_PATH = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    if ROOT_PATH not in sys.path:
        sys.path.append(ROOT_PATH)

    from engine.backbone.dinov3_adapter import DINOv3STAs
    from deimv2_det import DEIMV2_X_CFG

    deimv2_backboen_cfg = DEIMV2_X_CFG["DINOv3STAs"]
    deimv2_backboen_cfg.update(
        {
            "weights_path": "./outputs/deimv2_dinov3_x_custom/best_stg2_fintune_1110_e186_mAP68.pth"
        }
    )
    img_path = "./test/DJI_20250821061356_0001_S_frame_000123_20251009.jpg"
    image = Image.open(img_path).convert("RGB")
    dinov3_adapter_backbone = DINOv3STAs(**deimv2_backboen_cfg)
    imgsz = (640, 640)
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(imgsz),
            torchvision.transforms.ToTensor(),
        ]
    )
    print(dinov3_adapter_backbone)
    inputs = transform(image).unsqueeze(0)
    grad_cam = GradCAM(dinov3_adapter_backbone)


if __name__ == "__main__":
    _test()
