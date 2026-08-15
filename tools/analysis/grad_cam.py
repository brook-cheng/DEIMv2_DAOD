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

    def _get_sem_feats(self, input_image, feature_idx):
        # 从输入图像计算基础尺寸
        H_c = input_image.shape[2] // 16
        W_c = input_image.shape[3] // 16
        all_layers = self.model.dinov3.get_intermediate_layers(
            input_image, n=[feature_idx], return_class_token=True
        )
        num_scales = len(all_layers) - 2
        sem_feats = []

        for i, sem_feat in enumerate(all_layers):
            feat, _ = sem_feat
            # 转换特征维度 [B, N, C] -> [B, C, H, W]
            sem_feat = feat.transpose(1, 2).view(1, -1, H_c, W_c).contiguous()
            # 上采样到对应尺度
            resize_H, resize_W = int(H_c * 2 ** (num_scales - i)), int(
                W_c * 2 ** (num_scales - i)
            )
            sem_feat = torch.nn.functional.interpolate(
                sem_feat,
                size=[resize_H, resize_W],
                mode="bilinear",
                align_corners=False,
            )
            sem_feats.append(sem_feat)

        return sem_feats

    def gen_heatmap(self, input_image, feature_idx=0, feature_source="convs"):
        # 前向传播获取特征金字塔
        outputs = self.model(input_image)

        if feature_source == "convs":
            if isinstance(outputs, tuple):
                feature_map = outputs[feature_idx]
            else:
                feature_map = self.feature_maps
        elif feature_source == "sem_feats":
            sem_feats = self._get_sem_feats(input_image, feature_idx)
            feature_map = sem_feats[0]

        # 使用特征图L2范数作为反向目标
        # FIXME: 这里的loss设置似乎不合理，而且这个loss计算后没有参与到后续特征可视化步骤中
        loss = torch.sum(feature_map**2)

        self.model.zero_grad()
        loss.backward()

        # 池化梯度
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])

        # 加权特征图
        for i in range(self.feature_maps.shape[1]):
            self.feature_maps[:, i, :, :] *= pooled_gradients[i]

        # 生成热力图
        heatmap = torch.mean(self.feature_maps, dim=1).squeeze()
        heatmap = torch.nn.functional.relu(heatmap)
        heatmap = heatmap.cpu().numpy()
        if np.max(heatmap) == 0:
            return heatmap
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)
        return heatmap

    def visualize(
        self,
        input_image,
        original_image,
        alpha=0.5,
        feature_idx=0,
        feature_source="convs",
    ):
        heatmap = self.gen_heatmap(input_image, feature_idx, feature_source)
        heatmap = cv2.resize(
            heatmap, (original_image.shape[1], original_image.shape[0])
        )
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(original_image, 1 - alpha, heatmap_colored, alpha, 0)

        return overlay, heatmap_colored


class EigenCAM(GradCAM):
    def __init__(self, model, target_layer=None):
        self.model = model
        self.model.eval()
        self.feature_maps = None
        self.H_c = None
        self.W_c = None

    def gen_heatmap(self, input_image, feature_idx=0, feature_source="convs"):
        # 前向传播获取特征
        outputs = self.model(input_image)

        if feature_source == "convs":
            if isinstance(outputs, tuple):
                feature_map = outputs[feature_idx]
            else:
                feature_map = self.feature_maps
        elif feature_source == "sem_feats":
            sem_feats = self._get_sem_feats(input_image, feature_idx)
            feature_map = sem_feats[0]

        # 展平特征图 [C, H*W]
        C, H, W = feature_map.shape[1], feature_map.shape[2], feature_map.shape[3]
        flattened = feature_map.view(C, -1)

        # PCA计算
        U, _, _ = torch.pca_lowrank(flattened, q=1)
        heatmap = torch.matmul(U[:, 0], flattened).view(H, W).detach()

        # ReLU并归一化
        heatmap = torch.nn.functional.relu(heatmap)
        heatmap = heatmap.cpu().numpy()
        if np.max(heatmap) == 0:
            return heatmap
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)
        return heatmap


def _test():
    import os
    import sys

    ROOT_PATH = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    if ROOT_PATH not in sys.path:
        sys.path.append(ROOT_PATH)

    from engine.backbone.dinov3_adapter import DINOv3STAs
    from engine.core.yaml_config import YAMLConfig

    backbone_section = YAMLConfig(
        os.path.join(
            ROOT_PATH,
            "configs/custom/deimv2_dinov3_vits16p_freeze_test_eiou.yml",
        )
    ).global_cfg["DINOv3STAs"]
    backbone_kwargs = {
        k: v
        for k, v in backbone_section.items()
        if not k.startswith("_") and k != "type"
    }
    # DINOv3STAs strips a full fine-tuned checkpoint down to backbone.dinov3.*.
    backbone_kwargs["weights_path"] = (
        "./outputs/deimv2_dinov3_x_custom/best_stg2_freeze_1109_e186_mAP67.pth"
    )
    img_path = "./test/DJI_20250821061356_0001_S_frame_000123_20251009.jpg"
    image = Image.open(img_path).convert("RGB")
    dinov3_adapter_backbone = DINOv3STAs(**backbone_kwargs)
    imgsz = (640, 640)
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(imgsz),
            torchvision.transforms.ToTensor(),
        ]
    )
    inputs = transform(image).unsqueeze(0)

    plt.figure(figsize=(4, 6), dpi=640)
    plt.subplots_adjust(
        wspace=0.05, hspace=0.22, top=0.95, bottom=0.05, left=0.05, right=0.95
    )
    sem_feat_idx_list = [3, 5, 11]
    for idx, conv_layer in enumerate(
        [
            dinov3_adapter_backbone.convs[0],
            dinov3_adapter_backbone.convs[1],
            dinov3_adapter_backbone.convs[2],
        ]
    ):
        print(f"grad_cam_convs process {idx}")
        grad_cam = GradCAM(dinov3_adapter_backbone, conv_layer)
        grad_cam_convs_overlay, grad_cam_convs_heatmap = grad_cam.visualize(
            inputs, np.array(image), feature_idx=idx, feature_source="convs"
        )

        print(f"grad_cam_sem_feats process {idx}")
        grad_cam_sem_overlay, grad_cam_sem_heatmap = grad_cam.visualize(
            inputs,
            np.array(image),
            feature_idx=sem_feat_idx_list[idx],
            feature_source="sem_feats",
        )

        print(f"eigen_cam_convs process {idx}")
        eigen_cam = EigenCAM(dinov3_adapter_backbone, conv_layer)
        eigen_cam_convs_overlay, eigen_cam_convs_heatmap = eigen_cam.visualize(
            inputs, np.array(image), feature_idx=idx, feature_source="convs"
        )
        print(f"eigen_cam_sem_feats process {idx}")
        eigen_cam_sem_overlay, eigen_cam_sem_heatmap = eigen_cam.visualize(
            inputs,
            np.array(image),
            feature_idx=sem_feat_idx_list[idx],
            feature_source="sem_feats",
        )

        plt.subplot(6, 4, idx * 8 + 1)
        plt.imshow(grad_cam_convs_overlay)
        plt.title(f"c{idx+1}", fontsize=6)
        plt.axis("off")

        plt.subplot(6, 4, idx * 8 + 2)
        plt.imshow(eigen_cam_convs_overlay)
        plt.title(f"c{idx+1}_eg", fontsize=6)
        plt.axis("off")

        plt.subplot(6, 4, idx * 8 + 3)
        plt.imshow(grad_cam_convs_heatmap)
        plt.title(f"c{idx+1}_ht", fontsize=6)
        plt.axis("off")

        plt.subplot(6, 4, idx * 8 + 4)
        plt.imshow(eigen_cam_convs_heatmap)
        plt.title(f"c{idx+1}_eght", fontsize=6)
        plt.axis("off")

        plt.subplot(6, 4, idx * 8 + 5)
        plt.imshow(grad_cam_sem_overlay)
        plt.title(f"s{sem_feat_idx_list[idx]}", fontsize=6)
        plt.axis("off")

        plt.subplot(6, 4, idx * 8 + 6)
        plt.imshow(eigen_cam_sem_overlay)
        plt.title(f"s{sem_feat_idx_list[idx]}_eg", fontsize=6)
        plt.axis("off")

        plt.subplot(6, 4, idx * 8 + 7)
        plt.imshow(grad_cam_sem_heatmap)
        plt.title(f"s{sem_feat_idx_list[idx]}_ht", fontsize=6)
        plt.axis("off")

        plt.subplot(6, 4, idx * 8 + 8)
        plt.imshow(eigen_cam_sem_heatmap)
        plt.title(f"s{sem_feat_idx_list[idx]}_eght", fontsize=6)
        plt.axis("off")

    plt.savefig("./test/data/output/grad_cam/all.png", bbox_inches="tight")


if __name__ == "__main__":
    _test()
