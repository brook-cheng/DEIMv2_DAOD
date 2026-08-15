import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision
import cv2


def cosine_similarity_map_featsz(feats, img, xy: list[int]):
    w, h = img.shape[1], img.shape[0]
    patch_h = feats.shape[1]
    patch_w = feats.shape[0]
    patch_size = w // patch_w

    # Coordinate to patch token location
    xy[0] = xy[0] // patch_size
    xy[1] = xy[1] // patch_size

    # The patch index
    feats = feats.reshape(patch_h * patch_w, -1)
    idx = xy[1] * patch_w + xy[0]
    similarities = F.cosine_similarity(feats[idx].unsqueeze(0), feats)

    return similarities.view(patch_h, patch_w).cpu(), xy


def cosine_similarity_map_imgsz(feats, img, xy: list[int]):
    w, h = img.shape[1], img.shape[0]
    # The patch index
    feats = (
        torch.nn.functional.interpolate(
            feats.permute(2, 0, 1).unsqueeze(0),
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        )
        .squeeze(0)
        .permute(1, 2, 0)
    )
    feats = feats.reshape(w * h, -1)
    idx = xy[1] * w + xy[0]
    similarities = F.cosine_similarity(feats[idx].unsqueeze(0), feats)

    return similarities.view(h, w).cpu(), xy


def plot_cosine_similarity(
    feats,
    img,
    coords: list[tuple[int, int]],
    patch_size: int = 16,
    decrease_input: int = 1,
    merge: bool = True,
    alpha: float = 0.6,
):
    img_in = img.copy()
    if decrease_input > 1:
        w, h = img_in.size
        w = int(w / decrease_input)
        h = int(h / decrease_input)
        img_in = img_in.resize((w, h))

    n = len(coords)
    feats_num = len(feats)
    fig, axes = plt.subplots(
        feats_num, n + 1, figsize=((n + 1) * 2, feats_num * 2), dpi=640
    )

    for j, feat in enumerate(feats):
        # Original image
        axes[j, 0].imshow(img_in)
        axes[j, 0].set_title("Original image")
        axes[j, 0].axis("off")

        # Heatmaps for each coordinate
        for i, coord in enumerate(coords):
            coord_in = [int(coord[0] / decrease_input), int(coord[1] / decrease_input)]
            sim_ten, (px, py) = cosine_similarity_map_imgsz(feats[j], img_in, coord_in)
            sim_np = sim_ten.numpy()
            sim = (sim_np - sim_np.min()) / (sim_np.max() - sim_np.min())

            if merge:
                heatmap_colored = cv2.applyColorMap(
                    np.uint8(255 * sim), cv2.COLORMAP_JET
                )
                heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
                overlay = cv2.addWeighted(img_in, 1 - alpha, heatmap_colored, alpha, 0)
                sim = overlay
            axes[j, i + 1].imshow(sim, cmap="viridis")
            axes[j, i + 1].scatter(
                px,
                py,
                color="red",
                marker="+",
                s=100,
                linewidths=1,
            )

            axes[j, i + 1].set_title(f"Patch:{coord}")
            axes[j, i + 1].axis("off")

    plt.tight_layout()
    plt.savefig("./test/data/output/cosine_similarity.png")


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
    from tools.analysis.grad_cam import GradCAM

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
        "./outputs/deimv2_dinov3_x_custom/best_stg2_fintune_1110_e186_mAP68.pth"
    )
    img_path = "./test/DJI_20250821061356_0001_S_frame_000123_20251009.jpg"
    image = Image.open(img_path).convert("RGB")
    dinov3_adapter_backbone = DINOv3STAs(**backbone_kwargs)

    orginal_observer_pts = [
        (310, 350),
        (383, 418),
        (624, 277),
        (342, 286),
    ]
    imgsz = (640, 640)
    img_rs = cv2.resize(np.array(image), imgsz)
    img_width, img_height = image.size
    dst_observer_pts = [
        (int(x / img_width * imgsz[1]), int(y / img_height * imgsz[0]))
        for x, y in orginal_observer_pts
    ]
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(imgsz),
            torchvision.transforms.ToTensor(),
        ]
    )
    inputs = transform(image).unsqueeze(0)

    with torch.no_grad():
        tmp_feats = dinov3_adapter_backbone.dinov3.forward_features(inputs)[
            "x_norm_patchtokens"
        ]
        # whxc
        tmp_feats = dinov3_adapter_backbone.dinov3.head(tmp_feats[0])
        # wxhxc
        tmp_feats = tmp_feats.reshape(80, 80, -1)
        # bxcxwxh
        c2, c3, c4 = dinov3_adapter_backbone(inputs)
        # wxhxc
        c2 = c2[0].permute(1, 2, 0)
        c3 = c3[0].permute(1, 2, 0)
        c4 = c4[0].permute(1, 2, 0)
        grad_cam = GradCAM(dinov3_adapter_backbone, dinov3_adapter_backbone)
        # bxcxwxh
        sem_feat3 = grad_cam._get_sem_feats(inputs, 3)
        sem_feat5 = grad_cam._get_sem_feats(inputs, 5)
        sem_feat11 = grad_cam._get_sem_feats(inputs, 11)
        # wxhxc
        sem_feat3 = sem_feat3[0][0].permute(1, 2, 0)
        sem_feat5 = sem_feat5[0][0].permute(1, 2, 0)
        sem_feat11 = sem_feat11[0][0].permute(1, 2, 0)

        plot_cosine_similarity(
            [c2, sem_feat3, c3, sem_feat5, c4, sem_feat11, tmp_feats],
            img_rs,
            dst_observer_pts,
        )


if __name__ == "__main__":
    _test()
