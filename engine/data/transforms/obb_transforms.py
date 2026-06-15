"""OBB-aware data augmentation transforms.

坐标约定：OBB 格式 (cx, cy, w, h, θ)，θ ∈ [0, π) 弧度。
OBBConvertBoxes 之前所有变换在**像素坐标**下工作；归一化由 OBBConvertBoxes 完成。
"""

import random
import torch
import torch.nn as nn
from ...core import register
import torchvision.transforms.v2.functional as TF


@register()
class OBBFlip(nn.Module):
    """水平翻转图像和 OBB 框（像素坐标）。"""

    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def forward(self, sample):
        img, tgt, ds = sample
        if random.random() > self.p:
            return img, tgt, ds

        # 获取图像尺寸（像素）
        if hasattr(img, "shape"):
            _, h, w = img.shape  # CHW tensor
        else:
            w, h = img.size  # PIL Image

        # 水平翻转图像
        img = TF.hflip(img)

        # 翻转框（像素坐标下 cx' = W - cx，θ' = (π - θ) % π）
        b = tgt["boxes"]
        b[:, 0] = w - b[:, 0]
        b[:, 4] = (torch.pi - b[:, 4]) % torch.pi

        return img, tgt, ds


@register()
class OBBZoomOut(nn.Module):
    """随机四边填充（纯平移），在像素坐标下工作。"""

    def __init__(self, fill=0, pad_level=0.5):
        super().__init__()
        self.fill = fill
        self.pad_level = pad_level

    def forward(self, sample):
        img, tgt, ds = sample

        if hasattr(img, "shape"):
            c, h, w = img.shape
        else:
            w, h = img.size

        pad_top = int(random.randint(0, int(h * self.pad_level)))
        pad_bot = int(random.randint(0, int(h * self.pad_level)))
        pad_left = int(random.randint(0, int(w * self.pad_level)))
        pad_right = int(random.randint(0, int(w * self.pad_level)))

        img = TF.pad(img, [pad_left, pad_top, pad_right, pad_bot], fill=self.fill)

        # 纯平移，仅改中心，w/h/θ 不变
        if len(tgt["boxes"]) > 0:
            from ...deim.obb_geometry import affine_obb

            tgt["boxes"] = affine_obb(
                tgt["boxes"], sx=1.0, sy=1.0, tx=pad_left, ty=pad_top
            )

        return img, tgt, ds


@register()
class OBBResize(nn.Module):
    """缩放图像和 OBB 框（像素坐标）。size 格式为 [H, W]（与 torchvision 一致）。"""

    def __init__(self, size):
        super().__init__()
        self.size = size  # [H, W]

    def forward(self, sample):
        img, tgt, ds = sample

        if hasattr(img, "shape"):
            _, h, w = img.shape  # CHW
        else:
            w, h = img.size  # PIL

        img = TF.resize(img, self.size)

        H_out, W_out = self.size[0], self.size[1]
        sx = W_out / w
        sy = H_out / h

        if len(tgt["boxes"]) > 0:
            from ...deim.obb_geometry import affine_obb

            tgt["boxes"] = affine_obb(tgt["boxes"], sx=sx, sy=sy, tx=0.0, ty=0.0)

        return img, tgt, ds


@register()
class OBBConvertBoxes(nn.Module):
    """将 OBB 框从像素坐标归一化到 [0,1]（按图像实际尺寸，θ 不变）。"""

    def __init__(self, normalize=True, img_size=(640, 640)):
        super().__init__()
        self.n = normalize
        self.iw, self.ih = img_size  # 保留用于向后兼容，优先用实际尺寸

    def forward(self, sample):
        if self.n:
            img, tgt, ds = sample

            if hasattr(img, "shape"):
                _, H, W = img.shape
            else:
                W, H = img.size

            b = tgt["boxes"]
            b[:, 0] /= W
            b[:, 1] /= H
            b[:, 2] /= W
            b[:, 3] /= H
            # θ 不变，保持在 [0, π)
        return sample


@register()
class OBBSanitize(nn.Module):
    """过滤 w 或 h 过小的框。"""

    def __init__(self, min_size=1):
        super().__init__()
        self.ms = min_size

    def forward(self, sample):
        img, tgt, ds = sample
        b = tgt["boxes"]
        k = (b[:, 2] > self.ms) & (b[:, 3] > self.ms)
        tgt["boxes"] = b[k]
        tgt["labels"] = tgt["labels"][k]
        return img, tgt, ds


@register()
class OBBMosaic(nn.Module):
    """Mosaic 拼图对 OBB 的偏移（仅平移中心，w/h/θ 不变）。"""

    def __init__(self, offset_x=0, offset_y=0):
        super().__init__()
        self.ox = offset_x
        self.oy = offset_y

    def forward(self, sample):
        img, tgt, ds = sample
        b = tgt["boxes"]
        b[:, 0] += self.ox
        b[:, 1] += self.oy
        return img, tgt, ds


@register()
class OBBIoUCrop(nn.Module):
    """随机 IoU 引导裁剪。使用 HBB 轴对齐 IoU 近似选择裁剪区域（标注为近似），
    裁剪为纯平移操作，在像素坐标下工作。"""

    def __init__(self, p=1.0, scale=(0.3, 1.0), ratio=(0.5, 2.0), trials=40):
        super().__init__()
        self.p = p
        self.scale = scale
        self.ratio = ratio
        self.trials = trials

    def _hbb_iou(self, boxes, crop_x1, crop_y1, crop_x2, crop_y2):
        """近似 HBB IoU：用于筛选裁剪区域（不精确但足够用于"选哪个裁剪区"）。"""
        if len(boxes) == 0:
            return 0.0
        inter_x1 = torch.max(boxes[:, 0] - boxes[:, 2] / 2, torch.tensor(crop_x1))
        inter_y1 = torch.max(boxes[:, 1] - boxes[:, 3] / 2, torch.tensor(crop_y1))
        inter_x2 = torch.min(boxes[:, 0] + boxes[:, 2] / 2, torch.tensor(crop_x2))
        inter_y2 = torch.min(boxes[:, 1] + boxes[:, 3] / 2, torch.tensor(crop_y2))
        iw = (inter_x2 - inter_x1).clamp(min=0)
        ih = (inter_y2 - inter_y1).clamp(min=0)
        inter = iw * ih
        area = boxes[:, 2] * boxes[:, 3]
        union = area + (crop_x2 - crop_x1) * (crop_y2 - crop_y1) - inter
        return (inter / (union + 1e-8)).mean().item()

    def forward(self, sample):
        if random.random() > self.p:
            return sample
        img, tgt, ds = sample

        if hasattr(img, "shape"):
            _, h, w = img.shape
        else:
            w, h = img.size

        boxes = tgt["boxes"]

        best_iou, best_crop = -1, None
        for _ in range(self.trials):
            crop_w = int(random.uniform(self.scale[0], self.scale[1]) * w)
            crop_h = int(crop_w / random.uniform(*self.ratio))
            if crop_h > h:
                crop_h = int(crop_w * random.uniform(*self.ratio))
            if crop_h > h:
                continue
            x1 = random.randint(0, max(0, w - crop_w))
            y1 = random.randint(0, max(0, h - crop_h))
            x2, y2 = x1 + crop_w, y1 + crop_h
            # HBB IoU 近似用于选择裁剪区（标注为近似）
            iou = self._hbb_iou(boxes, x1, y1, x2, y2)
            if iou > best_iou:
                best_iou, best_crop = iou, (x1, y1, x2, y2)

        if best_crop is None:
            return sample

        x1, y1, x2, y2 = best_crop

        # 裁剪图像
        if hasattr(img, "shape"):
            img = img[:, y1:y2, x1:x2]
        else:
            img = TF.crop(img, y1, x1, y2 - y1, x2 - x1)

        # 纯平移：框中心减去裁剪起点，w/h/θ 不变
        if len(boxes) > 0:
            from ...deim.obb_geometry import affine_obb

            tgt["boxes"] = affine_obb(
                boxes, sx=1.0, sy=1.0, tx=-float(x1), ty=-float(y1)
            )
            # 过滤中心落在裁剪区外的框
            b = tgt["boxes"]
            keep = (
                (b[:, 0] > 0)
                & (b[:, 0] < (x2 - x1))
                & (b[:, 1] > 0)
                & (b[:, 1] < (y2 - y1))
                & (b[:, 2] > 1)
                & (b[:, 3] > 1)
            )
            tgt["boxes"] = b[keep]
            tgt["labels"] = tgt["labels"][keep]

        return img, tgt, ds
