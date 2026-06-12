"""OBB-aware data augmentation transforms."""

import random, torch, torch.nn as nn
import torch.nn.functional as F
from ...core import register
import torchvision.transforms.v2.functional as TF


# FIXME: 图片没有Flip
@register()
class OBBFlip(nn.Module):
    def forward(self, sample):
        img, tgt, ds = sample
        b = tgt["boxes"]
        _, h, w = img.shape if hasattr(img, "shape") else (img.mode, *img.size[::-1])
        b[:, 0] = w - b[:, 0]
        b[:, 4] = (torch.pi - b[:, 4]) % torch.pi
        return img, tgt, ds


@register()
class OBBZoomOut(nn.Module):
    """Pad image and shift box coordinates accordingly."""

    def __init__(self, fill=0, pad_level=0.5):
        super().__init__()
        self.fill = fill
        self.pad_level = pad_level

    def forward(self, sample):
        img, tgt, ds = sample

        c, h, w = img.shape if hasattr(img, "shape") else (img.mode, *img.size[::-1])
        pad_top = int(random.randint(0, int(h * self.pad_level)))
        pad_bot = int(random.randint(0, int(h * self.pad_level)))
        pad_left = int(random.randint(0, int(w * self.pad_level)))
        pad_right = int(random.randint(0, int(w * self.pad_level)))
        # use torchvision functional pad (handles both PIL and Tensor)
        img = TF.pad(img, [pad_left, pad_top, pad_right, pad_bot], fill=self.fill)
        # shift boxes
        new_h, new_w = h + pad_top + pad_bot, w + pad_left + pad_right
        if len(tgt["boxes"]) > 0:
            tgt["boxes"][:, 0] = (tgt["boxes"][:, 0] * w + pad_left) / new_w
            tgt["boxes"][:, 1] = (tgt["boxes"][:, 1] * h + pad_top) / new_h
            tgt["boxes"][:, 2] *= w / new_w
            tgt["boxes"][:, 3] *= h / new_h
        return img, tgt, ds


@register()
class OBBResize(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, sample):
        img, tgt, ds = sample
        _, h, w = img.shape if hasattr(img, "shape") else (3, *img.size[::-1])
        img = TF.resize(img, self.size)
        s = self.size
        b = tgt["boxes"]
        sx, sy = s[0] / w, s[1] / h
        b[:, 0] *= sx
        b[:, 1] *= sy
        b[:, 2] *= sx
        b[:, 3] *= sy
        return img, tgt, ds


@register()
class OBBConvertBoxes(nn.Module):
    def __init__(self, normalize=True, img_size=(640, 640)):
        super().__init__()
        self.n = normalize
        self.iw, self.ih = img_size

    def forward(self, sample):
        if self.n:
            img, tgt, ds = sample
            b = tgt["boxes"]
            b[:, 0] /= self.iw
            b[:, 1] /= self.ih
            b[:, 2] /= self.iw
            b[:, 3] /= self.ih
        return sample


@register()
class OBBSanitize(nn.Module):
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


# FIXME: 这里并没有实现真正的操作，需要在mosaic.py实现
@register()
class OBBMosaic(nn.Module):
    """Mosaic offset for OBB. Shifts cx,cy by grid offset."""

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


# FIXME: 这里使用了hbb iou的计算方式，存在问题，虽然影响可能没那么大
@register()
class OBBIoUCrop(nn.Module):
    """Random IoU-based crop for OBB. Samples multiple crop regions,
    picks the one with highest average IoU to GT boxes, then crops."""

    def __init__(self, p=1.0, scale=(0.3, 1.0), ratio=(0.5, 2.0), trials=40):
        super().__init__()
        self.p = p
        self.scale = scale
        self.ratio = ratio
        self.trials = trials

    def _hbb_iou(self, boxes, crop_x1, crop_y1, crop_x2, crop_y2):
        """Average HBB IoU between boxes and crop region."""
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
        _, h, w = img.shape if hasattr(img, "shape") else (img.mode, *img.size[::-1])
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
            iou = self._hbb_iou(boxes, x1 / w, y1 / h, x2 / w, y2 / h)
            if iou > best_iou:
                best_iou, best_crop = iou, (x1, y1, x2, y2)

        if best_crop is None:
            return sample

        x1, y1, x2, y2 = best_crop

        if hasattr(img, "shape"):
            img = img[:, y1:y2, x1:x2]
        else:
            img = TF.crop(img, y1, x1, y2 - y1, x2 - x1)

        if len(boxes) > 0:
            boxes[:, 0] = (boxes[:, 0] * w - x1) / (x2 - x1)
            boxes[:, 1] = (boxes[:, 1] * h - y1) / (y2 - y1)
            boxes[:, 2] *= w / (x2 - x1)
            boxes[:, 3] *= h / (y2 - y1)
            keep = (boxes[:, 2] > 0.005) & (boxes[:, 3] > 0.005)
            boxes = boxes[keep]
            tgt["boxes"] = boxes
            tgt["labels"] = tgt["labels"][keep]

        return img, tgt, ds
