import os
import torch
import torchvision

from PIL import Image
from ._dataset import DetDataset
from ...core import register

__all__ = ["DotaDataset"]


@register()
class DotaDataset(DetDataset):
    __inject__ = ["transforms"]

    def __init__(self, img_folder, ann_folder, classes_file, transforms):
        super(DotaDataset, self).__init__()
        from ...deim.obb_geometry import xyxyxyxy_to_xywhr

        self.xyxyxyxy_to_xywhr = xyxyxyxy_to_xywhr
        self.transforms = transforms
        self.img_folder = img_folder
        self.ann_folder = ann_folder
        self.classes_file = classes_file
        self.img_extensions = [".jpg", ".jpeg", ".png", ".bmp"]
        with open(classes_file, "r") as f:
            label_names = [line.strip() for line in f.readlines()]
        self.label_names = label_names
        self._img_ann_dict: dict[str, str] = None

    def __len__(self):
        self.img_ann_dict
        return len(self._img_ann_dict)

    @property
    def img_ann_dict(self) -> dict[str, str]:
        image_files = [
            f
            for f in os.listdir(self.img_folder)
            if str.lower(os.path.splitext(f)[-1]) in self.img_extensions
        ]
        ann_files = [f for f in os.listdir(self.ann_folder) if f.endswith(".txt")]

        self._img_ann_dict = {
            img_file: img_file.replace(os.path.splitext(img_file)[-1], ".txt")
            for img_file in image_files
            if img_file.replace(os.path.splitext(img_file)[-1], ".txt") in ann_files
        }
        if len(self._img_ann_dict) == 0:
            raise ValueError(
                f"Valid annotation is empty, in {self.img_folder} and {self.ann_folder}"
            )
        return self._img_ann_dict

    @property
    def cat2id(self):
        return {name: id for id, name in enumerate(self.label_names)}

    @property
    def id2cat(self):
        return {id: name for id, name in enumerate(self.label_names)}

    def load_item(self, index):
        self.img_ann_dict
        image_names = list(self._img_ann_dict.keys())
        image_absolute_path = os.path.join(self.img_folder, image_names[index])
        ann_absolute_path = os.path.join(
            self.ann_folder,
            self._img_ann_dict[image_names[index]],
        )
        img = Image.open(image_absolute_path)
        w, h = img.size
        cat_id_dict = self.cat2id
        boxes, labels = [], []
        with open(ann_absolute_path, "r") as f:
            ann_lines = f.readlines()
        for line in ann_lines:
            parts = line.strip().split()
            pts = [float(p) for p in parts[:8]]
            xyxyxyxy = torch.tensor(pts).reshape(4, 2)
            xywhr = self.xyxyxyxy_to_xywhr(xyxyxyxy)
            boxes.append(xywhr)
            cat_parts = parts[8 : len(parts) - 1]
            cat = " ".join(cat_parts)
            labels.append(cat_id_dict[cat])

        target = {
            "boxes": torch.stack(boxes),
            "labels": torch.tensor(labels),
            "orig_size": torch.tensor([w, h]),
        }
        return img, target
