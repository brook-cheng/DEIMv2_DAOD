import os
import torch
import torchvision

from collections import OrderedDict
import numpy as np
from PIL import Image
from ._dataset import DetDataset
from ...core import register

__all__ = ["DotaDataset"]


@register()
class DotaDataset(DetDataset):
    __inject__ = ["transforms"]

    def __init__(
        self,
        img_folder,
        ann_folder,
        classes_file,
        transforms,
        format,
        cache_images: str = "none",
        cache_ram: int = 0,
    ):
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
        self.format = format
        if self.format != "DOTA" and self.format != "YOLO-OBB":
            raise ValueError(
                f"Unsupported format: {self.format}, must be DOTA or YOLO-OBB"
            )
        self.cache_images = cache_images
        self.cache_ram = cache_ram
        # 一次性构建，替代原 property
        self._img_ann_dict = self._build_img_ann_dict()
        self._image_names = list(self._img_ann_dict.keys())
        self._cat2id = {name: id for id, name in enumerate(self.label_names)}
        self._id2cat = {id: name for id, name in enumerate(self.label_names)}
        # 一次性解析标注（存原始行）
        self._ann_cache: dict[str, list[str]] = {}
        self._preload_annotations()
        # 图片缓存
        self._img_cache = OrderedDict()
        self._npy_dir_path = os.path.join(img_folder + "_npy")

    def __len__(self):
        return len(self._img_ann_dict)

    def _build_img_ann_dict(self) -> dict:
        image_files = [
            f
            for f in os.listdir(self.img_folder)
            if str.lower(os.path.splitext(f)[-1]) in self.img_extensions
        ]
        ann_files = set(os.listdir(self.ann_folder))
        result = {
            img_file: img_file.replace(os.path.splitext(img_file)[-1], ".txt")
            for img_file in image_files
            if img_file.replace(os.path.splitext(img_file)[-1], ".txt") in ann_files
        }
        if len(result) == 0:
            raise ValueError(
                f"Valid annotation is empty, in {self.img_folder} and {self.ann_folder}"
            )
        return result

    def _preload_annotations(self) -> None:
        import time
        from tqdm import tqdm

        n = len(self._img_ann_dict)
        t0 = time.time()
        pbar = tqdm(
            self._img_ann_dict.items(), total=n,
            desc="[DotaDataset] Annotations", unit="file",
        )
        for img_name, ann_name in pbar:
            stem = os.path.splitext(img_name)[0]
            ann_path = os.path.join(self.ann_folder, ann_name)
            with open(ann_path, "r") as f:
                self._ann_cache[stem] = f.readlines()
        elapsed = time.time() - t0
        print(
            f"[DotaDataset] Annotations loaded: {n} files in {elapsed:.1f}s",
            flush=True,
        )

    def _npy_dir(self) -> str:
        return self._npy_dir_path

    def _convert_one(self, task) -> None:
        stem, img_path, npy_path = task
        if os.path.exists(npy_path):
            return
        try:
            img = Image.open(img_path).convert("RGB")
            np.save(npy_path, np.array(img), allow_pickle=False)
        except Exception as e:
            print(f"[DotaDataset] Failed to cache {img_path}: {e}")

    def precache_images(self, num_workers: int = 8) -> None:
        import time
        from multiprocessing.pool import ThreadPool
        from tqdm import tqdm

        os.makedirs(self._npy_dir(), exist_ok=True)
        tasks = [
            (
                os.path.splitext(img_name)[0],
                os.path.join(self.img_folder, img_name),
                os.path.join(self._npy_dir(), f"{os.path.splitext(img_name)[0]}.npy"),
            )
            for img_name in self._img_ann_dict.keys()
        ]
        total = len(tasks)
        skip = sum(1 for _, _, npy in tasks if os.path.exists(npy))
        conv = total - skip
        if conv == 0:
            print(
                f"[DotaDataset] Disk cache: {skip}/{total} already cached, skipping",
                flush=True,
            )
            return

        t0 = time.time()
        pbar = tqdm(
            total=conv, desc="[DotaDataset] Disk cache", unit="img",
        )
        with ThreadPool(num_workers) as pool:
            for _ in pool.imap_unordered(self._convert_one, tasks):
                pbar.update(1)
        pbar.close()
        elapsed = time.time() - t0
        print(
            f"[DotaDataset] Disk cache complete: {conv} converted, "
            f"{skip} cached, {conv/max(elapsed,1e-6):.0f} img/s, {elapsed:.1f}s",
            flush=True,
        )

    def _load_image(self, stem, image_path):
        if self.cache_ram > 0 and stem in self._img_cache:
            self._img_cache.move_to_end(stem)
            return self._img_cache[stem].copy()

        if self.cache_images == "disk":
            npy_path = os.path.join(self._npy_dir(), f"{stem}.npy")
            try:
                img = Image.fromarray(np.load(npy_path))
            except Exception:
                if os.path.exists(npy_path):
                    os.remove(npy_path)
                img = Image.open(image_path).convert("RGB")
        else:
            img = Image.open(image_path).convert("RGB")

        if self.cache_ram > 0:
            self._img_cache[stem] = img
            if len(self._img_cache) > self.cache_ram:
                self._img_cache.popitem(last=False)
        return img.copy()

    def _parse_from_lines(self, ann_lines, w, h):
        cat_id_dict = self._cat2id
        boxes, labels = [], []
        if self.format == "YOLO-OBB":
            for line in ann_lines:
                parts = line.strip().split()
                if len(parts) < 9:
                    continue
                cls_id = (
                    int(parts[0])
                    if parts[0].lstrip("-").isdigit()
                    else cat_id_dict.get(parts[0], 0)
                )
                pts = [float(p) for p in parts[1:9]]
                xyxyxyxy = torch.tensor(pts).reshape(4, 2)
                xyxyxyxy[:, 0] *= w
                xyxyxyxy[:, 1] *= h
                xywhr = self.xyxyxyxy_to_xywhr(xyxyxyxy)
                boxes.append(xywhr)
                labels.append(cls_id)
        elif self.format == "DOTA":
            for line in ann_lines:
                parts = line.strip().split()
                if len(parts) < 9:
                    continue
                pts = [float(p) for p in parts[:8]]
                xyxyxyxy = torch.tensor(pts).reshape(4, 2)
                xywhr = self.xyxyxyxy_to_xywhr(xyxyxyxy)
                boxes.append(xywhr)
                cat_parts = parts[8 : len(parts) - 1]
                cat = " ".join(cat_parts)
                labels.append(cat_id_dict[cat])
        else:
            raise ValueError(
                f"Unsupported format: {self.format}, must be DOTA or YOLO-OBB"
            )
        return (
            torch.stack(boxes) if boxes else torch.zeros(0, 5),
            torch.tensor(labels) if labels else torch.zeros(0, dtype=torch.long),
        )

    def load_item(self, index):
        image_absolute_path = os.path.join(self.img_folder, self._image_names[index])
        stem = os.path.splitext(self._image_names[index])[0]

        img = self._load_image(stem, image_absolute_path)
        w, h = img.size

        ann_lines = self._ann_cache[stem]
        boxes, labels = self._parse_from_lines(ann_lines, w, h)

        target = {
            "boxes": boxes,
            "labels": labels,
            "orig_size": torch.tensor([w, h]),
        }
        return img, target
