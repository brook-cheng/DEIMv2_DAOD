"""
DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
"""

import torch
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as F
import random
import math
from PIL import Image

from .._misc import convert_to_tv_tensor
from ...core import register


@register()
class Mosaic(T.Transform):
    """
    Applies Mosaic augmentation to a batch of images. Combines four randomly selected images
    into a single composite image with randomized transformations.
    """

    def __init__(
        self,
        output_size=320,
        max_size=None,
        rotation_range=0,
        translation_range=(0.1, 0.1),
        scaling_range=(0.5, 1.5),
        probability=1.0,
        fill_value=114,
        use_cache=True,
        max_cached_images=50,
        random_pop=True,
    ) -> None:
        """
        Args:
            output_size (int): Target size for resizing individual images.
            rotation_range (float): Range of rotation in degrees for affine transformation.
            translation_range (tuple): Range of translation for affine transformation.
            scaling_range (tuple): Range of scaling factors for affine transformation.
            probability (float): Probability of applying the Mosaic augmentation.
            fill_value (int): Fill value for padding or affine transformations.
            use_cache (bool): Whether to use cache. Defaults to True.
            max_cached_images (int): The maximum length of the cache.
            random_pop (bool): Whether to randomly pop a result from the cache.
        """
        super().__init__()
        self.resize = T.Resize(size=output_size, max_size=max_size)
        self.probability = probability
        self.affine_transform = T.RandomAffine(
            degrees=rotation_range,
            translate=translation_range,
            scale=scaling_range,
            fill=fill_value,
        )
        # 保存范围供 OBB 仿射使用
        self.rotation_range = rotation_range
        self.translation_range = translation_range
        self.scaling_range = scaling_range
        self.fill_value = fill_value
        self.use_cache = use_cache
        self.mosaic_cache = []
        self.max_cached_images = max_cached_images
        self.random_pop = random_pop

    def load_samples_from_dataset(self, image, target, dataset):
        """Loads and resizes a set of images and their corresponding targets."""
        # Append the main image
        get_size_func = (
            F.get_size if hasattr(F, "get_size") else F.get_spatial_size
        )  # torchvision >=0.17 is get_size
        image, target = self.resize(image, target)
        resized_images, resized_targets = [image], [target]
        max_height, max_width = get_size_func(resized_images[0])

        # randomly select 3 images
        sample_indices = random.choices(range(len(dataset)), k=3)
        for idx in sample_indices:
            # image, target = dataset.load_item(idx)
            image, target = self.resize(dataset.load_item(idx))
            height, width = get_size_func(image)
            max_height, max_width = max(max_height, height), max(max_width, width)
            resized_images.append(image)
            resized_targets.append(target)

        return resized_images, resized_targets, max_height, max_width

    def load_samples_from_cache(self, image, target, cache):
        image, target = self.resize(image, target)
        cache.append(dict(img=image, labels=target))

        if len(cache) > self.max_cached_images:
            if self.random_pop:
                index = random.randint(0, len(cache) - 2)  # do not remove last image
            else:
                index = 0
            cache.pop(index)
        sample_indices = random.choices(range(len(cache)), k=3)
        mosaic_samples = [
            dict(img=cache[idx]["img"].copy(), labels=self._clone(cache[idx]["labels"]))
            for idx in sample_indices
        ]  # sample 3 images
        mosaic_samples = [
            dict(img=image.copy(), labels=self._clone(target))
        ] + mosaic_samples

        get_size_func = F.get_size if hasattr(F, "get_size") else F.get_spatial_size
        sizes = [get_size_func(mosaic_samples[idx]["img"]) for idx in range(4)]
        max_height = max(size[0] for size in sizes)
        max_width = max(size[1] for size in sizes)

        return mosaic_samples, max_height, max_width

    def create_mosaic_from_cache(self, mosaic_samples, max_height, max_width):
        placement_offsets = [
            [0, 0],
            [max_width, 0],
            [0, max_height],
            [max_width, max_height],
        ]
        merged_image = Image.new(
            mode=mosaic_samples[0]["img"].mode,
            size=(max_width * 2, max_height * 2),
            color=0,
        )
        offsets = torch.tensor(
            [[0, 0], [max_width, 0], [0, max_height], [max_width, max_height]]
        ).repeat(1, 2)
        if mosaic_samples[0]["labels"]["boxes"].shape[-1] == 5:
            # OBB (cx,cy,w,h,θ)：仅平移中心，w/h/θ 不变
            offsets = torch.tensor(
                [[0, 0], [max_width, 0], [0, max_height], [max_width, max_height]],
                dtype=torch.float32,
            )
            offsets = torch.cat([offsets, torch.zeros(4, 3)], dim=-1)  # [dx,dy,0,0,0]

        mosaic_target = []
        for i, sample in enumerate(mosaic_samples):
            img = sample["img"]
            target = sample["labels"]

            merged_image.paste(img, placement_offsets[i])
            target["boxes"] = target["boxes"] + offsets[i]
            mosaic_target.append(target)

        merged_target = {}
        for key in mosaic_target[0]:
            merged_target[key] = torch.cat([target[key] for target in mosaic_target])

        return merged_image, merged_target

    def create_mosaic_from_dataset(self, images, targets, max_height, max_width):
        """Creates a mosaic image by combining multiple images."""
        placement_offsets = [
            [0, 0],
            [max_width, 0],
            [0, max_height],
            [max_width, max_height],
        ]
        merged_image = Image.new(
            mode=images[0].mode, size=(max_width * 2, max_height * 2), color=0
        )
        for i, img in enumerate(images):
            merged_image.paste(img, placement_offsets[i])

        """Merges targets into a single target dictionary for the mosaic."""
        offsets = torch.tensor(
            [[0, 0], [max_width, 0], [0, max_height], [max_width, max_height]]
        ).repeat(1, 2)
        if targets[0]["boxes"].shape[-1] == 5:
            # OBB (cx,cy,w,h,θ)：仅平移中心，w/h/θ 不变
            offsets = torch.tensor(
                [[0, 0], [max_width, 0], [0, max_height], [max_width, max_height]],
                dtype=torch.float32,
            )
            offsets = torch.cat([offsets, torch.zeros(4, 3)], dim=-1)  # [dx,dy,0,0,0]
        merged_target = {}
        for key in targets[0]:
            if key == "boxes":
                values = [target[key] + offsets[i] for i, target in enumerate(targets)]
            else:
                values = [target[key] for target in targets]

            merged_target[key] = (
                torch.cat(values, dim=0)
                if isinstance(values[0], torch.Tensor)
                else values
            )

        return merged_image, merged_target

    def _affine_obb(self, img_pil, target):
        """对 OBB mosaic 图施加仿射变换（旋转+缩放+平移）。

        图像用逆矩阵 warp（PIL.AFFINE 约定），框用前向矩阵变换；
        二者共享同一个 (A,b) 采样，保证图像与框一致。
        画布尺寸保持不变；范围平凡时 no-op。
        """
        W, H = img_pil.size

        # 采样仿射参数
        phi = math.radians(random.uniform(-self.rotation_range, self.rotation_range))
        s = random.uniform(self.scaling_range[0], self.scaling_range[1])
        tx = random.uniform(-self.translation_range[0] * W, self.translation_range[0] * W)
        ty = random.uniform(-self.translation_range[1] * H, self.translation_range[1] * H)

        # 范围平凡时跳过
        if abs(phi) < 1e-9 and abs(s - 1.0) < 1e-9 and abs(tx) < 0.5 and abs(ty) < 0.5:
            return img_pil, target

        # 构造前向矩阵 p' = A @ (p - c) + c + t = A @ p + b
        cx, cy = W / 2.0, H / 2.0
        cos_p, sin_p = math.cos(phi), math.sin(phi)
        A = torch.tensor(
            [[s * cos_p, -s * sin_p],
             [s * sin_p,  s * cos_p]], dtype=torch.float32
        )
        c = torch.tensor([cx, cy], dtype=torch.float32)
        t = torch.tensor([tx, ty], dtype=torch.float32)
        b = c + t - A @ c
        mat_fwd = torch.cat([A, b[:, None]], dim=1)  # (2,3) 前向

        # 逆矩阵用于图像 warp
        A_inv = torch.inverse(A)
        b_inv = -A_inv @ b
        data = (
            A_inv[0, 0].item(), A_inv[0, 1].item(), b_inv[0].item(),
            A_inv[1, 0].item(), A_inv[1, 1].item(), b_inv[1].item(),
        )
        img_pil = img_pil.transform(
            (W, H), Image.AFFINE, data,
            resample=Image.BILINEAR, fillcolor=self.fill_value,
        )

        # 前向矩阵变换框 + 过滤越界/微小框
        boxes = target["boxes"]
        if len(boxes) > 0:
            from ...deim.obb_geometry import affine_obb_matrix

            boxes = affine_obb_matrix(boxes, mat_fwd.to(boxes.dtype))
            keep = (
                (boxes[:, 0] > 0) & (boxes[:, 0] < W)
                & (boxes[:, 1] > 0) & (boxes[:, 1] < H)
                & (boxes[:, 2] > 1) & (boxes[:, 3] > 1)
            )
            target["boxes"] = boxes[keep]
            target["labels"] = target["labels"][keep]

        return img_pil, target

    @staticmethod
    def _clone(tensor_dict):
        return {key: value.clone() for (key, value) in tensor_dict.items()}

    def forward(self, *inputs):
        """
        Args:
            inputs (tuple): Input tuple containing (image, target, dataset).

        Returns:
            tuple: Augmented (image, target, dataset).
        """
        if len(inputs) == 1:
            inputs = inputs[0]
        image, target, dataset = inputs

        # Skip mosaic augmentation with probability 1 - self.probability
        if self.probability < 1.0 and random.random() > self.probability:
            return image, target, dataset

        # Prepare mosaic components
        if self.use_cache:
            mosaic_samples, max_height, max_width = self.load_samples_from_cache(
                image, target, self.mosaic_cache
            )
            mosaic_image, mosaic_target = self.create_mosaic_from_cache(
                mosaic_samples, max_height, max_width
            )
        else:
            resized_images, resized_targets, max_height, max_width = (
                self.load_samples_from_dataset(image, target, dataset)
            )
            mosaic_image, mosaic_target = self.create_mosaic_from_dataset(
                resized_images, resized_targets, max_height, max_width
            )

        # Clamp boxes and convert target formats
        is_obb = "boxes" in mosaic_target and mosaic_target["boxes"].shape[-1] == 5
        if "boxes" in mosaic_target and not is_obb:
            mosaic_target["boxes"] = convert_to_tv_tensor(
                mosaic_target["boxes"],
                "boxes",
                box_format="xyxy",
                spatial_size=mosaic_image.size[::-1],
            )
        if "masks" in mosaic_target:
            mosaic_target["masks"] = convert_to_tv_tensor(
                mosaic_target["masks"], "masks"
            )

        # Apply affine transformations
        if not is_obb:
            mosaic_image, mosaic_target = self.affine_transform(
                mosaic_image, mosaic_target
            )
        elif is_obb:
            mosaic_image, mosaic_target = self._affine_obb(
                mosaic_image, mosaic_target
            )

        return mosaic_image, mosaic_target, dataset
