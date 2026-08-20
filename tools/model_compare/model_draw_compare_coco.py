"""
根据 COCO 格式的标注文件，在同一张图上绘制不同模型的预测结果和真实标注，使用不同颜色区分，并添加图例说明。
"""

import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from PIL import Image, ImageDraw, ImageFont
from typing import List, Optional, Dict, Union
import matplotlib.pyplot as plt
import numpy as np
import json


from tools.model_compare.coco_utils import (
    deimv2_outputs_to_coco_annotations,
    show_coco_annotations_on_image,
    ultralytics_val_to_coco,
)


def draw_coco_annotations(
    img_dir: str,
    gt_coco_json: Union[str, Dict],
    coco_jsons: List[Union[str, Dict]],
    output_path: Optional[str] = None,
    score_threshold: float = 0.5,
    model_names: Optional[List[str]] = None,
):
    """
    在同一张图上绘制不同 COCO 标注文件中的信息，使用不同颜色区分

    Args:
        img_dir: 需要绘制的图片所在目录路径
        gt_coco_json: 真实标注信息的 COCO JSON 文件路径或字典数据
        coco_jsons: 所有模型预测结果的标注信息列表（COCO JSON 文件路径或字典数据）
        output_path: 输出图像路径或目录
                    - 如果为 None，返回单个 PIL Image 对象（仅当只处理一张图时）
                    - 如果指定为文件路径且只处理一张图，保存到该路径
                    - 如果指定为目录路径或处理多张图，保存到 output_path/{filename}_comparison.jpg
        score_threshold: 置信度阈值
        model_names: 模型名称列表，用于图例显示。如果为 None，则使用 "Model_1", "Model_2" 等

    Returns:
        如果只处理一张图且 output_path 为 None，返回 PIL Image 对象
        否则返回保存的文件路径列表
    """

    if isinstance(gt_coco_json, str):
        with open(gt_coco_json, "r", encoding="utf-8") as f:
            gt_data = json.load(f)
    else:
        gt_data = gt_coco_json

    if (
        not isinstance(gt_data, dict)
        or "images" not in gt_data
        or "annotations" not in gt_data
    ):
        raise ValueError(
            "gt_coco_json must be a valid COCO format dictionary or path to COCO JSON file"
        )

    images_info = gt_data["images"]

    if not images_info:
        raise ValueError("No images found in gt_coco_json")

    num_models = len(coco_jsons)

    if model_names is None:
        model_names = [f"Model_{i+1}" for i in range(num_models)]
    elif len(model_names) != num_models:
        print(
            f"Warning: model_names length ({len(model_names)}) doesn't match coco_jsons length ({num_models})"
        )
        model_names = [f"Model_{i+1}" for i in range(num_models)]

    colors = plt.cm.rainbow(np.linspace(0, 1, max(num_models + 1, 1)))

    gt_color = tuple([int(c * 255) for c in colors[0][:3]])
    model_colors = [
        tuple([int(c * 255) for c in colors[i + 1][:3]]) for i in range(num_models)
    ]

    print(f"Found {len(images_info)} images in ground truth annotations")
    print(f"Processing {len(images_info)} images with {num_models} models...")

    saved_paths = []

    for img_idx, img_info in enumerate(images_info):
        img_id = img_info["id"]
        img_filename = img_info["file_name"]
        img_path = os.path.join(img_dir, img_filename)

        if not os.path.exists(img_path):
            print(f"Warning: Image file not found: {img_path}, skipping")
            continue

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error reading image {img_path}: {e}, skipping")
            continue

        img_width, img_height = image.size

        masks_data = []

        gt_mask = show_coco_annotations_on_image(
            img_dir=img_dir,
            coco_annotations=gt_data,
            img_idxes=[img_idx],
            score_threshold=score_threshold,
            mask_flag=True,
        )

        if gt_mask is not None:
            masks_data.append(
                {
                    "name": "Ground Truth",
                    "mask": gt_mask,
                    "color": gt_color,
                }
            )

        print(
            f"[{img_idx+1}/{len(images_info)}] Generating masks for {num_models} models on: {img_filename}"
        )
        for model_idx, coco_json in enumerate(coco_jsons):
            if isinstance(coco_json, str):
                with open(coco_json, "r", encoding="utf-8") as f:
                    pred_data = json.load(f)
            else:
                pred_data = coco_json

            if (
                not isinstance(pred_data, dict)
                or "images" not in pred_data
                or "annotations" not in pred_data
            ):
                print(f"Warning: Invalid COCO format in model {model_idx+1}, skipping")
                continue

            pred_mask = show_coco_annotations_on_image(
                img_dir=img_dir,
                coco_annotations=pred_data,
                img_idxes=[img_idx],
                score_threshold=score_threshold,
                mask_flag=True,
            )

            if pred_mask is not None:
                masks_data.append(
                    {
                        "name": model_names[model_idx],
                        "mask": pred_mask,
                        "color": model_colors[model_idx],
                    }
                )

        if not masks_data:
            print(
                f"Warning: No valid masks generated for image {img_filename}, skipping"
            )
            continue

        merged_image = image

        for mask_info in masks_data:
            mask = mask_info["mask"]
            color = mask_info["color"]
            name = mask_info["name"]

            if mask.size != (img_width, img_height):
                mask = mask.resize((img_width, img_height), Image.NEAREST)

            mask_array = np.array(mask)

            merged_array = np.array(merged_image)

            unique_categories = np.unique(mask_array)
            for cat_id in unique_categories:
                if cat_id == 0:
                    continue

                category_mask = mask_array == cat_id

                merged_array[category_mask] = color

            merged_image = Image.fromarray(merged_array, mode="RGB")

        draw = ImageDraw.Draw(merged_image)

        try:
            legend_font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 50
            )
        except:
            legend_font = ImageFont.load_default()

        legend_x = 10
        legend_y = 10
        box_size = 80
        spacing = box_size

        for mask_info in masks_data:
            color = mask_info["color"]
            name = mask_info["name"]

            draw.rectangle(
                [legend_x, legend_y, legend_x + box_size * 2, legend_y + box_size],
                fill=color,
                outline=(255, 255, 255),
                width=1,
            )

            draw.text(
                (legend_x + box_size * 2 + 5, legend_y),
                name,
                fill=color,
                font=legend_font,
            )

            legend_y += spacing

        if output_path:
            if os.path.isdir(output_path) or len(images_info) > 1:
                os.makedirs(output_path, exist_ok=True)
                base_name = os.path.splitext(img_filename)[0]
                save_path = os.path.join(output_path, f"{base_name}_comparison.jpg")
            else:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                save_path = output_path

            merged_image.save(save_path)
            print(f"  Saved: {save_path}")
            saved_paths.append(save_path)
        else:
            if len(images_info) == 1:
                return merged_image
            else:
                print(f"Note: Multiple images processed but no output_path specified.")

    if saved_paths:
        print(f"\nTotal {len(saved_paths)} comparison images saved.")
        return saved_paths
    elif len(images_info) == 1 and not output_path:
        return merged_image
    else:
        return None


if __name__ == "__main__":

    img_dir = (
        "/home/cx/cx_dir/data/deimv2_train_data/dlzdt_dataset_20260331_hbb/images/val"
    )
    gt_json = "test/data/inputs/instances_gt_coco.json"
    pred_jsons = [
        "test/data/inputs/ultralytics_pred_coco_0.5.json",
        "test/data/inputs/deimv2_pred_coco_0.5.json",
        "test/data/inputs/deimv2_hp_pred_coco_0.5.json",
    ]
    model_names = [
        "YOLOv11-x",
        "DEIMv2-l",
        "DEIMv2-hp",
    ]
    output_path = "./test/data/outputs/model_compare_hp"

    draw_coco_annotations(
        img_dir, gt_json, pred_jsons, output_path, model_names=model_names
    )
