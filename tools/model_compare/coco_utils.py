import os
import json
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime
from typing import Dict, List, Optional, Union
import matplotlib.pyplot as plt
import numpy as np


def deimv2_outputs_to_coco_annotations(
    img_dir,
    outputs_dict,
    labels_map,
    output_json_path="./test/deimv2_predictions.json",
    year=2025,
    description="DEIMv2 Predictions",
    skip_background=True,
):
    """
    将 DEIMv2 模型预测结果转换为 COCO 标注格式并保存

    Args:
        img_dir: 图像目录路径，该目录下所有图片都会被包含在 images 字段中
        outputs_dict: 模型预测结果字典，格式为:
            {
                "image_name.jpg": {
                    "labels": [0, 1, 2],
                    "boxes": [[x1, y1, x2, y2], ...],
                    "scores": [0.9, 0.8, 0.7]
                },
                ...
            }
        labels_map: 类别映射字典，如 {0: "background", 1: "dlzdt", 2: "null"}
        output_json_path: 输出的 JSON 文件路径
        year: 年份信息
        description: 描述信息
        skip_background: 是否跳过 category_id=0 的背景类（COCO 标准）
    """
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)

    categories = []
    category_id_mapping = {}

    for label_id, label_name in labels_map.items():
        if skip_background and int(label_id) == 0:
            continue

        coco_category_id = int(label_id)
        categories.append(
            {"id": coco_category_id, "name": label_name, "supercategory": ""}
        )
        category_id_mapping[int(label_id)] = coco_category_id

    supported_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp")
    all_image_files = []
    for filename in sorted(os.listdir(img_dir)):
        if filename.lower().endswith(supported_extensions):
            all_image_files.append(filename)

    print(f"Found {len(all_image_files)} images in {img_dir}")

    images = []
    for img_idx, img_name in enumerate(all_image_files):
        img_path = os.path.join(img_dir, img_name)

        try:
            image = Image.open(img_path)
            img_width, img_height = image.size
        except Exception as e:
            img_width, img_height = 0, 0
            print(f"Error reading image {img_path}: {e}")
            continue

        images.append(
            {
                "license": 0,
                "url": None,
                "file_name": img_name,
                "height": img_height,
                "width": img_width,
                "date_captured": None,
                "id": img_idx + 1,
            }
        )

    annotations = []
    annotation_id = 1

    images_with_predictions = 0

    for img_idx, img_name in enumerate(all_image_files):
        if img_name not in outputs_dict:
            continue

        images_with_predictions += 1
        pred_result = outputs_dict[img_name]

        labels = pred_result["labels"]
        boxes = pred_result["boxes"]
        scores = pred_result["scores"]

        for label, box, score in zip(labels, boxes, scores):
            label_id = int(label)

            if skip_background and label_id == 0:
                continue

            x_min, y_min, x_max, y_max = box

            width = x_max - x_min
            height = y_max - y_min
            area = width * height

            coco_category_id = category_id_mapping.get(label_id, label_id)

            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": img_idx + 1,
                    "category_id": coco_category_id,
                    "bbox": [float(x_min), float(y_min), float(width), float(height)],
                    "area": float(area),
                    "iscrowd": 0,
                    "ignore": 0,
                    "segmentation": [],
                    "score": float(score),
                }
            )
            annotation_id += 1

    coco_format = {
        "info": {
            "year": year,
            "version": "1.0",
            "description": description,
            "contributor": "DEIMv2",
            "url": "",
            "date_created": datetime.now().strftime("%Y-%m-%d"),
        },
        "licenses": [{"id": 1, "url": "", "name": "Unknown"}],
        "categories": categories,
        "images": images,
        "annotations": annotations,
    }

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(coco_format, f, indent=4, ensure_ascii=False)

    print(f"COCO annotations saved to: {output_json_path}")
    print(f"Total images in directory: {len(images)}")
    print(f"Images with predictions: {images_with_predictions}")
    print(f"Total annotations: {len(annotations)}")
    print(f"Categories: {len(categories)}")
    if skip_background:
        print(
            "Note: Background class (category_id=0) has been skipped following COCO standard"
        )

    return coco_format


def show_coco_annotations_on_image(
    img_dir: str,
    coco_annotations: Union[Dict, str],
    img_idxes: Optional[List[int]] = None,
    img_names: Optional[List[str]] = None,
    labels_map: Optional[Dict] = None,
    score_threshold: float = 0.5,
    output_path: Optional[str] = None,
    save_flags: bool = True,
    mask_flag: bool = False,
):
    """
    在图像上显示 COCO 格式的标注结果

    Args:
        img_dir: 被标注图片所在文件夹路径
        coco_annotations: COCO 标注的 JSON 文件路径或从 JSON 文件中获取的字典数据
        img_idxes: 需要绘制的图像序号列表（基于 images 列表的索引）
        img_names: 需要绘制的图像名称列表
                  - 为 None 时，只从 img_idxes 中获取需要绘制的对象
                  - img_idxes 和 img_names 都为空时，绘制 coco_annotations 中所有对象
        labels_map: 类别映射字典，如 {1: "dlzdt", 2: "null"}。如果为 None，则从 coco_annotations 中提取
        score_threshold: 置信度阈值，只显示分数高于此阈值的标注
        output_path: 输出图像路径或目录
                    - 如果为 None，返回单个 PIL Image 对象（仅当只处理一张图时）
                    - 如果指定为文件路径且只处理一张图，保存到该路径
                    - 如果指定为目录路径或处理多张图，保存到 output_path/{filename}_annotated.jpg
        save_flags: 是否保存标注结果到文件（仅当 output_path 不为 None 时有效）
        mask_flag: 是否绘制分割掩码
                   - False: 在原始图像上绘制边界框和标签
                   - True: 生成 mask 图像，mask 值为类别 ID，方便多模型对比


    Returns:
        如果只处理一张图且 output_path 为 None，返回 PIL Image 对象
        否则返回保存的文件路径列表
    """
    import matplotlib.pyplot as plt
    import numpy as np

    if isinstance(coco_annotations, str):
        with open(coco_annotations, "r", encoding="utf-8") as f:
            coco_data = json.load(f)
    else:
        coco_data = coco_annotations

    if (
        not isinstance(coco_data, dict)
        or "images" not in coco_data
        or "annotations" not in coco_data
    ):
        raise ValueError(
            "coco_annotations must be a valid COCO format dictionary or path to COCO JSON file"
        )

    images_info = coco_data["images"]
    annotations = coco_data["annotations"]

    if labels_map is None:
        labels_map = {}
        if "categories" in coco_data:
            for cat in coco_data["categories"]:
                labels_map[cat["id"]] = cat["name"]

    num_classes = len(labels_map) if labels_map else 10
    colors = plt.cm.rainbow(np.linspace(0, 1, max(num_classes, 1)))
    color_map = (
        {
            label_id: tuple([int(c * 255) for c in colors[i % len(colors)]])
            for i, label_id in enumerate(labels_map.keys())
        }
        if labels_map
        else {}
    )

    if img_idxes is None and img_names is None:
        target_images = images_info
    else:
        target_images = []

        if img_idxes is not None:
            for idx in img_idxes:
                if 0 <= idx < len(images_info):
                    target_images.append(images_info[idx])
                else:
                    print(
                        f"Warning: Image index {idx} is out of range (0-{len(images_info)-1}), skipping"
                    )

        if img_names is not None:
            image_name_set = {img["file_name"] for img in target_images}
            for name in img_names:
                if name not in image_name_set:
                    matched_img = None
                    for img in images_info:
                        if img["file_name"] == name:
                            matched_img = img
                            break

                    if matched_img:
                        target_images.append(matched_img)
                        image_name_set.add(name)
                    else:
                        print(
                            f"Warning: Image '{name}' not found in COCO annotations, skipping"
                        )

    if not target_images:
        raise ValueError("No images selected for visualization")

    print(f"Processing {len(target_images)} images...")

    saved_paths = []
    text_fill_color = "white" if not mask_flag else "black"
    for img_info in target_images:
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
        reference_size = 640
        scale_factor = max(img_width, img_height) / reference_size

        line_width = max(2, int(2 * scale_factor))
        font_size = max(12, int(9 * scale_factor))
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size
            )
        except:
            font = ImageFont.load_default()

        img_annotations = [ann for ann in annotations if ann["image_id"] == img_id]

        filtered_annotations = []
        for ann in img_annotations:
            score = ann.get("score", 1.0)
            if score >= score_threshold:
                filtered_annotations.append(ann)
        if mask_flag:
            image = Image.new("L", (img_width, img_height), 0)
            draw = ImageDraw.Draw(image)
        else:
            draw = ImageDraw.Draw(image)
        for ann in filtered_annotations:
            category_id = ann["category_id"]
            bbox = ann["bbox"]
            score = ann.get("score", 1.0)

            x_min, y_min, width, height = bbox
            x_max = x_min + max(1, width)
            y_max = y_min + max(1, height)

            if mask_flag:
                color = category_id
            else:
                color = (
                    color_map.get(category_id, (255, 0, 0))
                    if color_map
                    else (255, 0, 0)
                )

            draw.rectangle(
                [x_min, y_min, x_max, y_max], outline=color, width=line_width
            )

            label_name = labels_map.get(category_id, f"class_{category_id}")
            text = f"{label_name}: {score:.2f}"

            text_bbox = draw.textbbox((x_min, y_min), text, font=font)
            text_height = text_bbox[3] - text_bbox[1] + line_width

            if y_min - text_height >= 0:
                text_y = y_min - text_height
            else:
                text_y = y_min

            draw.rectangle(
                [text_bbox[0], text_y, text_bbox[2], text_y + text_height],
                fill=color,
            )

            draw.text((x_min, text_y), text, fill=text_fill_color, font=font)

        if output_path:
            if os.path.isdir(output_path) or len(target_images) > 1:
                os.makedirs(output_path, exist_ok=True)
                base_name = os.path.splitext(img_filename)[0]
                save_path = os.path.join(output_path, f"{base_name}_annotated.jpg")
            else:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                save_path = output_path

            if save_flags:
                image.save(save_path)
                print(f"Visualization saved to: {save_path}")
                saved_paths.append(save_path)
        else:
            if len(target_images) == 1:
                return image
            else:
                print(
                    f"Note: Multiple images processed but no output_path specified. Use output_path to save results."
                )

    if saved_paths:
        return saved_paths
    elif len(target_images) == 1 and not output_path:
        return image
    else:
        return None


def ultralytics_val_to_coco(
    img_dir,
    input_json_path,
    output_json_path="./test/ultralytics_val_coco.json",
    year=2025,
    description="Ultralytics Validation Results",
    category_name_map=None,
):
    """
    将 Ultralytics 验证输出转换为 COCO 标注格式

    Args:
        img_dir: 图像目录路径，该目录下所有图片都会被包含在 images 字段中
        input_json_path: Ultralytics 验证输出的 JSON 文件路径
        output_json_path: 输出的 COCO 格式 JSON 文件路径
        year: 年份信息
        description: 描述信息
        category_name_map: 类别 ID 到名称的映射字典，如 {1: "bh_guagoutou"}
                          如果为 None，则使用默认类别名称
    """
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)

    with open(input_json_path, "r", encoding="utf-8") as f:
        ultralytics_data = json.load(f)

    supported_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp")
    all_image_files = []
    for filename in sorted(os.listdir(img_dir)):
        if filename.lower().endswith(supported_extensions):
            all_image_files.append(filename)

    print(f"Found {len(all_image_files)} images in {img_dir}")

    images_dict = {}
    for img_idx, filename in enumerate(all_image_files):
        img_path = os.path.join(img_dir, filename)
        try:
            image = Image.open(img_path)
            img_width, img_height = image.size
        except Exception as e:
            img_width, img_height = 0, 0
            print(f"Error reading image {img_path}: {e}")

        images_dict[filename] = {
            "license": 0,
            "url": None,
            "file_name": filename,
            "height": img_height,
            "width": img_width,
            "date_captured": None,
            "id": img_idx + 1,
        }

    predictions_by_image = {}
    category_ids = set()

    for pred in ultralytics_data:
        file_name = pred["file_name"]
        category_id = pred["category_id"]
        bbox = pred["bbox"]
        score = pred.get("score", 1.0)

        category_ids.add(category_id)

        if file_name not in predictions_by_image:
            predictions_by_image[file_name] = []

        predictions_by_image[file_name].append(
            {"category_id": category_id, "bbox": bbox, "score": score}
        )

    annotations = []
    annotation_id = 1

    for img_filename, preds in predictions_by_image.items():
        if img_filename not in images_dict:
            print(
                f"Warning: Image '{img_filename}' in predictions but not found in {img_dir}, skipping"
            )
            continue

        image_id = images_dict[img_filename]["id"]

        for pred in preds:
            category_id = pred["category_id"]
            x_min, y_min, width, height = pred["bbox"]
            area = width * height
            score = pred["score"]

            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": [float(x_min), float(y_min), float(width), float(height)],
                    "area": float(area),
                    "iscrowd": 0,
                    "ignore": 0,
                    "segmentation": [],
                    "score": float(score),
                }
            )
            annotation_id += 1

    if category_name_map is None:
        category_name_map = {
            cat_id: f"class_{cat_id}" for cat_id in sorted(category_ids)
        }

    categories = []
    for cat_id in sorted(category_ids):
        cat_name = category_name_map.get(cat_id, f"class_{cat_id}")
        categories.append({"id": cat_id, "name": cat_name, "supercategory": ""})

    images = list(images_dict.values())

    coco_format = {
        "info": {
            "year": year,
            "version": "1.0",
            "description": description,
            "contributor": "Ultralytics",
            "url": "",
            "date_created": datetime.now().strftime("%Y-%m-%d"),
        },
        "licenses": [{"id": 1, "url": "", "name": "Unknown"}],
        "categories": categories,
        "images": images,
        "annotations": annotations,
    }

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(coco_format, f, indent=4, ensure_ascii=False)

    print(f"COCO annotations saved to: {output_json_path}")
    print(f"Total images in directory: {len(images)}")
    print(f"Images with predictions: {len(predictions_by_image)}")
    print(f"Total annotations: {len(annotations)}")
    print(f"Categories: {len(categories)}")

    return coco_format
