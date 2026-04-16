import os
import json
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime


def deimv2_outputs_to_coco_annotations(
    img_dir,
    outputs_dict,
    labels_map,
    output_json_path="./test/predictions.json",
    year=2025,
    description="DEIMv2 Predictions",
    skip_background=True,
):
    """
    将 DEIMv2 模型预测结果转换为 COCO 标注格式并保存

    Args:
        img_dir: 图像目录路径
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

    images = []
    annotations = []
    annotation_id = 0

    for img_idx, (img_name, pred_result) in enumerate(outputs_dict.items()):
        img_path = os.path.join(img_dir, img_name)

        if not os.path.exists(img_path):
            print(f"Warning: Image not found: {img_path}")
            continue

        try:
            image = Image.open(img_path)
            img_width, img_height = image.size
        except Exception as e:
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
                "id": img_idx,
            }
        )

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
                    "image_id": img_idx,
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
    print(f"Total images: {len(images)}")
    print(f"Total annotations: {len(annotations)}")
    print(f"Categories: {len(categories)}")
    if skip_background:
        print(
            "Note: Background class (category_id=0) has been skipped following COCO standard"
        )

    return coco_format


def show_coco_annotations_on_image(
    img_path,
    coco_annotations,
    labels_map=None,
    score_threshold=0.5,
    output_path=None,
):
    """
    在图像上显示 COCO 格式的标注结果

    Args:
        img_path: 图像路径
        coco_annotations: COCO 格式的标注数据（字典）或单个图像的 annotations 列表
        labels_map: 类别映射字典，如 {1: "dlzdt", 2: "null"}
        score_threshold: 置信度阈值
        output_path: 输出图像路径，如果为 None 则返回 PIL Image 对象
    """
    import matplotlib.pyplot as plt
    import numpy as np

    image = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    img_width, img_height = image.size
    reference_size = 640
    scale_factor = max(img_width, img_height) / reference_size

    line_width = max(2, int(2 * scale_factor))
    font_size = max(12, int(15 * scale_factor))

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size
        )
    except:
        font = ImageFont.load_default()

    if isinstance(coco_annotations, dict):
        if "annotations" in coco_annotations:
            annotations = coco_annotations["annotations"]
        else:
            annotations = coco_annotations
    else:
        annotations = coco_annotations

    if labels_map is None:
        labels_map = {}
        if isinstance(coco_annotations, dict) and "categories" in coco_annotations:
            for cat in coco_annotations["categories"]:
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

    filtered_annotations = []
    for ann in annotations:
        score = ann.get("score", 1.0)
        if score >= score_threshold:
            filtered_annotations.append(ann)

    for ann in filtered_annotations:
        category_id = ann["category_id"]
        bbox = ann["bbox"]
        score = ann.get("score", 1.0)

        x_min, y_min, width, height = bbox
        x_max = x_min + width
        y_max = y_min + height

        color = color_map.get(category_id, (255, 0, 0)) if color_map else (255, 0, 0)

        draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=line_width)

        label_name = labels_map.get(category_id, f"class_{category_id}")
        text = f"{label_name}: {score:.2f}"

        text_bbox = draw.textbbox((x_min, y_min), text, font=font)
        text_height = text_bbox[3] - text_bbox[1]

        if y_min - text_height - 5 >= 0:
            text_y = y_min - text_height - 5
        else:
            text_y = y_min

        draw.rectangle(
            [text_bbox[0], text_y, text_bbox[2], text_y + text_height], fill=color
        )
        draw.text((x_min, text_y), text, fill="white", font=font)

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        image.save(output_path)
        print(f"Visualization saved to: {output_path}")
        return output_path
    else:
        return image
