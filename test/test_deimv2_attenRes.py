"""
使用deimv2模型进行预测,绘制结果并将结果转换为coco格式
"""

import os
import sys
import random

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

sys.path.append(os.path.join(ROOT_DIR, "tools", "label"))

from PIL import Image
from tqdm import tqdm
import torch

from tools.model_compare.coco_utils import (
    deimv2_outputs_to_coco_annotations,
    show_coco_annotations_on_image,
)


def export_predictions(
    img_dir,
    config_path,
    model_weight,
    score_threshold=0.5,
    num_visualize=10,
    output_dir="./test/outputs",
    output_json_path="predict_annotations.json",
    labels_map=None,
    device="cuda:0",
):
    """
    测试模型并导出 COCO 格式标注和可视化结果

    Args:
        img_dir: 图像目录
        config_path: 训练配置 YAML（模型结构、类别数、输入尺寸均来自该配置）
        model_weight: 模型权重路径
        score_threshold: 置信度阈值
        num_visualize: 随机抽取可视化的图片数量
        output_dir: 输出目录
        labels_map: 类别映射字典
        device: 设备
    """
    from torchvision import transforms

    from engine.core import YAMLConfig

    if labels_map is None:
        labels_map = {0: "background", 1: "dlzdt", 2: "null"}

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print("DEIMv2 Model Testing and Export")
    print("=" * 80)
    print(f"Image directory: {img_dir}")
    print(f"Config: {config_path}")
    print(f"Model weight: {model_weight}")
    print(f"Score threshold: {score_threshold}")
    print(f"Number of visualizations: {num_visualize}")
    print(f"Output directory: {output_dir}")
    print(f"Device: {device}")
    print("=" * 80)

    cfg = YAMLConfig(config_path)
    model, postprocessor = cfg.model.deploy(), cfg.postprocessor.deploy()
    model.eval()
    ckpt = torch.load(model_weight, map_location="cpu", weights_only=True)
    ema = ckpt.get("ema")
    state = (
        ema["module"]
        if isinstance(ema, dict) and "module" in ema
        else ckpt.get("model", ckpt)
    )
    model.load_state_dict(state)
    model.to(device)
    postprocessor.to(device)

    imgsz = tuple(cfg.yaml_cfg.get("eval_spatial_size", (640, 640)))
    transform = transforms.Compose(
        [transforms.Resize(imgsz), transforms.ToTensor()]
    )

    def infer(image):
        input_tensor = transform(image).unsqueeze(0).to(device)
        w, h = image.size
        orig_size = torch.tensor([[w, h]], device=device)
        with torch.no_grad():
            outputs = model(input_tensor)
            labels, boxes, scores = postprocessor(outputs, orig_size)
        keep = scores[0] > score_threshold
        return {
            "labels": labels[0][keep].cpu().numpy().tolist(),
            "boxes": boxes[0][keep].cpu().numpy().tolist(),
            "scores": scores[0][keep].cpu().numpy().tolist(),
        }

    img_list = [
        f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    if not img_list:
        print(f"No images found in {img_dir}")
        return

    print(f"\nFound {len(img_list)} images")
    print("\n=== Running Inference ===")

    outputs_dict = {}
    for img_name in tqdm(img_list, desc="Inference"):
        img_path = os.path.join(img_dir, img_name)
        try:
            image = Image.open(img_path).convert("RGB")
            outputs_dict[img_name] = infer(image)
        except Exception as e:
            print(f"Error processing {img_name}: {e}")
            continue

    print(f"\nInference completed for {len(outputs_dict)} images")

    print("\n=== Exporting COCO Annotations ===")
    coco_data = deimv2_outputs_to_coco_annotations(
        img_dir=img_dir,
        outputs_dict=outputs_dict,
        labels_map=labels_map,
        output_json_path=output_json_path,
        skip_background=True,
    )

    categories_map = {cat["id"]: cat["name"] for cat in coco_data["categories"]}

    print("\n=== Generating Visualizations ===")
    if len(outputs_dict) > 0:
        visualize_count = (
            min(num_visualize, len(outputs_dict))
            if num_visualize > 0
            else len(outputs_dict)
        )
        selected_images = random.sample(list(outputs_dict.keys()), visualize_count)

        print(f"Randomly selected {visualize_count} images for visualization")

        selected_img_names = selected_images

        saved_paths = show_coco_annotations_on_image(
            img_dir=img_dir,
            coco_annotations=coco_data,
            img_names=selected_img_names,
            labels_map=categories_map,
            score_threshold=score_threshold,
            output_path=output_dir,
        )

        if saved_paths:
            print(f"\nVisualizations saved to: {output_dir}")
            print(f"Total visualized: {len(saved_paths)} images")
        else:
            print("\nNo visualizations generated")
    else:
        print("No predictions to visualize")

    print("\n" + "=" * 80)
    print("Testing and Export Completed Successfully!")
    print("=" * 80)
    print(f"COCO annotations: {output_json_path}")
    print(f"Visualizations: {output_dir} (*.jpg)")
    print(f"Total images processed: {len(outputs_dict)}")
    print(f"Total annotations: {len(coco_data['annotations'])}")
    print("=" * 80)


if __name__ == "__main__":
    img_dir = (
        "/home/cx/cx_dir/data/deimv2_train_data/dlzdt_dataset_20260331_hbb/images/val"
    )
    config_path = "configs/custom/deimv2_dinov3_vith16p_freeze.yml"
    model_weight = "./outputs/dlzdt_vith16p_freeze/best_stg2.pth"

    score_threshold = 0.0
    num_visualize = 10
    output_dir = "./test/data/outputs/deimv2_attenRes/"
    output_json_path = "./test/data/inputs/deimv2_hp_pred_coco.json"

    labels_map = {0: "background", 1: "dlzdt", 2: "null"}

    export_predictions(
        img_dir=img_dir,
        config_path=config_path,
        model_weight=model_weight,
        score_threshold=score_threshold,
        num_visualize=num_visualize,
        output_dir=output_dir,
        output_json_path=output_json_path,
        labels_map=labels_map,
        device="cuda:1",
    )
