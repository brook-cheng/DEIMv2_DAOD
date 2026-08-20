"""
将Ultralytics的验证结果转换成COCO格式的标注文件，方便后续使用COCO工具进行评估和可视化。
"""

import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
from tools.model_compare.coco_utils import (
    deimv2_outputs_to_coco_annotations,
    show_coco_annotations_on_image,
    ultralytics_val_to_coco,
)

if __name__ == "__main__":

    print("-" * 50)
    print("Converting Ultralytics validation results to COCO format...")
    category_map = {1: "dlzdt"}
    img_dir = (
        "/home/cx/cx_dir/data/deimv2_train_data/dlzdt_dataset_20260331_hbb/images/val"
    )
    ultralytics_val_to_coco(
        img_dir=img_dir,
        input_json_path="test/data/inputs/ultralytics_val_res_0.5.json",
        output_json_path="test/data/inputs/ultralytics_pred_coco_0.5.json",
        category_name_map=category_map,
    )
    ultralytics_val_to_coco(
        img_dir=img_dir,
        input_json_path="test/data/inputs/ultralytics_val_res.json",
        output_json_path="test/data/inputs/ultralytics_pred_coco.json",
        category_name_map=category_map,
    )
