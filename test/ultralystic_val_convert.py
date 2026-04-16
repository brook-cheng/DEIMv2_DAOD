import os
import sys
import random

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

sys.path.append(os.path.join(ROOT_DIR, "tools", "label"))

from PIL import Image
from tqdm import tqdm

from tools.label.coco_utils import (
    deimv2_outputs_to_coco_annotations,
    show_coco_annotations_on_image,
    ultralytics_val_to_coco,
)


if __name__ == "__main__":

    category_map = {1: "dlzdt"}
    img_dir = "/mnt/d/project_data/model_test/deimv2_train_data/2026_3_31_hbb/dlzdt_dataset_20260331_hbb/images/val"
    input_path = "test/data/inputs/ultralytics_val_res.json"
    ultralytics_val_to_coco(
        img_dir=img_dir,
        input_json_path=input_path,
        output_json_path="test/data/inputs/ultralytics_pred.json",
        category_name_map=category_map,
    )
