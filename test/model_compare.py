import os
import sys

import json
import logging
import faster_coco_eval

faster_coco_eval.init_as_pycocotools()
from faster_coco_eval import COCO, COCOeval_faster

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

sys.path.append(os.path.join(ROOT_DIR, "tools", "label"))


from tools.label.coco_utils import (
    deimv2_outputs_to_coco_annotations,
    show_coco_annotations_on_image,
    ultralytics_val_to_coco,
)


def ultralytics_val_to_coco():
    category_map = {1: "dlzdt"}
    img_dir = "/mnt/d/project_data/model_test/deimv2_train_data/2026_3_31_hbb/dlzdt_dataset_20260331_hbb/images/val"
    input_path = "test/data/inputs/ultralytics_val_res.json"
    ultralytics_val_to_coco(
        img_dir=img_dir,
        input_json_path=input_path,
        output_json_path="test/data/inputs/ultralytics_pred_coco.json",
        category_name_map=category_map,
    )


def coco_eval(coco_gt_js, coco_dt_js):
    logging.root.setLevel("INFO")
    cocoGt = COCO(coco_gt_js)
    cocoDt = COCO(coco_dt_js)
    iouType = "bbox"

    cocoEval = COCOeval_faster(cocoGt, cocoDt, iouType)

    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    # print(cocoEval.stats_as_dict)
    # print(cocoEval.extended_metrics)


if __name__ == "__main__":
    gt_coco = "test/data/inputs/instances_gt_coco.json"
    diemv2_dt_coco = "test/data/inputs/deimv2_pred_coco.json"
    ultraly_dt_coco = "test/data/inputs/ultralytics_pred_coco.json"
    logging.root.setLevel("INFO")
    logging.debug("start eval")
    coco_eval(gt_coco, diemv2_dt_coco)
    print("-" * 20)
    coco_eval(gt_coco, ultraly_dt_coco)
