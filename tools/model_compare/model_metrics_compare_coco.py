"""
使用 COCOeval_faster 对多模型预测结果进行评估，终端打印不同模型指标对比。

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

from faster_coco_eval import COCO, COCOeval_faster
from typing import List, Optional, Dict
import pandas as pd
from pathlib import Path
from tabulate import tabulate


from typing import List, Dict
import pandas as pd
from pathlib import Path
from tabulate import tabulate


def compare_models_metrics(
    gt_coco_json: str,
    pred_coco_jsons: List[str],
    model_names: List[str] = None,
):
    if model_names is None:
        model_names = [Path(p).stem for p in pred_coco_jsons]

    gt_coco = COCO(gt_coco_json)
    all_stats: Dict[str, dict] = {}
    all_extended: Dict[str, dict] = {}

    for name, path in zip(model_names, pred_coco_jsons):
        print(f"[INFO] 评估: {name}")
        pred_coco = COCO(path)
        ev = COCOeval_faster(gt_coco, pred_coco, iouType="bbox")
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
        all_stats[name] = ev.stats_as_dict
        all_extended[name] = ev.extended_metrics

    # ==========================================================
    #  Table 1 — stats_as_dict
    # ==========================================================
    RENAME = {
        "AP_all": "mAP@[.50:.95]",
        "AP_50": "mAP@.50",
        "AP_75": "mAP@.75",
        "AP_small": "mAP(S)",
        "AP_medium": "mAP(M)",
        "AP_large": "mAP(L)",
        "AR_all": "AR@[.50:.95]",
        "AR_second": "AR@1",
        "AR_third": "AR@10",
        "AR_small": "AR(S)",
        "AR_medium": "AR(M)",
        "AR_large": "AR(L)",
        "AR_50": "AR@.50",
        "AR_75": "AR@.75",
    }
    ext_keys = ["map@50:95", "map@50", "precision", "recall"]

    def _na(v):
        return "N/A" if (v is None or v == -1.0) else f"{v:.4f}"

    # ---------- 统计总表 ----------
    rows = []
    for name in model_names:
        s = all_stats[name]
        rows.append({k: _na(s.get(orig)) for orig, k in RENAME.items()})
    df1 = pd.DataFrame(rows, index=model_names, columns=RENAME.values())
    df1.index.name = "Model"

    # ==========================================================
    #  Table 2 — extended_metrics（类别×指标 为行，模型 为列）
    # ==========================================================
    all_cls = list(
        dict.fromkeys(
            item["class"]
            for name in model_names
            for item in all_extended[name].get("class_map", [])
        )
    )

    rows2 = []
    for cls in all_cls:
        for mk in ext_keys:
            row = {}
            for name in model_names:
                val = None
                for item in all_extended[name].get("class_map", []):
                    if item["class"] == cls:
                        val = item.get(mk)
                        break
                row[name] = _na(val)
            rows2.append({"Class": f"{cls} | {mk}", **row})

    df2 = pd.DataFrame(rows2)
    df2 = df2.set_index("Class")

    # ---------- 打印 ----------
    print(f"\n{'=' * 90}")
    print("  Table 1  Overall Metrics  (stats_as_dict)")
    print(f"{'=' * 90}")
    print(tabulate(df1, headers="keys", tablefmt="grid", stralign="center"))

    print(f"\n{'=' * 90}")
    print("  Table 2  Per-Class Metrics  (extended_metrics)")
    print(f"{'=' * 90}")
    print(tabulate(df2, headers="keys", tablefmt="grid", stralign="center"))

    return df1, df2


if __name__ == "__main__":
    gt_coco_json = "test/data/inputs/instances_gt_coco.json"
    pred_coco_jsons = [
        "test/data/inputs/ultralytics_pred_coco.json",
        "test/data/inputs/deimv2_pred_coco.json",
        "test/data/inputs/deimv2_hp_pred_coco.json",
    ]

    model_names = ["YOLOv11-x", "DEIMv2-l", "DEIMv2-hp"]
    compare_models_metrics(
        gt_coco_json=gt_coco_json,
        pred_coco_jsons=pred_coco_jsons,
        model_names=model_names,
    )
