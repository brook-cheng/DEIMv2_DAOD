"""
OBB metrics comparison — evaluate multiple models via evaluate_dota and print tables.

Usage (script):
    python tools/model_compare/model_metrics_compare_obb.py

Usage (API):
    from tools.model_compare.model_metrics_compare_obb import compare_obb_models
    compare_obb_models(gt_dir, det_dirs, classes_file, model_names=...)
"""

import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from typing import Dict, List, Optional
from pathlib import Path

from engine.eval.dota_eval import evaluate_dota


def _print_table(headers, rows):
    """Print a simple aligned text table."""
    ncols = len(headers)
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))
    fmt = "  " + "  ".join(f"{{:<{w}}}" for w in col_widths)
    sep = "  " + "  ".join("-" * w for w in col_widths)
    print(fmt.format(*headers))
    print(sep)
    for row in rows:
        print(fmt.format(*[str(c) for c in row]))


def compare_obb_models(
    gt_dir: str,
    det_dirs: List[str],
    classes_file: str,
    model_names: Optional[List[str]] = None,
    iouv=None,
):
    """Evaluate multiple OBB models against the same GT and print comparison tables.

    Args:
        gt_dir:       directory of per-image ground truth .txt files.
        det_dirs:     list of per-image prediction directories (one per model).
        classes_file: path to classes.txt (one class per line).
        model_names:  list of model names (auto if None).
        iouv:         IoU thresholds (default 0.5:0.95).

    Returns:
        (df_overall, df_per_class) pandas DataFrames.
    """
    if model_names is None:
        model_names = [Path(d).parent.name for d in det_dirs]

    all_results: Dict[str, dict] = {}

    for name, det_dir in zip(model_names, det_dirs):
        print(f"[INFO] Evaluating: {name}")
        result = evaluate_dota(det_dir, gt_dir, classes_file, iouv=iouv)
        all_results[name] = result
        print(
            f"       mAP={result['mAP']:.4f}  AP50={result['AP50']:.4f}  "
            f"AP75={result['AP75']:.4f}"
        )

    # ── Table 1: Overall metrics ──
    OVERALL_KEYS = [
        ("mAP", "mAP@[.50:.95]"),
        ("AP50", "mAP@.50"),
        ("AP75", "mAP@.75"),
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("f1", "F1"),
    ]

    headers1 = ["Model"] + [label for _, label in OVERALL_KEYS]
    rows1 = []
    for name in model_names:
        r = all_results[name]
        rows1.append([name] + [f"{r.get(key, 0):.4f}" for key, _ in OVERALL_KEYS])

    # ── Table 2: Per-class AP50_95 ──
    all_classes = set()
    for r in all_results.values():
        all_classes.update(r.get("per_class", {}).keys())
    all_classes = sorted(all_classes)

    rows2 = []
    if all_classes:
        headers2 = ["Model"] + list(all_classes)
        for name in model_names:
            r = all_results[name]
            row = [name]
            for cls in all_classes:
                row.append(f"{r['per_class'].get(cls, {}).get('AP50_95', 0):.4f}")
            rows2.append(row)
    else:
        headers2 = ["Model"]

    # ── Print ──
    print(f"\n{'=' * 80}")
    print("  Table 1 — Overall Metrics")
    print(f"{'=' * 80}")
    _print_table(headers1, rows1)

    if rows2:
        print(f"\n{'=' * 80}")
        print("  Table 2 — Per-Class mAP@0.5:0.95")
        print(f"{'=' * 80}")
        _print_table(headers2, rows2)

    return {"overall": rows1, "per_class": rows2}


if __name__ == "__main__":
    demo_dir = os.path.join(os.path.dirname(__file__), "annotation_demo")

    # create demo GT with 3 planes
    gt_dir = os.path.join(demo_dir, "gt")
    os.makedirs(gt_dir, exist_ok=True)
    with open(os.path.join(gt_dir, "demo.txt"), "w") as f:
        f.write(
            "503.998464 416.002048 497.999872 294.001664 "
            "622.001152 298.000384 626.00192 421.997568 plane 0\n"
        )
        f.write(
            "573.99808 288.001024 442.000384 252.002304 "
            "465.999872 146.00192 595.99872 177.997824 plane 0\n"
        )
        f.write(
            "225.998848 404.000768 48.0013312 373.997568 "
            "72.0002048 176.001024 248.000512 211.999744 plane 0\n"
        )

    # DEIMv2 predictions = exact GT (perfect) with high scores
    deimv2_dir = os.path.join(demo_dir, "det_deimv2")
    os.makedirs(deimv2_dir, exist_ok=True)
    import shutil

    shutil.copy(
        os.path.join(demo_dir, "dota_annotation_obb_demo.txt"),
        os.path.join(deimv2_dir, "demo.txt"),
    )
    # fix: append score=0.9 to each line
    with open(os.path.join(deimv2_dir, "demo.txt"), "r") as f:
        lines = f.readlines()
    with open(os.path.join(deimv2_dir, "demo.txt"), "w") as f:
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 9:
                parts[-1] = "0.9"
            f.write(" ".join(parts) + "\n")

    # YOLO predictions: convert JSON demo to DOTA
    from tools.model_compare.obb_utils import ultralytics_obb_json_to_dota

    yolo_dir = os.path.join(demo_dir, "det_yolo")
    ultralytics_obb_json_to_dota(
        os.path.join(demo_dir, "ultralytics_val_obb_demo.json"),
        yolo_dir,
        category_map={1: "plane"},
        score_threshold=0.5,
    )

    classes_file = os.path.join(demo_dir, "dota_classes.txt")

    compare_obb_models(
        gt_dir=gt_dir,
        det_dirs=[deimv2_dir, yolo_dir],
        classes_file=classes_file,
        model_names=["DEIMv2-OBB", "YOLO-OBB"],
    )
