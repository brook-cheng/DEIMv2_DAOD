#!/usr/bin/env python3
"""DEIMv2-OBB 批量推理导出 CLI（单模型）。

批量多模型对比研究请编程使用 ``tools.compare.core.OBBModelSpec`` +
``run_model_specs``，或参照本文件底部的 SPECS 示例。

Usage:
    python tools/compare/run_infer.py \
        --img-dir <图像目录> --classes-txt <classes.txt> \
        --config <训练YAML> --ckpt <checkpoint> \
        --output-dir <DOTA输出目录> [--vis-dir <可视化目录>] \
        [--imgsz 640 640] [--max-det 300] [--score-threshold 0.0] \
        [--device cuda:0]
"""

import argparse

from tools.compare.core import OBBModelSpec, infer_obb_and_export, run_model_specs


def main() -> None:
    parser = argparse.ArgumentParser(description="DEIMv2-OBB inference → DOTA export")
    parser.add_argument("--img-dir", type=str)
    parser.add_argument("--classes-txt", type=str)
    parser.add_argument("--config", type=str)
    parser.add_argument("--ckpt", type=str)
    parser.add_argument("--output-dir", type=str)
    parser.add_argument("--vis-dir", type=str, default=None)
    parser.add_argument("--imgsz", type=int, nargs=2, default=(640, 640))
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--score-threshold", type=float, default=0.0)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    infer_obb_and_export(
        img_dir=args.img_dir,
        ckpt=args.ckpt,
        config=args.config,
        output_dir=args.output_dir,
        classes_txt=args.classes_txt,
        imgsz=tuple(args.imgsz),
        max_det=args.max_det,
        score_threshold=args.score_threshold,
        device=args.device,
        vis_dir=args.vis_dir,
    )


# 批量多模型示例（研究矩阵）：取消注释并直接运行本文件即可。
_SPECS_EXAMPLE = [
    OBBModelSpec(
        name="abl_rep0",
        config="configs/custom_obb/dlzdt/sp_fz_common.yml",
        ckpt="outputs/dlzdt_ablation/abl_rep0.pth",
        output_dir="./test/data/outputs/dlzdt_res/dlzdt_ablation/abl_rep0",
    ),
    OBBModelSpec(
        name="abl_rep3",
        config="configs/custom_obb/dlzdt/ablation/abl_rep3.yml",
        ckpt="outputs/dlzdt_ablation/abl_rep3.pth",
        output_dir="./test/data/outputs/dlzdt_res/dlzdt_ablation/abl_rep3",
    ),
]

if __name__ == "__main__":
    main()
