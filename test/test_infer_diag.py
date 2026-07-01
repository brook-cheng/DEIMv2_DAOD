"""DEIMv2-OBB 推理诊断：观察预测置信度分布、框质量、GT vs Pred 对比。

用法：
    python test/test_infer_diag.py
    python test/test_infer_diag.py --num 5 --conf 0.1 --conf-step 0.2
"""

import os, sys, argparse, math
import torch
import numpy as np
from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from engine.deim.obb_geometry import xywhr_to_xyxyxyxy
from engine.core import YAMLConfig
from engine.solver import TASKS

OUTPUT_DIR = os.path.join(ROOT, "test", "outputs", "infer_diag")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 按 class ID 固定颜色映射，同类目标颜色一致
COLORS = [
    (255, 50, 50),  # 0: red
    (50, 200, 50),  # 1: green
    (50, 50, 255),  # 2: blue
    (255, 200, 0),  # 3: orange/yellow
    (200, 0, 200),  # 4: magenta
    (0, 200, 200),  # 5: cyan
    (180, 100, 50),  # 6: brown
    (100, 180, 50),  # 7: olive
    (50, 100, 200),  # 8: steel blue
    (200, 50, 100),  # 9: rose
    (140, 140, 140),  # 10: gray
    (255, 100, 150),  # 11: pink
    (150, 255, 100),  # 12: lime
    (100, 150, 255),  # 13: sky
    (255, 150, 50),  # 14: tangerine
    (150, 50, 255),  # 15: violet
    (50, 255, 200),  # 16: mint
    (200, 200, 100),  # 17: sand
    (100, 200, 255),  # 18: light blue
    (255, 100, 200),  # 19: hot pink
]


def load_model(ckpt_path, config_path):
    """加载训练好的模型。"""
    cfg = YAMLConfig(config_path)
    solver = TASKS[cfg.yaml_cfg["task"]](cfg)
    solver.train()

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    solver.model.load_state_dict(ckpt["ema"]["module"])
    solver.model.cuda().eval()
    return solver.model, solver.postprocessor, cfg


def draw_obb(draw, boxes, labels, scores=None, width=2):
    """在 PIL ImageDraw 上绘制 OBB 旋转矩形，颜色由 label ID 固定映射。"""
    if boxes.numel() == 0:
        return
    verts = xywhr_to_xyxyxyxy(boxes)
    for i in range(len(boxes)):
        lid = int(labels[i].item())
        c = COLORS[lid % len(COLORS)]
        pts = [(float(verts[i, j, 0]), float(verts[i, j, 1])) for j in range(4)]
        draw.polygon(pts, outline=c, width=width)
        cx, cy = float(boxes[i, 0]), float(boxes[i, 1])
        label = str(lid)
        if scores is not None:
            label += f"({scores[i]:.2f})"
        bbox = draw.textbbox((cx + 4, cy - 14), label)
        draw.rectangle(
            [bbox[0] - 2, bbox[1] - 1, bbox[2] + 2, bbox[3] + 1], fill=(30, 30, 30)
        )
        draw.text((cx + 4, cy - 14), label, fill=c)


def score_buckets(scores, conf_step, summary=False):
    """按 conf_step 分段统计分数分布。

    Args:
        scores: np.ndarray 或 torch.Tensor，置信度数组
        conf_step: 分段步长（如 0.2 → 5 段：0-0.2, 0.2-0.4, ..., 0.8-1.0）
        summary: 返回简洁字符串还是列表

    Returns:
        如果 summary=True: 字符串，如 "0-0.2=45 0.2-0.4=32 ..."
        否则: list[(lo, hi, count)] 按 lo 升序排列
    """
    if isinstance(scores, torch.Tensor):
        scores = scores.cpu().numpy()
    n = int(math.ceil(1.0 / conf_step))
    buckets = []
    for k in range(n):
        lo = k * conf_step
        hi = min((k + 1) * conf_step, 1.0)
        cnt = ((scores >= lo) & (scores < hi)).sum()
        buckets.append((lo, hi, int(cnt)))
    if summary:
        return "  ".join(f"{lo:.1f}-{hi:.1f}={cnt}" for lo, hi, cnt in buckets)
    return buckets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        default=os.path.join(ROOT, "outputs/synthetic_exp_020/last_0701_add1.pth"),
    )
    parser.add_argument(
        "--config",
        default=os.path.join(
            ROOT, "configs/custom_obb/synthetic_configs/synthetic_exp_020.yml"
        ),
    )
    parser.add_argument("--num", type=int, default=4, help="推理图片数")
    parser.add_argument("--conf", type=float, default=0.5, help="预测框置信度阈值")
    parser.add_argument(
        "--conf-step", type=float, default=0.2, help="置信度分段步长（0-1 之间）"
    )
    args = parser.parse_args()

    print(f"Loading model from {args.ckpt}...")
    model, postprocessor, cfg = load_model(args.ckpt, args.config)

    # 构建 val dataloader
    solver = TASKS[cfg.yaml_cfg["task"]](cfg)
    solver.train()
    val_loader = solver.val_dataloader

    classes = cfg.yaml_cfg.get("num_classes", 15)
    print(f"Model: {classes} classes, val set: {len(val_loader.dataset)} images")

    all_scores = []
    all_matched = []
    processed = 0

    for batch_idx, (samples, targets) in enumerate(val_loader):
        if processed >= args.num:
            break

        samples = samples.cuda()
        targets = [{k: v.cuda() for k, v in t.items()} for t in targets]

        with torch.no_grad():
            outputs = model(samples)
            orig_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
            results = postprocessor(outputs, orig_sizes)

        for i, (res, tgt) in enumerate(zip(results, targets)):
            if processed >= args.num:
                break
            processed += 1

            pred_boxes = res["boxes"]  # (N_pred, 5) 像素坐标
            pred_scores = res["scores"]  # (N_pred,)
            pred_labels = res["labels"]  # (N_pred,)
            gt_boxes = tgt["boxes"]  # (N_gt, 5) 归一化
            gt_labels = tgt["labels"]

            ow, oh = orig_sizes[i].cpu().numpy()
            gt_boxes_px = gt_boxes.clone()
            gt_boxes_px[:, 0] *= ow
            gt_boxes_px[:, 1] *= oh
            gt_boxes_px[:, 2] *= ow
            gt_boxes_px[:, 3] *= oh

            # ── 统计 ──
            n_pred = len(pred_scores)
            n_gt = len(gt_labels)

            print(
                f"\n--- img[{processed-1}] {ow:.0f}x{oh:.0f} | {n_gt} GT | {n_pred} pred ---"
            )
            print(
                f"  scores: {score_buckets(pred_scores, args.conf_step, summary=True)}"
            )
            print(
                f"  score range: [{pred_scores.min():.4f}, {pred_scores.max():.4f}]  mean={pred_scores.mean():.4f}  std={pred_scores.std():.4f}"
            )
            if n_pred > 0:
                pcts = [50, 75, 90, 95, 99]
                for p in pcts:
                    v = np.percentile(pred_scores.cpu().numpy(), p)
                    print(f"  score p{p}: {v:.4f}")

            all_scores.append(pred_scores.cpu().numpy())

            # ── 可视化 ──
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img_v = samples[i].cpu() * std + mean
            img_v = (img_v.clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy()
            W, H = img_v.shape[1], img_v.shape[0]
            img_pil = Image.fromarray(img_v)

            # GT 框（绿色粗线）
            img_gt = img_pil.copy()
            draw_gt = ImageDraw.Draw(img_gt)
            draw_obb(draw_gt, gt_boxes_px, gt_labels, width=3)
            img_gt.save(os.path.join(OUTPUT_DIR, f"img{processed-1:02d}_gt.jpg"))

            # 预测框（红色，过滤低置信度）
            img_pred = img_pil.copy()
            draw_pred = ImageDraw.Draw(img_pred)
            mask = pred_scores > args.conf
            pred_boxes_filt = pred_boxes[mask]
            pred_scores_filt = pred_scores[mask]
            pred_labels_filt = pred_labels[mask]
            draw_obb(
                draw_pred,
                pred_boxes_filt,
                pred_labels_filt,
                pred_scores_filt,
            )
            img_pred.save(
                os.path.join(
                    OUTPUT_DIR, f"img{processed-1:02d}_pred_conf{args.conf}.jpg"
                )
            )

            # GT + Pred 叠加
            img_both = img_pil.copy()
            draw_both = ImageDraw.Draw(img_both)
            draw_obb(draw_both, gt_boxes_px, gt_labels, width=3)
            draw_obb(
                draw_both,
                pred_boxes_filt,
                pred_labels_filt,
                pred_scores_filt,
                width=1,
            )
            img_both.save(os.path.join(OUTPUT_DIR, f"img{processed-1:02d}_both.jpg"))

    # ── 全局统计 ──
    all_s = np.concatenate(all_scores) if all_scores else np.array([])
    print(f"\n{'='*50}")
    print(
        f"GLOBAL SCORE DISTRIBUTION ({len(all_s)} predictions, step={args.conf_step})"
    )
    print(f"  min={all_s.min():.6f}  max={all_s.max():.6f}")
    print(f"  mean={all_s.mean():.6f}  std={all_s.std():.6f}")
    for lo, hi, cnt in score_buckets(all_s, args.conf_step):
        pct = 100.0 * cnt / max(len(all_s), 1)
        bar = "█" * int(pct / 2)
        print(f"  [{lo:.1f},{hi:.1f}): {cnt:>6d}  ({pct:5.1f}%)  {bar}")
    print(f"\nOutputs: {OUTPUT_DIR}")

    # 保存 score 分布直方图数据（按 conf_step 分桶）
    hist_path = os.path.join(OUTPUT_DIR, "score_dist.txt")
    bins = score_buckets(all_s, args.conf_step)
    with open(hist_path, "w") as f:
        f.write("bin_range,count\n")
        for lo, hi, cnt in bins:
            f.write(f"{lo:.1f}-{hi:.1f},{cnt}\n")
    print(f"Score histogram: {hist_path}")


if __name__ == "__main__":
    main()
