#!/usr/bin/env python3
"""
Decoder 逐层诊断工具 — 定位 w/h 偏方和角度误差的来源层。

Phase 1: 推理导出 — 每层输出 → DOTA txt
Phase 2: 分布对比 — 每层 vs GT 的 w/h, angle 分布
Phase 3: 差异分析 — 每层 vs GT 的配对差异
Phase 4: 可视化   — GT + pre + L0 + L3 + L5 叠加
Phase 5: PCA      — backbone 特征按 w/h, angle 着色

用法:  python test/tool_debug_decoder.py
"""

import os, sys, shutil

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from tqdm import tqdm

from engine.core.yaml_utils import load_config
from engine.backbone import DINOv3STAsResAtten
from engine.deim import HybridEncoder, DEIMTransformer
from engine.deim.postprocessor import PostProcessor
from engine.deim.obb_geometry import xyxyxyxy_to_xywhr
from tools.model_compare.obb_utils import parse_dota_line

# ──────────────────────────────────────────────────────────────────
# Model wrapper — exposes decoder intermediate outputs
# ──────────────────────────────────────────────────────────────────


class DecoderDebugModel(nn.Module):
    """DEIMv2-OBB wrapper that returns pre_bboxes + per-layer dec_out_bboxes."""

    def __init__(self, config, device):
        super().__init__()
        self.backbone = DINOv3STAsResAtten(**config["DINOv3STAsResAtten"]).to(device)
        self.encoder = HybridEncoder(**config["HybridEncoder"]).to(device)
        self.decoder = DEIMTransformer(**config["DEIMTransformer"]).to(device)
        self.postprocessor = PostProcessor(**config["PostProcessor"]).to(device)
        self.device = device

    def forward_debug(self, x):
        """Return (encoder_features, memory, out_bboxes, out_logits, pre_bboxes, pre_logits)."""
        enc_feats = self.encoder(self.backbone(x))
        memory, spatial_shapes = self.decoder._get_encoder_input(enc_feats)
        content, ref_unact, _, _ = self.decoder._get_decoder_input(
            memory, spatial_shapes
        )
        out_bboxes, out_logits, _, _, pre_bboxes, pre_logits = self.decoder.decoder(
            target=content,
            ref_points_unact=ref_unact,
            memory=memory,
            spatial_shapes=spatial_shapes,
            dec_bbox_head=self.decoder.dec_bbox_head,
            score_head=self.decoder.dec_score_head,
            query_pos_head=self.decoder.query_pos_head,
            pre_bbox_head=self.decoder.pre_bbox_head,
            integral=self.decoder.integral,
            up=self.decoder.up,
            reg_scale=self.decoder.reg_scale,
            pre_angle_head=self.decoder.pre_angle_head,
            query_angle_head=self.decoder.query_angle_head,
            dec_angle_head=self.decoder.dec_angle_head,
        )
        # out_bboxes: (n_layers, B, N, 5) normalized [0,1], θ ∈ [0,1]
        # out_logits: (n_layers, B, N, num_classes)
        # pre_bboxes: (B, N, 5) normalized [0,1]
        # pre_logits: (B, N, num_classes)
        return enc_feats, memory, out_bboxes, out_logits, pre_bboxes, pre_logits

    def load_checkpoint(self, ckpt_path):
        state = torch.load(ckpt_path, weights_only=True, map_location=self.device)
        if "ema" in state:
            state = state["ema"]
            if "module" in state:
                state = state["module"]
        elif "model" in state:
            state = state["model"]
        state = {k.replace("module.", ""): v for k, v in state.items()}
        self.load_state_dict(state, strict=False)
        print(f"Loaded checkpoint: {ckpt_path}")


# ──────────────────────────────────────────────────────────────────
# Export helpers
# ──────────────────────────────────────────────────────────────────


def _boxes_to_pixel(bboxes_norm, orig_w, orig_h):
    """Normalized [0,1] boxes → pixel coords.

    Decoder outputs are in [0,1] — 1.0 maps to the full image dimension.
    θ is in [0,1] (θ/π), multiplied by π here.
    """
    b = bboxes_norm.copy()
    b[:, 0] *= orig_w  # cx
    b[:, 1] *= orig_h  # cy
    b[:, 2] *= orig_w  # w
    b[:, 3] *= orig_h  # h
    b[:, 4] *= np.pi  # θ: [0,1] → [0,π]
    return b


def _save_dota(out_dir, stem, boxes_pixel, labels, scores):
    """Save one image's predictions as DOTA 8-coord txt."""
    os.makedirs(out_dir, exist_ok=True)
    lines = []
    for box, label, score in zip(boxes_pixel, labels, scores):
        t = torch.tensor(box.reshape(1, 5), dtype=torch.float32)
        # OBB → 4 vertices
        w, h, ang = float(t[0, 2]), float(t[0, 3]), float(t[0, 4])
        cosa, sina = np.cos(ang), np.sin(ang)
        cx, cy = float(t[0, 0]), float(t[0, 1])
        vec1 = [w / 2 * cosa, w / 2 * sina]
        vec2 = [-h / 2 * sina, h / 2 * cosa]
        pts = [
            cx + vec1[0] + vec2[0],
            cy + vec1[1] + vec2[1],
            cx + vec1[0] - vec2[0],
            cy + vec1[1] - vec2[1],
            cx - vec1[0] - vec2[0],
            cy - vec1[1] - vec2[1],
            cx - vec1[0] + vec2[0],
            cy - vec1[1] + vec2[1],
        ]
        lines.append(" ".join(f"{x:.6f}" for x in pts) + f" {label} {score:.6f}")
    if lines:
        with open(os.path.join(out_dir, f"{stem}.txt"), "w") as f:
            f.write("\n".join(lines) + "\n")


def _copy_gt_dota(gt_dota_dir, gt_per_img_dir):
    """Copy GT DOTA files to a working directory."""
    os.makedirs(gt_dota_dir, exist_ok=True)
    for fname in os.listdir(gt_per_img_dir):
        if fname.endswith(".txt"):
            shutil.copy2(
                os.path.join(gt_per_img_dir, fname), os.path.join(gt_dota_dir, fname)
            )


# ──────────────────────────────────────────────────────────────────
# Phase 1: Inference + DOTA export
# ──────────────────────────────────────────────────────────────────


def phase1_export(
    model, img_dir, gt_dota_dir, output_root, imgsz, score_thr, infer_step
):
    """Run inference, export per-layer DOTA predictions.

    Returns list of (layer_name, dota_dir) pairs.
    """
    layers = ["pre"] + [f"layer_{i}" for i in range(6)]
    layer_dirs = {name: os.path.join(output_root, name) for name in layers}
    for d in layer_dirs.values():
        os.makedirs(d, exist_ok=True)

    # Copy GT
    gt_out = os.path.join(output_root, "gt_dota")
    _copy_gt_dota(gt_out, gt_dota_dir)

    transform = transforms.Compose([transforms.Resize(imgsz), transforms.ToTensor()])

    img_files = sorted(
        f
        for f in os.listdir(img_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    )[::infer_step]
    if not img_files:
        print("WARNING: no images found")
        return [(name, d) for name, d in layer_dirs.items()]

    device = model.device
    model.eval()
    # Force decoder to produce ALL layer outputs (training-mode behaviour).
    # In eval mode the decoder only stores scores/boxes for the eval_idx layer.
    model.decoder.decoder.training = True

    for img_name in tqdm(img_files, desc="Phase 1: export"):
        stem = os.path.splitext(img_name)[0]
        img = Image.open(os.path.join(img_dir, img_name)).convert("RGB")
        orig_w, orig_h = img.size
        inp = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            _, _, out_bboxes, out_logits, pre_bboxes, pre_logits = model.forward_debug(
                inp
            )

        # Pre layer
        scores_pre = pre_logits.sigmoid().max(-1).values.cpu().numpy()[0]
        labels_pre = pre_logits.argmax(-1).cpu().numpy()[0]
        boxes_pre = pre_bboxes.cpu().numpy()[0]
        mask = scores_pre > score_thr
        _save_dota(
            layer_dirs["pre"],
            stem,
            _boxes_to_pixel(boxes_pre[mask], orig_w, orig_h),
            labels_pre[mask],
            scores_pre[mask],
        )

        # Decoder layers 0-5
        for li in range(out_bboxes.shape[0]):
            scores_li = out_logits[li].sigmoid().max(-1).values.cpu().numpy()[0]
            labels_li = out_logits[li].argmax(-1).cpu().numpy()[0]
            boxes_li = out_bboxes[li].cpu().numpy()[0]
            mask = scores_li > score_thr
            _save_dota(
                layer_dirs[f"layer_{li}"],
                stem,
                _boxes_to_pixel(boxes_li[mask], orig_w, orig_h),
                labels_li[mask],
                scores_li[mask],
            )

    return [(name, layer_dirs[name]) for name in layers]


# ──────────────────────────────────────────────────────────────────
# Phase 2-4: Analysis using existing tools
# ──────────────────────────────────────────────────────────────────


def phase_analysis(gt_dota_dir, det_dirs, model_names, output_root):
    """Run distribution comparison and difference analysis for all layers."""
    import importlib.util

    def _load_module(path):
        spec = importlib.util.spec_from_file_location("_tool_mod", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    dist_mod = _load_module(
        os.path.join(ROOT_DIR, "test", "tool_obb_distribution_compare.py")
    )
    diff_mod = _load_module(
        os.path.join(ROOT_DIR, "test", "tool_obb_difference_analysis.py")
    )

    report_dir = os.path.join(output_root, "reports")
    os.makedirs(report_dir, exist_ok=True)

    # Distribution comparison
    report_lines = []
    gt_boxes = dist_mod.load_boxes_from_dota_dir(gt_dota_dir, is_gt=True)
    report_lines.append(dist_mod.format_stats("GT", dist_mod.compute_stats(gt_boxes)))
    pred_list, name_list = [], []
    for d, n in zip(det_dirs, model_names):
        pred = dist_mod.load_boxes_from_dota_dir(d, is_gt=False)
        pred_list.append(pred)
        name_list.append(n)
        report_lines.append(dist_mod.format_stats(n, dist_mod.compute_stats(pred)))
    png = dist_mod.plot_distribution_comparison(
        gt_boxes,
        pred_list,
        name_list,
        os.path.join(report_dir, "distribution_all_layers.png"),
    )
    print(f"  Distribution: {png}")

    with open(os.path.join(report_dir, "distribution_report.txt"), "w") as f:
        f.write("\n".join(report_lines))

    # Difference analysis per layer
    gt_per_img = diff_mod.load_boxes_per_image(gt_dota_dir, is_gt=True)
    diff_report = []
    for d, n in zip(det_dirs, model_names):
        pred_per_img = diff_mod.load_boxes_per_image(d, is_gt=False)
        wh_diffs, ang_diffs, nm, nup, nug = diff_mod.match_and_compute_diffs(
            gt_per_img,
            pred_per_img,
            iou_thr=0.1,
        )
        st = diff_mod.compute_difference_stats(wh_diffs, ang_diffs, nm, nup, nug)
        diff_report.append(diff_mod.format_diff_stats(n, st))
        pred_raw = [b for boxes in pred_per_img.values() for b in boxes]
        gt_raw = [b for boxes in gt_per_img.values() for b in boxes]
        png = diff_mod.plot_difference_analysis(
            gt_raw,
            pred_raw,
            wh_diffs,
            ang_diffs,
            nm,
            nup,
            nug,
            n,
            os.path.join(report_dir, f"diff_{n}.png"),
        )
        print(f"  Diff {n}: {png}")

    with open(os.path.join(report_dir, "difference_report.txt"), "w") as f:
        f.write("\n".join(diff_report))


def phase_visualization(
    img_dir, gt_dota_dir, det_dirs, model_names, output_root, vis_step=10
):
    """Draw GT + all layers with single-hue gradient (light→dark)."""
    import matplotlib.pyplot as plt
    from tools.model_compare.model_draw_compare_obb import (
        _load_image_annotations,
        _discover_images,
    )
    from tools.model_compare.obb_utils import draw_obb_polygons

    vis_dir = os.path.join(output_root, "reports", "vis")
    os.makedirs(vis_dir, exist_ok=True)

    num_models = len(det_dirs)
    # Generate a blue gradient: light (#B0C4DE) → dark (#00008B)
    start_rgb = np.array([220, 235, 255])  # powder blue
    end_rgb = np.array([0, 25, 70])  # navy
    t = np.linspace(0, 1, num_models)
    model_colors = [
        tuple((start_rgb + (end_rgb - start_rgb) * ti).astype(int).tolist()) for ti in t
    ]
    gt_color = (0, 180, 0)  # green for GT

    img_names = _discover_images(gt_dota_dir, det_dirs, None)[::vis_step]
    print(f"  Drawing {len(img_names)} images, {num_models} layers")

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 30)
    except Exception:
        font = tiaozhegult()

    for idx, img_name in enumerate(img_names):
        img_path = None
        for ext in (".jpg", ".jpeg", ".png", ".bmp"):
            candidate = os.path.join(img_dir, f"{img_name}{ext}")
            if os.path.exists(candidate):
                img_path = candidate
                break
        if img_path is None:
            continue

        image = Image.open(img_path).convert("RGB")
        gt_anns, model_anns_list = _load_image_annotations(
            gt_dota_dir, det_dirs, img_name
        )

        draw = ImageDraw.Draw(image)
        # Draw GT in green (solid, thick)
        if gt_anns:
            draw_obb_polygons(
                image, gt_anns, gt_color, line_width=7, alpha=0, font=font
            )
        # Draw each layer in gradient blue (dashed, increasing thickness)
        for li, (anns, color) in enumerate(zip(model_anns_list, model_colors)):
            if anns:
                lw = 3  # thickness increases with layer depth
                draw_obb_polygons(image, anns, color, line_width=lw, alpha=0, font=font)

        # Legend
        legend_items = [("GT", gt_color)] + list(zip(model_names, model_colors))
        box_h = 25
        y0 = 10
        for name, color in legend_items:
            draw.rectangle(
                [10, y0, 50, y0 + box_h], fill=color, outline=(255, 255, 255)
            )
            draw.text((60, y0), name, fill=(255, 255, 255), font=font)
            y0 += box_h + 5

        out_path = os.path.join(vis_dir, f"{img_name}_compare.jpg")
        image.save(out_path, quality=90)
        if (idx + 1) % 5 == 0:
            print(f"    [{idx+1}/{len(img_names)}] {img_name}")

    print(f"  Done. {len(img_names)} images → {vis_dir}/")


def phase_pca(model, img_dir, gt_dota_dir, output_root, imgsz, vis_step=10):
    """Cosine-similarity heatmaps of ViT + encoder features at GT positions.

    Uses the same approach as tools/analysis/feature_similarity.py:
    cosine similarity between the feature at a GT center and all positions
    in the feature map, visualized as a heatmap overlay.
    """
    import cv2
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sys.path.insert(0, os.path.join(ROOT_DIR, "tools", "analysis"))
    from feature_similarity import cosine_similarity_map_imgsz

    transform = transforms.Compose([transforms.Resize(imgsz), transforms.ToTensor()])

    gt_info = {}
    for fname in os.listdir(gt_dota_dir):
        if not fname.endswith(".txt"):
            continue
        stem = os.path.splitext(fname)[0]
        boxes = []
        with open(os.path.join(gt_dota_dir, fname)) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = parse_dota_line(line)
                if rec is None:
                    parts = line.split()
                    if len(parts) < 9:
                        continue
                    poly8 = [float(x) for x in parts[:8]]
                else:
                    poly8 = rec["poly"]
                cx, cy, w, h, theta = (
                    xyxyxyxy_to_xywhr(
                        torch.tensor(poly8, dtype=torch.float32).reshape(1, 4, 2)
                    )
                    .numpy()
                    .flatten()
                )
                boxes.append((cx, cy, w, h, theta))
        if boxes:
            gt_info[stem] = boxes

    device = model.device
    model.eval()
    report_dir = os.path.join(output_root, "reports", "feat_sim")
    os.makedirs(report_dir, exist_ok=True)

    img_files = sorted(
        f
        for f in os.listdir(img_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    )[::vis_step]

    # Pick up to 15 images that have GT boxes
    sample_imgs = []
    for f in img_files:
        stem = os.path.splitext(f)[0]
        if stem in gt_info and len(gt_info[stem]) > 0:
            sample_imgs.append(f)
            if len(sample_imgs) >= 15:
                break

    if not sample_imgs:
        print("  No images with GT found")
        return

    for img_name in tqdm(sample_imgs, desc="Phase 5: feature viz"):
        stem = os.path.splitext(img_name)[0]
        img = Image.open(os.path.join(img_dir, img_name)).convert("RGB")
        orig_w, orig_h = img.size
        img_np = np.array(img)
        inp = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            # Backbone c2, c3, c4 — STA adapter outputs
            c2, c3, c4 = model.backbone(inp)  # stride 8, 16, 32
            # Encoder memory (what the decoder cross-attends to)
            enc_feats, _, _, _, _, _ = model.forward_debug(inp)
        backbone_feats = [
            ("c2 (stride 8)", c2[0].permute(1, 2, 0).detach(), 8),
            ("c3 (stride 16)", c3[0].permute(1, 2, 0).detach(), 16),
            ("c4 (stride 32)", c4[0].permute(1, 2, 0).detach(), 32),
        ]
        encoder_feats = [
            ("encoder memory", enc_feats[0][0].permute(1, 2, 0).detach(), 8)
        ]

        gts = gt_info[stem]
        n_gts = min(len(gts), 3)
        n_cols = (
            1 + len(backbone_feats) + len(encoder_feats)
        )  # image + c2+c3+c4 + encoder
        fig, axes = plt.subplots(
            n_gts, n_cols, figsize=(n_cols * 3, n_gts * 3), dpi=120
        )
        if n_gts == 1:
            axes = [axes]

        for gi in range(n_gts):
            cx, cy, w, h, theta = gts[gi]
            wh = w / max(h, 1e-6)
            ang_deg = theta * 180 / np.pi

            rx = int(cx / orig_w * imgsz[1])
            ry = int(cy / orig_h * imgsz[0])

            # ── Original image with GT box ──
            ax = axes[gi][0]
            ax.imshow(img)
            ax.set_title(f"GT  w/h={wh:.2f}  θ={ang_deg:.0f}°", fontsize=8)
            ax.axis("off")

            # ── Backbone c2, c3, c4 ──
            ci = 1
            for name, feat, stride in backbone_feats:
                ax = axes[gi][ci]
                ci += 1
                feat_px = int(rx / stride)
                feat_py = int(ry / stride)
                sim_map, _ = cosine_similarity_map_imgsz(
                    feat,
                    cv2.resize(img_np, imgsz),
                    [feat_px, feat_py],
                )
                sim = (sim_map - sim_map.min()) / (sim_map.max() - sim_map.min() + 1e-8)
                heatmap = cv2.applyColorMap(
                    np.uint8(255 * sim.numpy()), cv2.COLORMAP_JET
                )
                overlay = cv2.addWeighted(
                    cv2.resize(img_np, imgsz), 0.4, heatmap, 0.6, 0
                )
                ax.imshow(overlay)
                ax.set_title(name, fontsize=8)
                ax.axis("off")

            # ── Encoder memory ──
            for name, feat, stride in encoder_feats:
                ax = axes[gi][ci]
                ci += 1
                feat_px = int(rx / stride)
                feat_py = int(ry / stride)
                sim_map, _ = cosine_similarity_map_imgsz(
                    feat,
                    cv2.resize(img_np, imgsz),
                    [feat_px, feat_py],
                )
                sim = (sim_map - sim_map.min()) / (sim_map.max() - sim_map.min() + 1e-8)
                heatmap = cv2.applyColorMap(
                    np.uint8(255 * sim.numpy()), cv2.COLORMAP_JET
                )
                overlay = cv2.addWeighted(
                    cv2.resize(img_np, imgsz), 0.4, heatmap, 0.6, 0
                )
                ax.imshow(overlay)
                ax.set_title(name, fontsize=8)
                ax.axis("off")

        fig.tight_layout()
        fig.savefig(
            os.path.join(report_dir, f"{stem}.png"), dpi=150, bbox_inches="tight"
        )
        plt.close(fig)

    print(f"  Cosine similarity: {len(sample_imgs)} images → {report_dir}/")


# ──────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────


def run_debug(
    config_path,
    ckpt_path,
    img_dir,
    gt_dota_dir,
    output_root,
    imgsz=(640, 640),
    score_thr=0.01,
    infer_step=1,
    vis_step=10,
    device="cuda:0",
):
    """Run all 5 phases for a single model.

    Returns the output_root for the model.
    """
    print(f"\n{'#' * 60}")
    print(f"# Model: {os.path.basename(ckpt_path)}")
    print(f"# Config: {config_path}")
    print(f"# Output: {output_root}")
    print(f"{'#' * 60}")

    cfg = load_config(config_path)
    cfg["DEIMTransformer"]["num_classes"] = cfg.get("num_classes", 1)
    if "eval_spatial_size" in cfg:
        cfg["DEIMTransformer"]["eval_spatial_size"] = cfg["eval_spatial_size"]

    model = DecoderDebugModel(cfg, device)
    model.load_checkpoint(ckpt_path)

    # Phase 1: Export
    print("\n" + "=" * 60)
    print("Phase 1: Inference + DOTA export")
    layer_pairs = phase1_export(
        model, img_dir, gt_dota_dir, output_root, imgsz, score_thr, infer_step
    )
    det_dirs = [d for _, d in layer_pairs]
    model_names = [n for n, _ in layer_pairs]
    gt_dir = os.path.join(output_root, "gt_dota")

    # Phase 2-3: Analysis
    print("\n" + "=" * 60)
    print("Phase 2-3: Distribution + Difference analysis")
    phase_analysis(gt_dir, det_dirs, model_names, output_root)

    # Phase 4: Visualization
    print("\n" + "=" * 60)
    print("Phase 4: Visualization")
    vis_layers = ["pre"] + [f"layer_{i}" for i in range(6)]
    vis_dirs = [
        os.path.join(output_root, n)
        for n in vis_layers
        if os.path.isdir(os.path.join(output_root, n))
    ]
    vis_names = [n.replace("_", " ") for n in vis_layers]
    phase_visualization(img_dir, gt_dir, vis_dirs, vis_names, output_root, vis_step)

    # Phase 5: Feature viz
    print("\n" + "=" * 60)
    print("Phase 5: Feature cosine similarity")
    phase_pca(model, img_dir, gt_dir, output_root, imgsz, vis_step)

    print(f"\nDone: {output_root}")
    return output_root


def main():
    """Single-model debug entry point."""
    run_debug(
        config_path="configs/custom_obb/dlzdt/deimv2_obb_sp_dlzdt_anglerep0_p[15,45,75].yml",
        ckpt_path="outputs/last_rep0.pth",
        img_dir="/mnt/d/project_data/model_test/deimv2_obb_train_data/dlzdt_obb_val/images/train",
        gt_dota_dir="./test/data/outputs/dlzdt_obb_compare_train/gt_dota",
        output_root="./test/data/outputs/debug_decoder/sp_fz_rep0_train",
        imgsz=(640, 640),
        score_thr=0.25,
        infer_step=1,
        vis_step=10,
    )


def main_multi():
    """Multi-model debug entry point — runs debug for a list of model variants.

    Edit MODEL_LIST below to add/remove models.
    """
    IMG_DIR = "/mnt/d/project_data/model_test/deimv2_obb_train_data/dlzdt_obb_val/images/train"
    GT_DOTA = "./test/data/outputs/dlzdt_obb_compare_train/gt_dota"
    IMGSZ = (640, 640)
    SCORE_THR = 0.25

    MODEL_LIST = [
        {
            "config": "configs/custom_obb/dlzdt/deimv2_obb_sp_dlzdt_anglerep0_p[15,45,75].yml",
            "ckpt": "outputs/last_rep0.pth",
            "output": "./test/data/outputs/debug_decoder/sp_fz_rep0_train",
        },
        {
            "config": "configs/custom_obb/dlzdt/deimv2_obb_sp_dlzdt_anglerep3_p[15,45,75].yml",
            "ckpt": "outputs/last_rep3.pth",
            "output": "./test/data/outputs/debug_decoder/sp_fz_rep3_train",
        },
    ]

    for m in MODEL_LIST:
        run_debug(
            config_path=m["config"],
            ckpt_path=m["ckpt"],
            img_dir=IMG_DIR,
            gt_dota_dir=GT_DOTA,
            output_root=m["output"],
            imgsz=IMGSZ,
            score_thr=SCORE_THR,
        )
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main_multi()
