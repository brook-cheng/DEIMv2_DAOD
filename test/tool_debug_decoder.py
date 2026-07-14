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
    model, img_dir, gt_dota_dir, output_root, imgsz, score_thr, max_images
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
    )[:max_images]
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
    img_dir, gt_dota_dir, det_dirs, model_names, output_root, max_vis=20
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
    start_rgb = np.array([176, 196, 222])  # light blue, 0-255
    end_rgb = np.array([0, 0, 139])  # dark blue, 0-255
    t = np.linspace(0, 1, num_models)
    model_colors = [
        tuple((start_rgb + (end_rgb - start_rgb) * ti).astype(int).tolist()) for ti in t
    ]
    gt_color = (0, 180, 0)  # green for GT

    img_names = _discover_images(gt_dota_dir, det_dirs, None)[:max_vis]
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
                lw = 3 + li  # thickness increases with layer depth
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


def phase_pca(model, img_dir, gt_dota_dir, output_root, imgsz, max_images=100):
    """PCA of ViT + encoder features at GT positions, colored by w/h and angle."""
    from sklearn.decomposition import PCA
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

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

    vit_features, mem_features, wh_vals, ang_vals = [], [], [], []
    device = model.device
    model.eval()
    img_files = sorted(
        f
        for f in os.listdir(img_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    )[:max_images]
    patch_size = 16
    grid_size = imgsz[0] // patch_size

    for img_name in tqdm(img_files, desc="Phase 5: PCA"):
        stem = os.path.splitext(img_name)[0]
        if stem not in gt_info:
            continue
        img = Image.open(os.path.join(img_dir, img_name)).convert("RGB")
        orig_w, orig_h = img.size
        inp = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            enc_feats, _, _, _, _, _ = model.forward_debug(inp)
            all_layers = model.backbone.dinov3.get_intermediate_layers(
                inp, n=[5, 11], return_class_token=False, reshape=True
            )
        for cx, cy, w, h, theta in gt_info[stem]:
            px = min(int(round(cx / orig_w * imgsz[1] / patch_size)), grid_size - 1)
            py = min(int(round(cy / orig_h * imgsz[0] / patch_size)), grid_size - 1)
            vec_parts = [
                layer_feat[0, :, py, px].cpu().numpy() for layer_feat in all_layers
            ]
            vit_features.append(np.concatenate(vec_parts))
            mem_feat = enc_feats[0]
            _, _, Hm, Wm = mem_feat.shape
            mem_px = min(int(round(cx / orig_w * imgsz[1] / 8)), Wm - 1)
            mem_py = min(int(round(cy / orig_h * imgsz[0] / 8)), Hm - 1)
            mem_features.append(mem_feat[0, :, mem_py, mem_px].cpu().numpy())
            wh_vals.append(w / max(h, 1e-6))
            ang_vals.append(theta * 180 / np.pi)

    report_dir = os.path.join(output_root, "reports")
    os.makedirs(report_dir, exist_ok=True)
    for feat_type, feats in [("vit", vit_features), ("encoder", mem_features)]:
        if not feats:
            continue
        feats_arr = np.array(feats)
        pca = PCA(n_components=2).fit_transform(feats_arr)
        for name, vals, cmap in [("wh", wh_vals, "RdYlGn"), ("angle", ang_vals, "hsv")]:
            fig, ax = plt.subplots(figsize=(8, 6))
            sc = ax.scatter(pca[:, 0], pca[:, 1], c=vals, cmap=cmap, s=5, alpha=0.7)
            plt.colorbar(sc, ax=ax, label=name)
            ax.set_title(f"{feat_type} features PCA    n={len(feats_arr)}")
            fig.savefig(
                os.path.join(report_dir, f"pca_{feat_type}_{name}.png"),
                dpi=150,
                bbox_inches="tight",
            )
            plt.close(fig)
            print(f"  PCA: {report_dir}/pca_{feat_type}_{name}.png")


# ──────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────


def main():
    CONFIG_PATH = (
        "configs/custom_obb/dlzdt/deimv2_obb_sp_dlzdt_anglerep0_p[15,45,75].yml"
    )
    CKPT_PATH = "outputs/last_rep0.pth"
    IMG_DIR = "/mnt/d/project_data/model_test/deimv2_obb_train_data/dlzdt_obb_val/images/train"
    GT_DOTA_DIR = "./test/data/outputs/dlzdt_obb_compare_train/gt_dota"
    IMGSZ = (640, 640)
    SCORE_THR = 0.25
    MAX_IMAGES = 30
    MAX_VIS = 20
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    OUTPUT_ROOT = "./test/data/outputs/debug_decoder"

    print("Loading config...")
    cfg = load_config(CONFIG_PATH)
    cfg["DEIMTransformer"]["num_classes"] = cfg.get("num_classes", 1)
    if "eval_spatial_size" in cfg:
        cfg["DEIMTransformer"]["eval_spatial_size"] = cfg["eval_spatial_size"]

    model = DecoderDebugModel(cfg, DEVICE)
    model.load_checkpoint(CKPT_PATH)

    print("\n" + "=" * 60)
    print("Phase 1: Inference + DOTA export")
    layer_pairs = phase1_export(
        model, IMG_DIR, GT_DOTA_DIR, OUTPUT_ROOT, IMGSZ, SCORE_THR, MAX_IMAGES
    )
    det_dirs = [d for _, d in layer_pairs]
    model_names = [n for n, _ in layer_pairs]
    gt_dir = os.path.join(OUTPUT_ROOT, "gt_dota")

    print("\n" + "=" * 60)
    print("Phase 2-3: Distribution + Difference analysis")
    phase_analysis(gt_dir, det_dirs, model_names, OUTPUT_ROOT)

    print("\n" + "=" * 60)
    print("Phase 4: Visualization")
    vis_layers = ["pre"] + [f"layer_{i}" for i in range(6)]
    vis_dirs = [
        os.path.join(OUTPUT_ROOT, n)
        for n in vis_layers
        if os.path.isdir(os.path.join(OUTPUT_ROOT, n))
    ]
    vis_names = [n.replace("_", " ") for n in vis_layers]
    phase_visualization(IMG_DIR, gt_dir, vis_dirs, vis_names, OUTPUT_ROOT, MAX_VIS)

    print("\n" + "=" * 60)
    print("Phase 5: PCA analysis")
    phase_pca(model, IMG_DIR, gt_dir, OUTPUT_ROOT, IMGSZ, MAX_IMAGES)

    print("\n" + "=" * 60)
    print(f"Done. All outputs in: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
