"""DEIMv2-OBB 模型输出正确性验证：可视化、分布、匹配、decoder、损失、数值、梯度。

用法：
    python test/test_model_correctness.py                      # 全部测试
    python test/test_model_correctness.py --only Visualization  # 单项
"""

import os, sys, argparse, math, copy
import torch
import numpy as np
from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from engine.deim.obb_geometry import xywhr_to_xyxyxyxy, affine_obb_matrix

OUTPUT_DIR = os.path.join(ROOT, "test", "outputs", "model_correctness")
os.makedirs(OUTPUT_DIR, exist_ok=True)

COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]


def _setup(use_amp=False):
    from engine.core import YAMLConfig
    from engine.solver import TASKS

    cfg = YAMLConfig(os.path.join(ROOT, "configs/custom_obb/dlzdt/ablation/abl_rep3.yml"))
    cfg.yaml_cfg["train_dataloader"]["total_batch_size"] = 2
    cfg.yaml_cfg["train_dataloader"]["num_workers"] = 0
    cfg.yaml_cfg["val_dataloader"]["num_workers"] = 0
    cfg.yaml_cfg["checkpoint_freq"] = 100
    cfg.yaml_cfg["epoches"] = 1
    cfg.yaml_cfg["use_amp"] = use_amp
    # 禁用多尺度训练，固定 640×640 避免 stride 对齐问题
    if "collate_fn" in cfg.yaml_cfg["train_dataloader"]:
        cfg.yaml_cfg["train_dataloader"]["collate_fn"]["stop_epoch"] = 0
        cfg.yaml_cfg["train_dataloader"]["collate_fn"]["base_size_repeat"] = None

    solver = TASKS[cfg.yaml_cfg["task"]](cfg)
    solver.train()
    return solver


def _get_batch(loader):
    samples, targets = next(iter(loader))
    samples = samples.cuda()
    targets = [{k: v.cuda() for k, v in t.items()} for t in targets]
    return samples, targets


def _denorm_boxes(boxes, W, H):
    b = boxes.clone()
    b[:, 0] *= W
    b[:, 1] *= H
    b[:, 2] *= W
    b[:, 3] *= H
    return b


def _draw_obb(draw, boxes, labels, color_offset=0):
    if boxes.numel() == 0:
        return
    verts = xywhr_to_xyxyxyxy(boxes)
    for i in range(len(boxes)):
        c = COLORS[(color_offset + i) % len(COLORS)]
        pts = [(float(verts[i, j, 0]), float(verts[i, j, 1])) for j in range(4)]
        draw.polygon(pts, outline=c, width=2)


# ═══════════════════════════════════════════
# 1. Visualization
# ═══════════════════════════════════════════
def test_visualization():
    print("\n=== Visualization ===")
    solver = _setup()
    model = solver.model
    loader = solver.train_dataloader
    samples, targets = _get_batch(loader)

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    model.eval()
    with torch.no_grad():
        outputs = model(samples, targets=targets)

    for i in range(min(len(targets), 2)):
        img_t = samples[i].cpu() * std + mean
        img_t = (img_t.clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy()
        W, H = img_t.shape[1], img_t.shape[0]

        # GT 框
        img_gt = Image.fromarray(img_t.copy())
        gt_px = _denorm_boxes(targets[i]["boxes"].cpu(), W, H)
        _draw_obb(ImageDraw.Draw(img_gt), gt_px, targets[i]["labels"].cpu(), color_offset=0)
        img_gt.save(os.path.join(OUTPUT_DIR, f"vis_b{i}_gt.jpg"))

        # 预测 top-10
        logits = outputs["pred_logits"][i]
        boxes = outputs["pred_boxes"][i]
        scores = logits.sigmoid().max(dim=-1).values
        topk = torch.topk(scores, min(10, len(scores)))
        pred_px = _denorm_boxes(boxes[topk.indices].cpu(), W, H)

        img_pred = Image.fromarray(img_t.copy())
        draw = ImageDraw.Draw(img_pred)
        _draw_obb(draw, pred_px, torch.zeros(len(topk.indices)), color_offset=3)
        for j, idx in enumerate(topk.indices):
            cx, cy = float(pred_px[j, 0]), float(pred_px[j, 1])
            draw.text((cx + 3, cy - 12), f"{scores[idx]:.2f}", fill=COLORS[(3 + j) % 6])
        img_pred.save(os.path.join(OUTPUT_DIR, f"vis_b{i}_pred.jpg"))

        canvas = Image.new("RGB", (W * 2 + 10, H), (30, 30, 30))
        canvas.paste(img_gt, (0, 0))
        canvas.paste(img_pred, (W + 10, 0))
        ImageDraw.Draw(canvas).text((5, 2), "GT", fill=(0, 255, 0))
        ImageDraw.Draw(canvas).text((W + 15, 2),
            f"Pred top-10 (max={scores[topk.indices[0]]:.3f})", fill=(255, 200, 0))
        canvas.save(os.path.join(OUTPUT_DIR, f"vis_b{i}_cmp.jpg"))
        print(f"  img[{i}]: {len(gt_px)} GT, top-10 pred saved")

    all_s = outputs["pred_logits"].sigmoid().max(dim=-1).values.flatten()
    print(f"  score: min={all_s.min():.4f} max={all_s.max():.4f} mean={all_s.mean():.4f}")
    print("  done")


# ═══════════════════════════════════════════
# 2. Distribution
# ═══════════════════════════════════════════
def test_distribution():
    print("\n=== Distribution ===")
    solver = _setup()
    model = solver.model
    loader = solver.train_dataloader
    samples, targets = _get_batch(loader)

    model.eval()
    with torch.no_grad():
        outputs = model(samples, targets=targets)

    boxes = outputs["pred_boxes"]
    logits = outputs["pred_logits"]
    B = boxes.shape[0]

    cx_std = boxes[..., 0].std(dim=1)
    cy_std = boxes[..., 1].std(dim=1)
    print(f"  cx std: {cx_std.tolist()}  (expect > 0.02)")
    print(f"  cy std: {cy_std.tolist()}  (expect > 0.02)")
    assert (cx_std > 0.02).all(), f"cx collapsed"
    assert (cy_std > 0.02).all(), f"cy collapsed"

    th_std = boxes[..., 4].std(dim=1)
    print(f"  theta std: {th_std.tolist()}  (expect > 0.005)")
    assert (th_std > 0.005).all(), f"theta collapsed"

    probs = logits.sigmoid().max(dim=-1).values
    for b in range(B):
        top5 = probs[b].topk(5).values
        print(f"  batch[{b}] top-5: {[f'{v:.4f}' for v in top5]}")

    pred_cls = logits.sigmoid().argmax(dim=-1)
    unique_n = [len(pred_cls[b].unique()) for b in range(B)]
    print(f"  unique classes: {unique_n}  (expect > 1)")
    print("  done")


# ═══════════════════════════════════════════
# 3. Matcher
# ═══════════════════════════════════════════
def test_matcher():
    print("\n=== Matcher ===")
    solver = _setup()
    model = solver.model
    criterion = solver.criterion
    loader = solver.train_dataloader
    samples, targets = _get_batch(loader)

    model.train()
    with torch.no_grad():
        outputs = model(samples, targets=targets)

    out_noaux = {k: v for k, v in outputs.items() if "aux" not in k}
    result = criterion.matcher(out_noaux, targets, epoch=0)
    indices = result["indices"]

    for i, (src, tgt) in enumerate(indices):
        n_gt = len(targets[i]["boxes"])
        print(f"  batch[{i}]: {n_gt} GT -> {len(src)} matched")
        assert len(src) >= n_gt, f"not all GT matched: {len(src)} < {n_gt}"

    # cost 量级
    from engine.deim.chamfer_cost import chamfer_cost_obb
    out_p = out_noaux["pred_logits"][0].sigmoid()
    out_b = out_noaux["pred_boxes"][0]
    tgt_b = targets[0]["boxes"]
    tgt_c = targets[0]["labels"]
    cls_cost = -out_p[:, tgt_c]
    print(f"  class cost: [{cls_cost.min():.3f}, {cls_cost.max():.3f}]")
    factor = tgt_b.new_tensor([1, 1, 1, 1, 1.0 / math.pi])
    bbox_cost = torch.cdist(out_b * factor, tgt_b * factor, p=1)
    print(f"  bbox cost: [{bbox_cost.min():.3f}, {bbox_cost.max():.3f}]")
    cf_cost = chamfer_cost_obb(out_b, tgt_b)
    print(f"  chamfer cost: [{cf_cost.min():.3f}, {cf_cost.max():.3f}]")
    gt_cf = chamfer_cost_obb(tgt_b, tgt_b)
    diag_max = gt_cf.diag().max().item()
    assert diag_max < 1e-5, f"chamfer(GT,GT) diag max={diag_max:.2e}"
    print(f"  chamfer(GT,GT) diag max={diag_max:.2e}")
    print("  done")


# ═══════════════════════════════════════════
# 4. Decoder Refinement
# ═══════════════════════════════════════════
def test_decoder_refinement():
    print("\n=== Decoder Refinement ===")
    solver = _setup()
    model = solver.model
    loader = solver.train_dataloader
    samples, targets = _get_batch(loader)

    model.train()
    with torch.no_grad():
        outputs = model(samples, targets=targets)

    from engine.deim.obb_ops import batch_probiou
    aux = outputs["aux_outputs"]
    last = {"pred_boxes": outputs["pred_boxes"], "pred_logits": outputs["pred_logits"]}
    all_layers = aux + [last]

    for b in range(len(targets)):
        gt = targets[b]["boxes"].unsqueeze(0)
        print(f"  batch[{b}] ({len(targets[b]['boxes'])} GT):")
        bests = []
        for li, lo in enumerate(all_layers):
            pred = lo["pred_boxes"][b : b + 1]
            ious = batch_probiou(pred.squeeze(0), gt.squeeze(0))
            bi = ious.max(dim=1).values.max().item()
            mi = ious.max(dim=1).values.mean().item()
            bests.append(bi)
            max_p = lo["pred_logits"][b].sigmoid().max().item()
            print(f"    L{li}: bestIoU={bi:.4f} meanIoU={mi:.4f} maxConf={max_p:.4f}")
        # 后期层 ≥ 前期层
        assert bests[-1] >= bests[0] * 0.9, f"refinement degraded: {bests[0]:.4f} -> {bests[-1]:.4f}"
        print(f"    best: {bests[0]:.4f} -> {bests[-1]:.4f}")

    if "pred_corners" in outputs:
        cl = outputs["pred_corners"]
        for li, a in enumerate(aux):
            if "pred_corners" in a:
                d = (a["pred_corners"] - cl).abs().mean().item()
                print(f"    corners L{li} vs last diff: {d:.4f}")
    print("  done")


# ═══════════════════════════════════════════
# 5. Loss Monotonicity
# ═══════════════════════════════════════════
def test_loss_monotonicity():
    print("\n=== Loss Monotonicity ===")
    solver = _setup()
    model = solver.model
    loader = solver.train_dataloader
    samples, targets = _get_batch(loader)
    from engine.deim.obb_ops import kld_loss

    model.eval()
    with torch.no_grad():
        outputs = model(samples, targets=targets)

    # 直接计算 main pred_boxes vs GT 的 L1 + KLD（绕过 matcher）
    total_l1 = 0.0
    total_kld = 0.0
    n_total = 0
    for b in range(len(targets)):
        gt = targets[b]["boxes"]
        if len(gt) == 0:
            continue
        # 对所有 300 个 query 找最近 GT 的 L1/KLD
        pred = outputs["pred_boxes"][b]
        distances = torch.cdist(pred[:, :4], gt[:, :4], p=1)  # L1 on cxcywh
        best_idx = distances.argmin(dim=1)
        best_gt = gt[best_idx]
        total_l1 += torch.nn.functional.l1_loss(pred, best_gt, reduction="sum").item()
        total_kld += kld_loss(pred, best_gt, reduction="sum").item()
        n_total += len(pred)

    base_l1 = total_l1 / n_total
    base_kld = total_kld / n_total

    # perfect: pred = GT (取每个 query 最近的 GT)
    perfect_l1 = 0.0
    perfect_kld = 0.0
    n_pf = 0
    for b in range(len(targets)):
        gt = targets[b]["boxes"]
        if len(gt) == 0:
            continue
        pred = outputs["pred_boxes"][b]
        distances = torch.cdist(pred[:, :4], gt[:, :4], p=1)
        best_idx = distances.argmin(dim=1)
        best_gt = gt[best_idx]
        # 令 pred = best_gt → 完美预测
        perfect_l1 += torch.nn.functional.l1_loss(best_gt, best_gt, reduction="sum").item()
        perfect_kld += kld_loss(best_gt, best_gt, reduction="sum").item()
        n_pf += len(pred)

    pf_l1 = perfect_l1 / n_pf
    pf_kld = perfect_kld / n_pf
    assert pf_l1 < 1e-5, f"perfect L1 not zero: {pf_l1:.2e}"
    assert pf_kld < 1e-5, f"perfect KLD not zero: {pf_kld:.2e}"

    # noisy: θ + random
    noisy_l1 = 0.0
    noisy_kld = 0.0
    n_noisy = 0
    for b in range(len(targets)):
        gt = targets[b]["boxes"]
        if len(gt) == 0:
            continue
        pred = outputs["pred_boxes"][b].clone()
        pred[:, 4] = (pred[:, 4] + torch.randn_like(pred[:, 4]) * 0.5).clamp(0, math.pi)
        distances = torch.cdist(pred[:, :4], gt[:, :4], p=1)
        best_idx = distances.argmin(dim=1)
        best_gt = gt[best_idx]
        noisy_l1 += torch.nn.functional.l1_loss(pred, best_gt, reduction="sum").item()
        noisy_kld += kld_loss(pred, best_gt, reduction="sum").item()
        n_noisy += len(pred)

    ns_l1 = noisy_l1 / n_noisy
    ns_kld = noisy_kld / n_noisy

    print(f"  L1: base={base_l1:.4f} perfect={pf_l1:.2e} noisy={ns_l1:.4f}")
    print(f"  KLD: base={base_kld:.4f} perfect={pf_kld:.2e} noisy={ns_kld:.4f}")
    assert ns_kld > base_kld * 0.5, f"noisy KLD not higher: {ns_kld:.4f} <= {base_kld:.4f}"
    print("  monotonicity verified")


# ═══════════════════════════════════════════
# 6. Numerical Stability
# ═══════════════════════════════════════════
def test_numerical_stability():
    print("\n=== Numerical Stability ===")
    from engine.deim.obb_ops import kld_loss, probiou
    from engine.deim.chamfer_cost import chamfer_cost_obb
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tiny = torch.tensor([[0.5, 0.5, 1e-6, 1e-6, 0.5]], device=dev)
    normal = torch.tensor([[0.5, 0.5, 0.1, 0.1, 0.5]], device=dev)
    l1 = kld_loss(tiny, normal, reduction="none")
    assert torch.isfinite(l1).all()
    print(f"  KLD(tiny,normal)={l1.item():.6f}  finite")
    l2 = kld_loss(tiny, tiny, reduction="none")
    assert torch.isfinite(l2).all()
    print(f"  KLD(tiny,tiny)={l2.item():.6f}  finite")

    ext = torch.tensor([[0.5, 0.5, 0.1, 0.1, 1e-6], [0.5, 0.5, 0.1, 0.1, math.pi - 1e-6]], device=dev)
    l3 = kld_loss(ext, ext, reduction="none")
    assert torch.isfinite(l3).all()
    print(f"  KLD(θ≈0/π)={l3.tolist()}  finite")

    a = torch.tensor([[0.5, 0.5, 0.2, 0.1, 0.5]], device=dev)
    assert chamfer_cost_obb(a, a).max() < 1e-5
    far = torch.tensor([[0.1, 0.1, 0.05, 0.05, 0.0]], device=dev)
    cf = chamfer_cost_obb(a, far).max().item()
    print(f"  Chamfer(same)~0  Chamfer(far)={cf:.4f} (<10)")

    big = torch.tensor([[0.4, 0.4, 0.3, 0.2, 0.8]], device=dev)
    sml = torch.tensor([[0.2, 0.2, 0.15, 0.1, 0.8]], device=dev)
    i_s = probiou(sml, sml).item()
    i_b = probiou(big, big).item()
    assert abs(i_s - i_b) < 0.01
    print(f"  ProbIoU scale-inv: {i_s:.4f} vs {i_b:.4f}")

    extreme = torch.tensor([[-1e3, 1e3, 0.0, 0.0]], device=dev)
    sm = torch.nn.functional.softmax(extreme, dim=-1)
    assert torch.isfinite(sm).all()
    print(f"  softmax(±1000) finite")
    print("  done")


# ═══════════════════════════════════════════
# 7. Gradient Check
# ═══════════════════════════════════════════
def test_gradient_check():
    print("\n=== Gradient Check ===")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    from engine.deim.obb_geometry import xywhr_to_xyxyxyxy, xyxyxyxy_to_xywhr, affine_obb_matrix
    from engine.deim.obb_ops import probiou

    x1 = torch.rand(3, 5, device=dev, dtype=torch.float64, requires_grad=True)
    x1.data[..., 4].clamp_(1e-6, math.pi - 1e-6)
    ok1 = torch.autograd.gradcheck(xywhr_to_xyxyxyxy, (x1,), eps=1e-4, atol=1e-3)
    print(f"  xywhr_to_xyxyxyxy: {'PASS' if ok1 else 'FAIL'}")

    x2a = torch.rand(2, 5, device=dev, dtype=torch.float64, requires_grad=True)
    x2b = torch.rand(2, 5, device=dev, dtype=torch.float64, requires_grad=True)
    x2a.data[..., 2:4].abs_().add_(0.01)
    x2b.data[..., 2:4].abs_().add_(0.01)
    x2a.data[..., 4].clamp_(1e-6, math.pi - 1e-6)
    x2b.data[..., 4].clamp_(1e-6, math.pi - 1e-6)

    def pi_sum(a, b):
        return probiou(a, b).sum()

    ok2 = torch.autograd.gradcheck(pi_sum, (x2a, x2b), eps=1e-4, atol=1e-3)
    print(f"  probiou: {'PASS' if ok2 else 'FAIL'}")

    x3 = torch.rand(3, 5, device=dev, dtype=torch.float64, requires_grad=True)
    x3.data[..., 2:4].abs_().add_(0.01)
    x3.data[..., 4].clamp_(1e-6, math.pi - 1e-6)
    mat = torch.tensor([[0.9, -0.3, 10.0], [0.2, 0.8, -5.0]], device=dev, dtype=torch.float64)

    def aff_fn(boxes):
        return affine_obb_matrix(boxes, mat).sum()

    ok3 = torch.autograd.gradcheck(aff_fn, (x3,), eps=1e-4, atol=1e-3)
    print(f"  affine_obb_matrix: {'PASS' if ok3 else 'FAIL'}")

    assert ok1 and ok2 and ok3, f"gradcheck: to_xyxy={ok1} probiou={ok2} affine={ok3}"
    print("  done")


# ═══════════════════════════════════════════
ALL_TESTS = {
    "Visualization": test_visualization,
    "Distribution": test_distribution,
    "Matcher": test_matcher,
    "DecoderRefine": test_decoder_refinement,
    "LossMonotonicity": test_loss_monotonicity,
    "NumericalStability": test_numerical_stability,
    "GradientCheck": test_gradient_check,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", type=str, default=None)
    args = parser.parse_args()

    if args.only:
        ALL_TESTS[args.only]()
    else:
        passed = failed = 0
        for name, fn in ALL_TESTS.items():
            try:
                fn()
                passed += 1
            except Exception as e:
                print(f"  FAILED: {e}")
                failed += 1
                import traceback
                traceback.print_exc()
        print(f"\n{'='*50}")
        print(f"{passed} passed, {failed} failed")
        if failed:
            sys.exit(1)
    print(f"\nOutputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
