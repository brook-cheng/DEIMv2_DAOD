"""DEIMv2-OBB 模型输出验证：前向值域、梯度稳定性、过拟合能力。

用法：
    python test/test_model_output.py                    # 全部测试
    python test/test_model_output.py --only Forward      # 仅前向
    python test/test_model_output.py --only Gradient     # 仅梯度
    python test/test_model_output.py --only Overfit      # 仅过拟合
"""

import os, sys, argparse, math, torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)


def _setup(cfg_overrides=None):
    from engine.core import YAMLConfig
    from engine.solver import TASKS

    cfg = YAMLConfig(os.path.join(ROOT, "configs/custom_obb/dlzdt/ablation/abl_rep3.yml"))
    cfg.yaml_cfg["train_dataloader"]["total_batch_size"] = 2
    cfg.yaml_cfg["train_dataloader"]["num_workers"] = 0
    cfg.yaml_cfg["val_dataloader"]["num_workers"] = 0
    cfg.yaml_cfg["checkpoint_freq"] = 100
    cfg.yaml_cfg["epoches"] = 1
    cfg.yaml_cfg["use_amp"] = False
    if cfg_overrides:
        cfg.yaml_cfg.update(cfg_overrides)

    solver = TASKS[cfg.yaml_cfg["task"]](cfg)
    solver.train()
    return solver


def _get_batch(loader):
    samples, targets = next(iter(loader))
    samples = samples.cuda()
    targets = [{k: v.cuda() for k, v in t.items()} for t in targets]
    return samples, targets


# ═══════════════════════════════════════════
# 1. 前向输出验证
# ═══════════════════════════════════════════
def test_forward():
    print("\n=== Model Forward ===")
    solver = _setup()
    model = solver.model
    loader = solver.train_dataloader
    samples, targets = _get_batch(loader)

    print(f"  input: {samples.shape}, {len(targets)} targets")
    for i, t in enumerate(targets):
        print(f"    target[{i}]: {len(t['boxes'])} boxes, θ∈[{t['boxes'][:,4].min():.3f},{t['boxes'][:,4].max():.3f}]")

    model.train()
    with torch.no_grad():
        outputs = model(samples, targets=targets)

    # 主输出：形状+值域
    checks = {
        "pred_logits":  (list(outputs["pred_logits"].shape),  "class logits"),
        "pred_boxes":   (list(outputs["pred_boxes"].shape),   "OBB (cx,cy,w,h,θ)"),
        "pred_corners": (list(outputs["pred_corners"].shape), "FDR distribution"),
        "ref_points":   (list(outputs["ref_points"].shape),   "reference boxes"),
    }
    for key, (shape, desc) in checks.items():
        v = outputs[key]
        ok = torch.isfinite(v).all()
        print(f"  {key}: {shape} ({desc})  finite={ok.item()}  "
              f"min={v.min():.4f}  max={v.max():.4f}  mean={v.mean():.4f}")
        assert ok, f"{key} contains NaN/Inf"

    # box 值域
    boxes = outputs["pred_boxes"]
    assert boxes.shape[-1] == 5, f"expected 5-dim OBB, got {boxes.shape[-1]}"
    for j, name in enumerate(["cx", "cy", "w", "h"]):
        assert (boxes[..., j] >= -0.01).all() and (boxes[..., j] <= 1.01).all(), \
            f"{name} out of [0,1]: min={boxes[...,j].min():.4f} max={boxes[...,j].max():.4f}"
    assert (boxes[..., 4] >= 0).all() and (boxes[..., 4] < math.pi).all(), \
        f"θ out of [0,π): min={boxes[...,4].min():.4f} max={boxes[...,4].max():.4f}"
    print("  ✓ box ranges: cx/cy/w/h∈[0,1], θ∈[0,π)")

    # 分类置信度
    probs = outputs["pred_logits"].sigmoid().max(dim=-1).values
    print(f"  max class prob: min={probs.min():.4f} max={probs.max():.4f} mean={probs.mean():.4f}")
    assert probs.max() > 0.01, f"prob too low: {probs.max():.6f}"

    # 辅助输出
    for key in ["aux_outputs", "enc_aux_outputs", "dn_outputs", "pre_outputs"]:
        if key in outputs and outputs[key] is not None:
            items = outputs[key] if isinstance(outputs[key], list) else [outputs[key]]
            for i, item in enumerate(items):
                if isinstance(item, dict):
                    ok = torch.isfinite(item.get("pred_boxes", torch.zeros(1))).all()
                    n_box = item.get("pred_boxes", torch.zeros(1)).shape[1] if "pred_boxes" in item else "?"
                    print(f"  {key}[{i}]: {n_box} queries, finite={ok.item()}")
                    assert ok, f"{key}[{i}] NaN"

    # denoising meta
    if "dn_meta" in outputs:
        print(f"  dn_meta: {outputs['dn_meta'].get('dn_num_group', '?')} groups, "
              f"{outputs['dn_meta'].get('dn_num_split', '?')} split")

    print("  ✓ all outputs finite and valid")


# ═══════════════════════════════════════════
# 2. 梯度稳定性验证
# ═══════════════════════════════════════════
def test_gradient():
    print("\n=== Model Gradient ===")
    solver = _setup()
    model = solver.model
    criterion = solver.criterion
    loader = solver.train_dataloader
    samples, targets = _get_batch(loader)

    model.train()
    criterion.train()

    # 前向 + 损失
    outputs = model(samples, targets=targets)
    losses = criterion(outputs, targets, epoch=0)

    # 按损失类型汇总
    loss_types = {}
    for k, v in losses.items():
        if isinstance(v, torch.Tensor) and v.numel() == 1:
            base = k.split("_")[1] if "_" in k else k
            loss_types.setdefault(base, []).append((k, v.item()))

    print("  loss summary:")
    for base, items in sorted(loss_types.items()):
        vals = [v for _, v in items]
        print(f"    {base}: {len(items)}x, range=[{min(vals):.4f},{max(vals):.4f}]")

    # 主反向 + 梯度检查
    total_loss = sum(v for v in losses.values() if isinstance(v, torch.Tensor) and v.numel() == 1)
    assert torch.isfinite(total_loss), f"total loss is NaN/Inf: {total_loss}"
    model.zero_grad()
    total_loss.backward()

    max_grad, min_grad = 0.0, float("inf")
    nan_params = []
    for name, p in model.named_parameters():
        if p.grad is not None:
            gnorm = p.grad.norm().item()
            if not math.isfinite(gnorm):
                nan_params.append(name)
            max_grad = max(max_grad, gnorm)
            min_grad = min(min_grad, gnorm)

    print(f"  total loss: {total_loss.item():.4f}")
    print(f"  grad norm: min={min_grad:.4f} max={max_grad:.2f}")
    if nan_params:
        print(f"  ⚠ NaN gradients ({len(nan_params)} params): {nan_params[:3]}...")
    else:
        print(f"  ✓ all gradients finite")

    # 逐 loss 分量检查
    print("  per-loss check (re-running forward for each):")
    key_losses = [k for k in losses if "aux" not in k and "enc" not in k and "dn" not in k]
    for loss_name in key_losses[:6]:  # 只检查主要 loss
        model.zero_grad()
        outputs2 = model(samples, targets=targets)
        losses2 = criterion(outputs2, targets, epoch=0)
        if loss_name in losses2 and torch.isfinite(losses2[loss_name]):
            try:
                losses2[loss_name].backward()
                has_nan = any(p.grad is not None and not torch.isfinite(p.grad).all()
                              for p in model.parameters())
                print(f"    {loss_name}: {losses2[loss_name].item():.4f} -> "
                      f"{'NaN!' if has_nan else 'OK'}")
            except RuntimeError as e:
                print(f"    {loss_name}: backward error: {str(e)[:60]}")
    print("  ✓ gradient test complete")


# ═══════════════════════════════════════════
# 3. 小数据过拟合验证
# ═══════════════════════════════════════════
def test_overfit():
    print("\n=== Model Overfit ===")
    solver = _setup()
    model = solver.model
    criterion = solver.criterion
    optimizer = solver.optimizer
    loader = solver.train_dataloader
    samples, targets = _get_batch(loader)

    n_imgs = len(targets)
    n_boxes = sum(len(t["boxes"]) for t in targets)
    print(f"  overfitting on {n_imgs} images, {n_boxes} boxes total")

    model.train()
    criterion.train()
    loss_history = []

    for step in range(20):
        optimizer.zero_grad()
        outputs = model(samples, targets=targets)
        losses = criterion(outputs, targets, epoch=0)
        total_loss = sum(v for v in losses.values()
                         if isinstance(v, torch.Tensor) and v.numel() == 1)

        if not torch.isfinite(total_loss):
            print(f"  step {step}: loss=NaN — UNSTABLE")
            break

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        loss_history.append(total_loss.item())

        if step % 5 == 0 or step == 19:
            print(f"  step {step:2d}: loss={total_loss.item():.4f}")

    if len(loss_history) >= 2:
        first_3 = sum(loss_history[:3]) / 3
        last_3 = sum(loss_history[-3:]) / 3
        ok = last_3 < first_3
        print(f"  loss: {first_3:.4f} → {last_3:.4f}  {'↓ decreasing ✓' if ok else '✗ flat'}")
        assert ok, "loss not decreasing — model cannot learn"
    print("  ✓ overfit test complete")


ALL_TESTS = {
    "Forward": test_forward,
    "Gradient": test_gradient,
    "Overfit": test_overfit,
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


if __name__ == "__main__":
    main()
