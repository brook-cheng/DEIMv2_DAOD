"""GradNorm 单元测试：验证自适应 loss 权重的正确性。

测试覆盖：
1. 初始化：w_i = 1.0
2. step() 后 w_i 发生变化
3. 重归一化：w_i 均值 ≈ 1.0
4. 梯度大的 loss → 权重降低（反之升高）
5. step() → backward() → optimizer.step() 完整流程无错误
6. 多次 step() 权重持续更新
"""

import sys
import os
import copy

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn


def make_toy_model():
    """简单 2 层网络：shared → head_a (cls) + head_b (reg)"""
    shared = nn.Linear(10, 20, bias=False)
    head_a = nn.Linear(20, 5, bias=False)
    head_b = nn.Linear(20, 3, bias=False)
    model = nn.ModuleDict({"shared": shared, "head_a": head_a, "head_b": head_b})
    return model


def make_toy_losses(model, x):
    """模拟两个 task loss，经过 shared 层"""
    h = model["shared"](x)
    out_a = model["head_a"](h)
    out_b = model["head_b"](h)
    loss_a = out_a.sum().pow(2)
    loss_b = out_b.sum().pow(2)
    return {"loss_a": loss_a, "loss_b": loss_b}


def test_init_weights_one():
    model = make_toy_model()
    gn = __import_gradnorm(model, ["loss_a", "loss_b"])
    assert torch.allclose(gn.w_i, torch.ones(2)), f"Expected [1,1], got {gn.w_i}"
    print("[PASS] test_init_weights_one")


def test_step_changes_weights():
    model = make_toy_model()
    gn = __import_gradnorm(model, ["loss_a", "loss_b"])
    x = torch.randn(4, 10)
    loss_dict = make_toy_losses(model, x)
    weights_before = gn.w_i.detach().clone()
    gn.step(loss_dict)
    weights_after = gn.w_i.detach().clone()
    assert not torch.allclose(weights_before, weights_after), \
        f"Weights unchanged: before={weights_before}, after={weights_after}"
    print(f"[PASS] test_step_changes_weights: {weights_before.tolist()} → {weights_after.tolist()}")


def test_renormalization():
    """w_i 均值应保持 ≈ 1.0（因为 sum / sum * T → 均值 = 1）"""
    model = make_toy_model()
    gn = __import_gradnorm(model, ["loss_a", "loss_b"])
    x = torch.randn(4, 10)
    for _ in range(20):
        loss_dict = make_toy_losses(model, x)
        gn.step(loss_dict)
    mean_w = gn.w_i.mean().item()
    assert abs(mean_w - 1.0) < 0.15, f"Mean w_i = {mean_w}, expected ≈ 1.0"
    print(f"[PASS] test_renormalization: mean w_i = {mean_w:.4f}")


def test_large_gradient_lower_weight():
    """梯度范数大的 loss → 权重应降低（GradNorm 核心特性）"""
    model = make_toy_model()

    with torch.no_grad():
        model["head_a"].weight *= 10.0

    gn = __import_gradnorm(model, ["loss_a", "loss_b"], lr=1e-6)
    x = torch.randn(4, 10)

    for _ in range(50):
        loss_dict = make_toy_losses(model, x)
        gn.step(loss_dict)

    w_a, w_b = gn.w_i[0].item(), gn.w_i[1].item()
    assert w_a < w_b, f"Expected w_a < w_b (large grad → low weight), got w_a={w_a:.4f}, w_b={w_b:.4f}"
    print(f"[PASS] test_large_gradient_lower_weight: w_a={w_a:.4f} < w_b={w_b:.4f}")


def test_full_train_flow():
    """step() → weighted backward() → optimizer.step() 完整流程"""
    model = make_toy_model()
    gn = __import_gradnorm(model, ["loss_a", "loss_b"])
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    x = torch.randn(4, 10)
    initial_loss = None
    for step in range(50):
        loss_dict = make_toy_losses(model, x)

        # GradNorm step（保留 forward graph）
        current_weights = gn.step(loss_dict)

        # 用 w_i.detach() 加权
        loss = sum(loss_dict[name] * gn.w_i[i].detach() for i, name in enumerate(gn.loss_names))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step == 0:
            initial_loss = loss.item()
        if step == 49:
            final_loss = loss.item()

    assert final_loss < initial_loss, \
        f"Loss did not decrease: {initial_loss:.4f} → {final_loss:.4f}"
    print(f"[PASS] test_full_train_flow: loss {initial_loss:.4f} → {final_loss:.4f}")


def test_device_handling():
    """w_i 应自动移到正确 device"""
    if not torch.cuda.is_available():
        print("[SKIP] test_device_handling (no CUDA)")
        return
    device = "cuda:0"
    model = make_toy_model().to(device)
    gn = __import_gradnorm(model, ["loss_a", "loss_b"])
    x = torch.randn(4, 10, device=device)
    loss_dict = make_toy_losses(model, x)
    gn.step(loss_dict)
    assert gn.w_i.device.type == "cuda", f"w_i on {gn.w_i.device}, expected cuda"
    print(f"[PASS] test_device_handling: w_i on {gn.w_i.device}")


def test_weights_logged():
    """step() 返回的 dict 应包含所有 loss 名称"""
    model = make_toy_model()
    gn = __import_gradnorm(model, ["loss_a", "loss_b"])
    x = torch.randn(4, 10)
    loss_dict = make_toy_losses(model, x)
    result = gn.step(loss_dict)
    assert set(result.keys()) == {"loss_a", "loss_b"}
    assert all(isinstance(v, float) for v in result.values())
    print(f"[PASS] test_weights_logged: {result}")


def __import_gradnorm(model, loss_names, lr=1e-3):
    from engine.solver.gradnorm import GradNorm
    return GradNorm(
        model,
        loss_names=loss_names,
        shared_param_pattern=r"shared\.weight",
        alpha=0.12,
        lr=lr,
    )


if __name__ == "__main__":
    test_init_weights_one()
    test_step_changes_weights()
    test_renormalization()
    test_large_gradient_lower_weight()
    test_full_train_flow()
    test_device_handling()
    test_weights_logged()
    print("\n✅ All GradNorm tests passed")