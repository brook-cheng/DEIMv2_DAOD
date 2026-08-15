"""Tests for engine.deim.obb_error_classify.classify_errors.

锁定 classify_errors 的契约:
  - 输入: index-aligned 的 (N,5) gt/pred OBB 张量 (cx,cy,w,h,θ_rad)
  - 输出: dict {"ok", "genuine_angle_error", "swap_artifact"} (int 计数)
  - 语义:
      |Δθ| ≤ angle_threshold_deg                          → ok
      |Δθ| > threshold 且 ProbIoU ≥ iou_threshold          → swap_artifact
                                                           (w/h 轴互换伪误差)
      |Δθ| > threshold 且 ProbIoU < iou_threshold          → genuine_angle_error
"""

import math

import pytest
import torch

from engine.deim.obb_error_classify import classify_errors


def _box(cx, cy, w, h, theta_deg):
    """构造单条 (cx,cy,w,h,θ_rad) OBB。"""
    return torch.tensor([[cx, cy, w, h, math.radians(theta_deg)]], dtype=torch.float32)


def _stack(boxes):
    """把多个单条 (1,5) 张量堆叠成 (N,5)。"""
    return torch.cat(boxes, dim=0)


# ──────────────────────────────────────────────────────────────────
# 结构契约
# ──────────────────────────────────────────────────────────────────


def test_returns_dict_with_required_keys():
    gt = _box(0, 0, 100, 50, 0)
    pred = _box(0, 0, 100, 50, 0)
    out = classify_errors(gt, pred)
    assert set(out.keys()) == {"ok", "genuine_angle_error", "swap_artifact"}
    for v in out.values():
        assert isinstance(v, int)


def test_empty_input_returns_all_zeros():
    gt = torch.empty((0, 5), dtype=torch.float32)
    pred = torch.empty((0, 5), dtype=torch.float32)
    out = classify_errors(gt, pred)
    assert out == {"ok": 0, "genuine_angle_error": 0, "swap_artifact": 0}


def test_counts_partition_all_pairs():
    """ok + genuine + swap 必须等于输入对数 N。"""
    gt = _stack(
        [
            _box(0, 0, 100, 50, 0),
            _box(0, 0, 100, 50, 0),
            _box(10, 10, 80, 40, 10),
        ]
    )
    pred = _stack(
        [
            _box(0, 0, 100, 50, 0),        # ok: 相同
            _box(0, 0, 100, 50, 45),       # genuine: 45° 旋转, IoU 低
            _box(10, 10, 40, 80, 100),     # swap: 90° + w/h 互换
        ]
    )
    out = classify_errors(gt, pred)
    assert out["ok"] + out["genuine_angle_error"] + out["swap_artifact"] == 3


# ──────────────────────────────────────────────────────────────────
# ok: 小角度差
# ──────────────────────────────────────────────────────────────────


def test_identical_pairs_all_ok():
    gt = _box(5, 5, 100, 50, 30)
    pred = _box(5, 5, 100, 50, 30)
    out = classify_errors(gt, pred, angle_threshold_deg=15.0)
    assert out == {"ok": 1, "genuine_angle_error": 0, "swap_artifact": 0}


def test_small_angle_within_threshold_is_ok():
    gt = _box(0, 0, 100, 50, 0)
    pred = _box(0, 0, 100, 50, 10)  # 10° < 15° 阈值
    out = classify_errors(gt, pred, angle_threshold_deg=15.0)
    assert out["ok"] == 1
    assert out["genuine_angle_error"] == 0
    assert out["swap_artifact"] == 0


def test_angle_exactly_at_threshold_is_ok():
    """边界: |Δθ| == threshold 不算 large (严格 > 才进 large 分支)。"""
    gt = _box(0, 0, 100, 50, 0)
    pred = _box(0, 0, 100, 50, 15)  # 恰好 15°
    out = classify_errors(gt, pred, angle_threshold_deg=15.0)
    assert out["ok"] == 1


# ──────────────────────────────────────────────────────────────────
# genuine_angle_error: 大角度差 + 低 IoU
# ──────────────────────────────────────────────────────────────────


def test_large_angle_offset_low_iou_is_genuine_error():
    """45° 旋转 + 中心偏移 → 物理框几乎不重叠, ProbIoU 低 → 真实角度误差.

    注意: 仅旋转不改中心时, ProbIoU 仍可能 ≥0.5 (高斯分布对同中心框宽松),
    所以 genuine 用例必须让 IoU 真正降低 (平移或极端纵横比).
    """
    gt = _box(0, 0, 100, 50, 0)
    pred = _box(80, 80, 100, 50, 45)
    out = classify_errors(gt, pred, angle_threshold_deg=15.0, iou_threshold=0.5)
    assert out["genuine_angle_error"] == 1
    assert out["swap_artifact"] == 0
    assert out["ok"] == 0


def test_large_angle_translation_low_iou_is_genuine():
    """大角度差 + 中心偏移 → 低 IoU → genuine。"""
    gt = _box(0, 0, 80, 40, 0)
    pred = _box(60, 60, 80, 40, 60)
    out = classify_errors(gt, pred, angle_threshold_deg=15.0, iou_threshold=0.5)
    assert out["genuine_angle_error"] == 1


# ──────────────────────────────────────────────────────────────────
# swap_artifact: 大角度差(~90°) + w/h 互换 → 高 IoU
# ──────────────────────────────────────────────────────────────────


def test_classic_wh_swap_at_90deg_is_artifact():
    """(w=100,h=50,θ=0) vs (w=50,h=100,θ=90°) 是同一个物理框.

    angle_diff = 90° (> 15°), ProbIoU ≈ 1.0 (≥ 0.5) → swap_artifact.
    这是 swap artifact 的判别用例: w/h 轴归属不同导致表观 90° 角度差,
    但几何上完全重合。
    """
    gt = _box(0, 0, 100, 50, 0)
    pred = _box(0, 0, 50, 100, 90)
    out = classify_errors(gt, pred, angle_threshold_deg=15.0, iou_threshold=0.5)
    assert out["swap_artifact"] == 1
    assert out["genuine_angle_error"] == 0
    assert out["ok"] == 0


def test_wh_swap_at_minus_90deg_is_artifact():
    """-90° (即 270°) + w/h 互换 同样是 swap artifact。"""
    gt = _box(0, 0, 100, 50, 0)
    pred = _box(0, 0, 50, 100, -90)
    out = classify_errors(gt, pred, angle_threshold_deg=15.0, iou_threshold=0.5)
    assert out["swap_artifact"] == 1


def test_large_angle_high_iou_but_not_90deg_still_swap_by_iou():
    """判别只看 IoU 不看是否恰好 90°: 大角度差 + 高 IoU 即判 swap。

    构造: 近 90° 偏移 + w/h 近互换, ProbIoU 仍 ≥ 阈值 → swap_artifact。
    """
    gt = _box(0, 0, 100, 50, 0)
    pred = _box(0, 0, 52, 100, 88)  # 接近 90° + 近互换 → 仍高 IoU
    out = classify_errors(gt, pred, angle_threshold_deg=15.0, iou_threshold=0.3)
    assert out["swap_artifact"] == 1


# ──────────────────────────────────────────────────────────────────
# 阈值边界
# ──────────────────────────────────────────────────────────────────


def test_iou_threshold_boundary_swap():
    """ProbIoU 恰好 == iou_threshold → 仍判 swap (>= 阈值)。"""
    # 构造一对 ProbIoU 可控的对; 用 wh 互换 + 90° 得 ProbIoU≈1.0 >> 0.5
    gt = _box(0, 0, 100, 50, 0)
    pred = _box(0, 0, 50, 100, 90)
    # iou_threshold 提到 0.99, ProbIoU≈1.0 仍 >= → swap
    out = classify_errors(gt, pred, angle_threshold_deg=15.0, iou_threshold=0.99)
    assert out["swap_artifact"] == 1


# ──────────────────────────────────────────────────────────────────
# 多对混合
# ──────────────────────────────────────────────────────────────────


def test_mixed_batch_classifies_each_pair_independently():
    gt = _stack(
        [
            _box(0, 0, 100, 50, 0),      # ok (相同)
            _box(0, 0, 100, 50, 0),      # genuine (45° + 平移, 低 IoU)
            _box(0, 0, 100, 50, 0),      # swap (w/h 互换 + 90°)
            _box(5, 5, 80, 80, 20),      # ok (5° < 15°)
        ]
    )
    pred = _stack(
        [
            _box(0, 0, 100, 50, 0),      # ok
            _box(80, 80, 100, 50, 45),   # genuine
            _box(0, 0, 50, 100, 90),     # swap
            _box(5, 5, 80, 80, 25),      # ok (Δθ=5°)
        ]
    )
    out = classify_errors(gt, pred, angle_threshold_deg=15.0, iou_threshold=0.5)
    assert out["ok"] == 2
    assert out["genuine_angle_error"] == 1
    assert out["swap_artifact"] == 1


def test_default_thresholds():
    """默认 iou_threshold=0.5, angle_threshold_deg=15.0。"""
    gt = _box(0, 0, 100, 50, 0)
    pred_ok = _box(0, 0, 100, 50, 5)
    assert classify_errors(gt, pred_ok)["ok"] == 1

    pred_genuine = _box(80, 80, 100, 50, 45)  # 平移 + 旋转 → 低 IoU
    assert classify_errors(gt, pred_genuine)["genuine_angle_error"] == 1

    pred_swap = _box(0, 0, 50, 100, 90)
    assert classify_errors(gt, pred_swap)["swap_artifact"] == 1


# ──────────────────────────────────────────────────────────────────
# OBB π-周期性: θ 和 θ+180° 是同一个角度
# ──────────────────────────────────────────────────────────────────


def test_angle_periodicity_180deg_is_ok():
    """θ=0° 和 θ=180° 对 OBB 是同一个朝向 → angle_diff=0 → ok。"""
    gt = _box(0, 0, 100, 50, 0)
    pred = _box(0, 0, 100, 50, 180)
    out = classify_errors(gt, pred, angle_threshold_deg=15.0)
    assert out["ok"] == 1


def test_angle_wrap_near_pi():
    """θ=179° 和 θ=1° 的最短角度差是 2° → ok (不是 178°)。"""
    gt = _box(0, 0, 100, 50, 179)
    pred = _box(0, 0, 100, 50, 1)
    out = classify_errors(gt, pred, angle_threshold_deg=15.0)
    assert out["ok"] == 1
