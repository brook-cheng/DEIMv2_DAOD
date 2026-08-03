# OBB ADR 分解 Loss 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 `DEIMCriterion` 新增 `adr_loss` 路径：将 5D OBB 分解为外接矩形（4D）+ 顶点偏移（2D），外接矩形用 HBB L1+GIoU、偏移用 L1 分别计算 loss，完全移除角度 loss 项，保留 KLD，并通过配置开关支持消融。

**Architecture:** 三个递进实现任务：(1) `adr_loss` 构造参数 + weight_dict 校验；(2) `loss_boxes` 中 ADR 分解分支（主路径 + 空匹配守卫）；(3) 边界与梯度质量测试。随后创建两个消融配置文件。每任务 TDD 独立可测试。matcher / decoder / HBB 路径 / FGL / DDF 一律不动。

**Tech Stack:** Python 3.12, PyTorch, pytest

## Global Constraints

- **禁止修改 matcher、decoder、HBB 路径、FGL/DDF loss**（spec §4 非目标）。
- `adr_loss` 默认 `False`；`adr_loss=False` 时 `loss_boxes` 行为与当前完全一致（spec 验收标准 6）。
- ADR 路径不得产出 `loss_angle` / `loss_probiou` / `loss_bbox` 键（spec 验收标准 5）。
- 分解必须使用现有 `engine/deim/obb_geometry.py:oriented_box_to_external_rect`（spec §5.1），不新写几何函数。
- 校验失败抛 `ValueError`，且**不得修改传入的 weight_dict**（仿现有 `test_new_mode_rejects_missing_weight_without_mutation`）。
- 测试运行目录：仓库根 `/mnt/d/cx/thired/deimv2_daod`（`test_deim_criterion_obb_loss.py` 直接从根导入 `engine`，无 sys.path 注入）。
- 测试风格：pytest 函数式（无模块级脚本），预期值用手算常数或生产函数计算，不重复实现几何公式。
- 配置基于 `configs/custom_obb/dlzdt/sp_fz_rep0_nloss.yml` 复制创建，不修改原文件；matcher 段原样保留（spec §4）。
- 数值断言用 `torch.allclose(..., atol=1e-6)`；梯度断言只查有限性 + cx/cy 分量非零（offset 经 argmax gather 梯度部分断开，属已知几何属性，见 Task 3 说明）。
- **⚠️ 禁止任何 git 操作**：不执行 `git add`、`git commit`、`git push`、`git stash` 等。git 管理工作（暂存/提交/推送）由用户审查后自行完成。所有 Task 末尾的「提交」步骤已移除，实现完成后仅报告改动文件清单。

---

### Task 1: `adr_loss` 构造参数与 weight_dict 校验

**Files:**
- Modify: `engine/deim/deim_criterion.py`（`__init__` 签名 69-73 行附近 + 赋值与校验 109-126 行附近）
- Create: `test/test_obb_adr_loss.py`

**Interfaces:**
- Consumes: `DEIMCriterion.__init__` 现有参数 `weight_dict`, `box_mode`, `keep_kld`
- Produces: 构造参数 `adr_loss: bool = False`；实例属性 `self.adr_loss`；校验规则——`box_mode=="obb" and adr_loss=True` 时 `weight_dict` 必须含 `loss_extrect_l1` / `loss_extrect_giou` / `loss_offset_l1`，且 `keep_kld=True` 时还必须含 `loss_kld`

- [ ] **Step 1: 写失败测试**

创建 `test/test_obb_adr_loss.py`：

```python
"""OBB ADR decomposition loss tests (spec 2026-08-04-obb-adr-loss-design).

Covers the adr_loss flag, weight_dict validation, the ADR loss_boxes
branch (external rect L1/GIoU + offset L1 + optional KLD), and edge
cases (empty matches, keep_kld=False, gradient flow).

Run:
    pytest test/test_obb_adr_loss.py -v
"""

import math

import pytest
import torch

from engine.deim.deim_criterion import DEIMCriterion
from engine.deim.obb_geometry import oriented_box_to_external_rect
from engine.deim.obb_ops import kld_loss
from engine.deim.box_ops import generalized_box_iou, box_xyxy_to_cxcywh


ADR_WEIGHTS = {
    "loss_extrect_l1": 5.0,
    "loss_extrect_giou": 2.0,
    "loss_offset_l1": 1.0,
    "loss_kld": 2.0,
}


def _adr_criterion(*, keep_kld=True, weights=None, **kwargs):
    """Build a DEIMCriterion with adr_loss=True for direct loss_boxes
    testing. matcher is None because loss_boxes does not invoke it.
    """
    w = weights if weights is not None else dict(ADR_WEIGHTS)
    return DEIMCriterion(
        matcher=None,
        weight_dict=w,
        losses=["boxes"],
        num_classes=1,
        box_mode="obb",
        adr_loss=True,
        keep_kld=keep_kld,
        **kwargs,
    )


def _pair(pred, target, *, requires_grad=False):
    """Build matched outputs/targets/indices for a single-box case."""
    pred_boxes = torch.tensor(
        [pred], dtype=torch.float32, requires_grad=requires_grad
    )
    outputs = {"pred_boxes": pred_boxes.unsqueeze(0)}
    targets = [{"boxes": torch.tensor([target]), "labels": torch.tensor([0])}]
    indices = [(torch.tensor([0]), torch.tensor([0]))]
    return pred_boxes, outputs, targets, indices


# ---------------------------------------------------------------------------
# Task 1: adr_loss flag and weight_dict validation
# ---------------------------------------------------------------------------

def test_adr_init_accepts_flag():
    """adr_loss=True with full ADR weights must construct without error."""
    criterion = _adr_criterion()
    assert criterion.adr_loss is True


def test_adr_flag_defaults_false():
    """adr_loss must default to False; legacy construction is unchanged."""
    criterion = DEIMCriterion(
        matcher=None,
        weight_dict={"loss_bbox": 2.0, "loss_kld": 1.0},
        losses=["boxes"],
        num_classes=1,
        box_mode="obb",
    )
    assert criterion.adr_loss is False


@pytest.mark.parametrize(
    "weights",
    [
        # missing loss_extrect_l1
        {"loss_extrect_giou": 2.0, "loss_offset_l1": 1.0, "loss_kld": 2.0},
        # missing loss_extrect_giou
        {"loss_extrect_l1": 5.0, "loss_offset_l1": 1.0, "loss_kld": 2.0},
        # missing loss_offset_l1
        {"loss_extrect_l1": 5.0, "loss_extrect_giou": 2.0, "loss_kld": 2.0},
        # keep_kld=True -> loss_kld required
        {"loss_extrect_l1": 5.0, "loss_extrect_giou": 2.0, "loss_offset_l1": 1.0},
    ],
)
def test_adr_missing_weight_raises(weights):
    """adr_loss=True must raise ValueError naming the missing key."""
    with pytest.raises(ValueError, match="loss_"):
        _adr_criterion(weights=weights)


def test_adr_missing_weight_error_names_key():
    """The ValueError message must contain the missing key name."""
    weights = {"loss_extrect_l1": 5.0, "loss_extrect_giou": 2.0}
    with pytest.raises(ValueError, match="loss_offset_l1"):
        _adr_criterion(weights=weights)


def test_adr_missing_weight_does_not_mutate_dict():
    """Raising must not mutate the caller's weight_dict."""
    weights = {"loss_extrect_l1": 5.0, "loss_extrect_giou": 2.0}
    original = weights.copy()
    with pytest.raises(ValueError):
        _adr_criterion(weights=weights)
    assert weights == original


def test_adr_nokld_does_not_require_loss_kld():
    """keep_kld=False with adr_loss=True must not require loss_kld."""
    weights = {"loss_extrect_l1": 5.0, "loss_extrect_giou": 2.0, "loss_offset_l1": 1.0}
    criterion = _adr_criterion(keep_kld=False, weights=weights)
    assert criterion.keep_kld is False
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest test/test_obb_adr_loss.py -v`
Expected: FAIL——`TypeError: __init__() got an unexpected keyword argument 'adr_loss'`

- [ ] **Step 3: 实现参数与校验**

在 `engine/deim/deim_criterion.py` 的 `__init__` 签名末尾（`angle_lambda=3.0,` 之后）追加：

```python
        adr_loss=False,
```

在 `self.angle_lambda = angle_lambda`（109 行）之后、现有 yolo 校验块（111 行）之前插入：

```python
        self.adr_loss = adr_loss

        if self.box_mode == "obb" and self.adr_loss:
            required_keys = ["loss_extrect_l1", "loss_extrect_giou", "loss_offset_l1"]
            if self.keep_kld:
                required_keys.append("loss_kld")
            missing_keys = [key for key in required_keys if key not in weight_dict]
            if missing_keys:
                raise ValueError(
                    "OBB ADR-loss mode weight_dict is missing required keys: "
                    + ", ".join(missing_keys)
                )
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest test/test_obb_adr_loss.py -v`
Expected: PASS（8 passed）

---

### Task 2: `loss_boxes` 的 ADR 分解分支

**Files:**
- Modify: `engine/deim/deim_criterion.py`（imports 18-27 行 + `loss_boxes` OBB 分支 294-295 行）
- Modify: `test/test_obb_adr_loss.py`（追加主路径测试）

**Interfaces:**
- Consumes: Task 1 的 `self.adr_loss` / `self.keep_kld`；现有 `oriented_box_to_external_rect`（`(...,5) -> ((...,4) xyxy, (...,2) (epsilon, eta))`）、`box_xyxy_to_cxcywh`、`generalized_box_iou`（`(N,4),(N,4) -> (N,N)`，断言 `x2>=x1`）、`kld_loss(pred,target,reduction="none")`
- Produces: `loss_boxes` 在 `adr_loss=True` 时返回键集 `{loss_extrect_l1, loss_extrect_giou, loss_offset_l1}` +（`keep_kld=True` 时）`{loss_kld}`；空匹配返回标量 0；**绝不返回** `loss_angle` / `loss_probiou` / `loss_bbox`

- [ ] **Step 1: 写失败测试**

追加到 `test/test_obb_adr_loss.py`：

```python
# ---------------------------------------------------------------------------
# Task 2: ADR loss_boxes branch (main path)
# ---------------------------------------------------------------------------

# Axis-aligned manual anchors:
# pred [0.5,0.5,0.4,0.2,0.0] -> ext (0.3,0.4,0.7,0.6) -> cxcywh (0.5,0.5,0.4,0.2)
# tgt  [0.55,0.45,0.5,0.3,0.0] -> ext (0.3,0.3,0.8,0.6) -> cxcywh (0.55,0.45,0.5,0.3)
# L1 = 0.05+0.05+0.10+0.10 = 0.30
_AA_PRED = [0.5, 0.5, 0.4, 0.2, 0.0]
_AA_TGT = [0.55, 0.45, 0.5, 0.3, 0.0]


def test_adr_keys_and_no_angle_loss():
    """ADR path must return ext-rect/offset/kld keys and never
    loss_angle / loss_probiou / loss_bbox."""
    pred = [0.6, 0.55, 0.35, 0.15, 0.4]
    target = [0.5, 0.5, 0.3, 0.2, 0.5]
    _, outputs, targets, indices = _pair(pred, target)
    losses = _adr_criterion().loss_boxes(outputs, targets, indices, 1.0)

    assert set(losses) == {
        "loss_extrect_l1", "loss_extrect_giou", "loss_offset_l1", "loss_kld"
    }
    assert "loss_angle" not in losses
    assert "loss_probiou" not in losses
    assert "loss_bbox" not in losses


def test_adr_extrect_l1_axis_aligned_manual():
    """loss_extrect_l1 must equal manual L1 on (cx,cy,ext_w,ext_h)."""
    _, outputs, targets, indices = _pair(_AA_PRED, _AA_TGT)
    losses = _adr_criterion().loss_boxes(outputs, targets, indices, 1.0)

    expected = torch.tensor(0.30)
    assert torch.allclose(losses["loss_extrect_l1"], expected, atol=1e-6), (
        f"loss_extrect_l1={losses['loss_extrect_l1'].item():.6f} != 0.30"
    )


def test_adr_extrect_giou_axis_aligned_manual():
    """loss_extrect_giou must equal 1 - GIoU on xyxy external rects.

    pred ext (0.3,0.4,0.7,0.6) vs tgt ext (0.4,0.4,0.6,0.6):
    IoU=0.5, convex hull area = union area -> GIoU=0.5 -> loss=0.5.
    """
    pred = [0.5, 0.5, 0.4, 0.2, 0.0]
    target = [0.5, 0.5, 0.2, 0.2, 0.0]
    _, outputs, targets, indices = _pair(pred, target)
    losses = _adr_criterion().loss_boxes(outputs, targets, indices, 1.0)

    expected = torch.tensor(0.5)
    assert torch.allclose(losses["loss_extrect_giou"], expected, atol=1e-6), (
        f"loss_extrect_giou={losses['loss_extrect_giou'].item():.6f} != 0.5"
    )


def test_adr_rotated_components_match_production_decomposition():
    """For a rotated pair, all three component losses must equal manual
    values computed from the production decomposition."""
    pred = [0.5, 0.5, 0.4, 0.2, math.pi / 4]
    target = [0.5, 0.5, 0.4, 0.2, 0.5]
    _, outputs, targets, indices = _pair(pred, target)

    pred_t = torch.tensor([pred], dtype=torch.float32)
    target_t = torch.tensor([target], dtype=torch.float32)
    ext_p, off_p = oriented_box_to_external_rect(pred_t)
    ext_t, off_t = oriented_box_to_external_rect(target_t)

    exp_l1 = F_l1_on(ext_p, ext_t)
    exp_off = (off_p - off_t).abs().sum(-1).sum()
    exp_giou = 1 - torch.diag(generalized_box_iou(ext_p, ext_t)).sum()

    losses = _adr_criterion().loss_boxes(outputs, targets, indices, 1.0)
    assert torch.allclose(losses["loss_extrect_l1"], exp_l1, atol=1e-6)
    assert torch.allclose(losses["loss_extrect_giou"], exp_giou, atol=1e-6)
    assert torch.allclose(losses["loss_offset_l1"], exp_off, atol=1e-6)


def F_l1_on(a, b):
    """Helper: sum of |cxcywh(a) - cxcywh(b)| (matches spec 5.2 formula)."""
    return (box_xyxy_to_cxcywh(a) - box_xyxy_to_cxcywh(b)).abs().sum()


def test_adr_kld_matches_production():
    """loss_kld must equal kld_loss(reduction='none').sum() / num_boxes."""
    pred = [0.6, 0.55, 0.35, 0.15, 0.4]
    target = [0.5, 0.5, 0.3, 0.2, 0.5]
    _, outputs, targets, indices = _pair(pred, target)
    losses = _adr_criterion().loss_boxes(outputs, targets, indices, 1.0)

    pred_t = torch.tensor([pred], dtype=torch.float32)
    target_t = torch.tensor([target], dtype=torch.float32)
    expected = kld_loss(pred_t, target_t, reduction="none").sum()
    assert torch.allclose(losses["loss_kld"], expected, atol=1e-6)


def test_adr_empty_matches_return_finite_zeros():
    """Empty matched pairs must return scalar zero for every ADR key."""
    outputs = {"pred_boxes": torch.zeros(1, 0, 5)}
    targets = [{"boxes": torch.zeros(0, 5), "labels": torch.zeros(0, dtype=torch.long)}]
    empty = torch.zeros(0, dtype=torch.long)

    losses = _adr_criterion().loss_boxes(outputs, targets, [(empty, empty)], 1.0)

    assert set(losses) == set(ADR_WEIGHTS)
    assert all(v.ndim == 0 and v.item() == 0.0 for v in losses.values())
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest test/test_obb_adr_loss.py -v`
Expected: Task 1 的 8 个通过；Task 2 的新增 6 个 FAIL（`KeyError: 'loss_extrect_l1'`）

- [ ] **Step 3: 实现 ADR 分支**

修改 imports（18-19 行与 19-27 行）：

```python
from .obb_geometry import periodic_angle_distance, oriented_box_to_external_rect
```

```python
from .box_ops import (
    box_cxcywh_to_xyxy,
    box_xyxy_to_cxcywh,
    box_iou,
    generalized_box_iou,
    diou,
    ciou,
    eiou,
    siou,
)
```

将 `loss_boxes` 中（294 行）`elif self.box_mode == "obb":` 与 `if self.use_yolo_probiou or self.use_yolo_angle:` 之间插入 ADR 分支：

```python
        elif self.box_mode == "obb":
            if self.adr_loss:
                if src_boxes.shape[0] == 0:
                    zero = torch.zeros(
                        (), device=src_boxes.device, dtype=src_boxes.dtype
                    )
                    losses["loss_extrect_l1"] = zero
                    losses["loss_extrect_giou"] = zero
                    losses["loss_offset_l1"] = zero
                    if self.keep_kld:
                        losses["loss_kld"] = zero
                    return losses

                ext_rect_src, offsets_src = oriented_box_to_external_rect(src_boxes)
                ext_rect_tgt, offsets_tgt = oriented_box_to_external_rect(target_boxes)

                ext_src_cxcywh = box_xyxy_to_cxcywh(ext_rect_src)
                ext_tgt_cxcywh = box_xyxy_to_cxcywh(ext_rect_tgt)
                loss_extrect_l1 = F.l1_loss(
                    ext_src_cxcywh, ext_tgt_cxcywh, reduction="none"
                ).sum(-1)
                loss_extrect_giou = 1 - torch.diag(
                    generalized_box_iou(ext_rect_src, ext_rect_tgt)
                )
                loss_offset_l1 = F.l1_loss(
                    offsets_src, offsets_tgt, reduction="none"
                ).sum(-1)

                losses["loss_extrect_l1"] = loss_extrect_l1.sum() / num_boxes
                losses["loss_extrect_giou"] = loss_extrect_giou.sum() / num_boxes
                losses["loss_offset_l1"] = loss_offset_l1.sum() / num_boxes
                if self.keep_kld:
                    losses["loss_kld"] = (
                        kld_loss(src_boxes, target_boxes, reduction="none").sum()
                        / num_boxes
                    )
            elif self.use_yolo_probiou or self.use_yolo_angle:
```

其余代码不变（yolo 分支与 periodic 分支保持原样）。

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest test/test_obb_adr_loss.py -v`
Expected: PASS（14 passed）

---

### Task 3: 边界与梯度质量测试

**Files:**
- Modify: `test/test_obb_adr_loss.py`（追加测试；若梯度断言暴露问题，再微调 criterion 空守卫——预期不需要）

**Interfaces:**
- Consumes: Task 2 的 `loss_boxes` ADR 分支
- Produces: 质量锁定——`keep_kld=False` 键集、梯度有限性、梯度在 cx/cy 上非零

- [ ] **Step 1: 写失败测试**

追加到 `test/test_obb_adr_loss.py`：

```python
# ---------------------------------------------------------------------------
# Task 3: boundary conditions and gradient quality
# ---------------------------------------------------------------------------

def test_adr_nokld_omits_kld_key():
    """keep_kld=False must omit loss_kld and keep the three ADR keys."""
    pred = [0.6, 0.55, 0.35, 0.15, 0.4]
    target = [0.5, 0.5, 0.3, 0.2, 0.5]
    _, outputs, targets, indices = _pair(pred, target)

    weights = {
        "loss_extrect_l1": 5.0,
        "loss_extrect_giou": 2.0,
        "loss_offset_l1": 1.0,
    }
    losses = _adr_criterion(keep_kld=False, weights=weights).loss_boxes(
        outputs, targets, indices, 1.0
    )

    assert set(losses) == {"loss_extrect_l1", "loss_extrect_giou", "loss_offset_l1"}
    assert "loss_kld" not in losses
    assert torch.isfinite(torch.stack(list(losses.values()))).all()


def test_adr_nokld_empty_returns_three_zeros():
    """keep_kld=False + empty matches must return exactly three zero keys."""
    outputs = {"pred_boxes": torch.zeros(1, 0, 5)}
    targets = [{"boxes": torch.zeros(0, 5), "labels": torch.zeros(0, dtype=torch.long)}]
    empty = torch.zeros(0, dtype=torch.long)
    weights = {
        "loss_extrect_l1": 5.0,
        "loss_extrect_giou": 2.0,
        "loss_offset_l1": 1.0,
    }

    losses = _adr_criterion(keep_kld=False, weights=weights).loss_boxes(
        outputs, targets, [(empty, empty)], 1.0
    )

    assert set(losses) == {"loss_extrect_l1", "loss_extrect_giou", "loss_offset_l1"}
    assert all(v.ndim == 0 and v.item() == 0.0 for v in losses.values())


def test_adr_gradient_flows_finite_and_center_directed():
    """Backward through the ADR losses must yield finite gradients with
    non-zero cx/cy components.

    NOTE: vertex offsets are computed via argmin/argmax gather, which
    does not backpropagate through the selected vertex — the offset
    terms contribute gradient only via x_max/y_max. This is a known
    geometric property of oriented_box_to_external_rect; the assertion
    therefore locks finiteness everywhere and non-zero cx/cy only.
    """
    pred, outputs, targets, indices = _pair(
        [0.5, 0.5, 0.4, 0.2, 0.3],
        [0.55, 0.45, 0.3, 0.2, 0.5],
        requires_grad=True,
    )
    losses = _adr_criterion().loss_boxes(outputs, targets, indices, 1.0)

    sum(losses.values()).backward()

    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all(), f"non-finite grad: {pred.grad}"
    assert pred.grad[0, 0] != 0.0, "gradient on cx must be non-zero"
    assert pred.grad[0, 1] != 0.0, "gradient on cy must be non-zero"


def test_adr_giou_identical_boxes_is_zero():
    """Perfectly matched external rects must give loss_extrect_giou == 0."""
    box = [0.5, 0.5, 0.4, 0.2, 0.0]
    _, outputs, targets, indices = _pair(box, box)
    losses = _adr_criterion().loss_boxes(outputs, targets, indices, 1.0)

    assert torch.allclose(losses["loss_extrect_giou"], torch.tensor(0.0), atol=1e-5)
    assert torch.allclose(losses["loss_offset_l1"], torch.tensor(0.0), atol=1e-6)
```

- [ ] **Step 2: 运行测试确认通过**

Run: `python -m pytest test/test_obb_adr_loss.py -v`
Expected: PASS（18 passed）。若空匹配/梯度断言失败，说明 Task 2 守卫或分解几何有缺陷，修复 criterion 对应位置后重跑（修复不得改动 Task 2 已锁定的键集与数值断言）。

---

### Task 4: ADR 消融配置文件

**Files:**
- Create: `configs/custom_obb/dlzdt/sp_fz_rep0_nloss_adr.yml`
- Create: `configs/custom_obb/dlzdt/sp_fz_rep0_nloss_adr_nokld.yml`
- Verify: 离线脚本（Step 2 内联，不落盘）

**Interfaces:**
- Consumes: Task 1-3 的 `adr_loss` / `keep_kld` / 三个 ADR weight_dict 键
- Produces: 两个可训练配置——`adr`（外接矩形 L1+GIoU + 偏移 L1 + KLD）、`adr_nokld`（同上去 KLD）；matcher 段与 `sp_fz_rep0_nloss.yml` 完全一致（spec §4）

- [ ] **Step 1: 创建基础 ADR 配置**

`cp configs/custom_obb/dlzdt/sp_fz_rep0_nloss.yml configs/custom_obb/dlzdt/sp_fz_rep0_nloss_adr.yml`

编辑 `sp_fz_rep0_nloss_adr.yml`：

1. 第 11 行 `output_dir` 改为：
```yaml
output_dir: ./outputs/deimv2_obb_dlzdt_sp_fz_rep0_nloss_adr
```
2. 第 136-155 行 `DEIMCriterion` 段整体替换为：
```yaml
DEIMCriterion:
  weight_dict: {
    loss_mal: 1,
    loss_extrect_l1: 5,
    loss_extrect_giou: 2,
    loss_offset_l1: 1,
    loss_kld: 2,
    loss_fgl: 0.15,
    loss_ddf: 1.5
  }
  losses: ['mal', 'boxes', 'local']
  adr_loss: true
  use_yolo_probiou: false
  use_yolo_angle: false
  keep_kld: true
  angle_lambda: 0.0
  gamma: 1.0
  alpha: 0.75
  reg_max: 32
  box_mode: obb
  obbox_rep_dim: 6
```
3. matcher 段（157-168 行）**原样保留**，不得改动。

- [ ] **Step 2: 创建无 KLD 变体**

`cp configs/custom_obb/dlzdt/sp_fz_rep0_nloss_adr.yml configs/custom_obb/dlzdt/sp_fz_rep0_nloss_adr_nokld.yml`

编辑 `sp_fz_rep0_nloss_adr_nokld.yml`：

1. `output_dir` 改为：
```yaml
output_dir: ./outputs/deimv2_obb_dlzdt_sp_fz_rep0_nloss_adr_nokld
```
2. `weight_dict` 去掉 `loss_kld` 行（其余不变）：
```yaml
  weight_dict: {
    loss_mal: 1,
    loss_extrect_l1: 5,
    loss_extrect_giou: 2,
    loss_offset_l1: 1,
    loss_fgl: 0.15,
    loss_ddf: 1.5
  }
```
3. `keep_kld: true` 改为 `keep_kld: false`

- [ ] **Step 3: 离线验证两个配置结构**

运行（从仓库根 `deimv2_daod`）：

```bash
python - <<'EOF'
import yaml

for name in ["sp_fz_rep0_nloss_adr", "sp_fz_rep0_nloss_adr_nokld"]:
    path = f"configs/custom_obb/dlzdt/{name}.yml"
    cfg = yaml.safe_load(open(path, encoding="utf-8"))
    crit = cfg["DEIMCriterion"]
    assert crit["adr_loss"] is True, name
    assert crit["use_yolo_probiou"] is False and crit["use_yolo_angle"] is False, name
    wd = crit["weight_dict"]
    assert {"loss_extrect_l1", "loss_extrect_giou", "loss_offset_l1"} <= set(wd), name
    assert "loss_angle" not in wd and "loss_probiou" not in wd, name
    assert crit["keep_kld"] == (name.endswith("_adr")), name
    assert crit["box_mode"] == "obb" and crit["obbox_rep_dim"] == 6, name
    # matcher segment must be byte-identical to the base config
    base = yaml.safe_load(open("configs/custom_obb/dlzdt/sp_fz_rep0_nloss.yml", encoding="utf-8"))
    assert crit["matcher"] == base["DEIMCriterion"]["matcher"], name
    print(f"OK: {name}")
EOF
```

Expected: 两行 `OK: ...` 输出，无断言错误。

---

### Task 5: 全量回归验证

**Files:**
- 无新增；验证范围 `test/test_obb_adr_loss.py` + 既有相关测试

**Interfaces:**
- Consumes: Task 1-4 全部产物
- Produces: 回归证明——ADR 新测试全过，且既有 criterion / ADR 几何 / matcher 测试无一回归（spec 验收标准 6：默认路径行为不变）

- [ ] **Step 1: 运行全部相关测试**

Run: `python -m pytest test/test_obb_adr_loss.py test/test_deim_criterion_obb_loss.py test/test_obb_adr_geometry.py -q`

Expected: 全部 PASS（此前基线 123 passed + 新增 18 = 141 passed, 0 failed）。若 `test_deim_criterion_obb_loss.py` 或 `test_obb_adr_geometry.py` 出现失败，立即停止并修复（禁止通过删改既有测试掩盖回归）。

- [ ] **Step 2: 快速冒烟——默认配置路径不受影响**

Run: `python -m pytest test/test_obb_loss_integration.py test/test_obb_loss_experiment_configs.py -q`

Expected: PASS（若这两个文件因数据依赖跳过，跳过视为通过；失败则停止排查）。

---

## Self-Review 记录

- **Spec 覆盖**：§5.1 分解函数 → Task 2（使用现有 `oriented_box_to_external_rect`）；§5.2 外接矩形 L1+GIoU → Task 2；§5.3 偏移 L1 → Task 2；§5.4 KLD 保留 → Task 2/3（`keep_kld` 开关）；§5.5 总 loss 各分量可配 → Task 4 weight_dict；§6.1 `adr_loss` 参数与校验 → Task 1；§6.2 配置示例 → Task 4；§6.3 loss 分发 → Task 2 分支顺序；§7 消融配置 → Task 4（`adr` + `adr_nokld` 两文件）；§8 边界（缺 key / 空标注 / nokld / 数值稳定）→ Task 1/2/3；§9 测试计划 → Task 1-3；§10 验收标准 1-7 → Task 2（1/3/4/5）、Task 1（2）、Task 5（6）、Task 4（7）。无缺口。
- **占位符扫描**：全部步骤含完整代码/命令，无 TBD/TODO。
- **类型一致性**：`adr_loss`、`keep_kld`、`loss_extrect_l1`/`loss_extrect_giou`/`loss_offset_l1`/`loss_kld` 键名在 Task 1-4 间一致；测试辅助函数 `_adr_criterion` / `_pair` 签名跨 Task 一致；`oriented_box_to_external_rect` / `box_xyxy_to_cxcywh` / `generalized_box_iou` / `kld_loss` 均为现有生产函数，签名与用法核对过。
