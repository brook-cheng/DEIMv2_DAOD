# DEIMv2-OBB Loss Refinement：移植 YOLO-OBB 损失体系

Date: 2026-07-16

## 1. Purpose

将 YOLO-OBB 的三个关键损失设计移植到 DEIMv2-OBB 的 `DEIMCriterion` 和 `HungarianMatcher` 中，同时保留 DEIMv2 的核心优势（FGL + DDF 蒸馏、MAL 分类、匈牙利匹配框架）。目标是在不牺牲 DEIMv2 现有能力的前提下，修复当前 OBB 损失在角度建模、几何度量和损失对齐上的已知缺陷。

## 2. Background

### 2.1 当前问题

DEIMv2-OBB 的损失体系有三个经代码审查确认的缺陷：

1. **角度损失过于朴素**：`periodic_angle_distance`（`obb_geometry.py:18-39`）只是 `min(|Δθ|, |π-Δθ|)` 的 L1 变体，不理解 w↔h 互换时角度等价性，也不理解正方形物体的角度无关性。

2. **几何损失冗余且不对齐**：`loss_bbox`（L1 on (cx,cy,w,h) + periodic angle L1）和 `loss_kld`（KLD）同时存在。L1 作者自己标注了 `# FIXME: L1距离存在缺陷`（`deim_criterion.py:265`）；KLD 值域 `[0, ∞)`，与 eval 使用的 ProbIoU 不对齐。

3. **匹配与损失度量不一致**：`HungarianMatcher` 使用 `cost_probiou` 做匹配，但 `loss_boxes` 使用 KLD + L1 做监督——匹配选出来的最优配对，loss 却用不同的几何度量去优化。

### 2.2 设计原则

- **保留 DEIMv2 优势**：FGL+DDF 蒸馏、MAL 分类、匈牙利匹配框架、CDN、encoder aux
- **替换 L1 而非 KLD**：KLD 作为高斯分布距离有其理论价值；L1 是明确的瓶颈
- **matcher 与 criterion 同度量**：匹配的 cost 和训练的 loss 使用一致的几何度量
- **只改 OBB 路径**：所有修改由 `self.box_mode == "obb"` 门控

## 3. Design Goals

1. 用 YOLO-OBB 的 `sin²(2Δθ) × AR_weight` 角度损失替换当前的 `periodic_angle_distance` L1
2. 引入 ProbIoU box loss 作为新的主几何损失，与 matcher 的 `cost_probiou` 对齐
3. 保留 KLD 作为辅助损失（可选，通过 weight 控制）
4. 同步更新 `HungarianMatcher` 的 cost 计算，确保匹配与损失使用相同度量
5. 所有新增损失通过 `weight_dict` 控制权重，支持 Kendall Uncertainty Weighting
6. 零影响 HBB 路径

## 4. Non-Goals

- 不修改 FGL/DDF 蒸馏体系
- 不修改 MAL 分类损失
- 不修改匈牙利匹配框架本身（只调整 cost 组成）
- 不修改 decoder 结构或 OBB 表示（ADR）
- 不修改 `postprocessor`、`evaluate` 或数据集管线

## 5. Detailed Changes

### 5.1 新增文件：`engine/deim/yolo_obb_loss.py`

独立的损失函数模块，从 YOLO-OBB 提取核心计算：

```python
def yolo_angle_loss(pred_bboxes, target_bboxes, fg_mask, weight, 
                     target_scores_sum, lambda_val=3.0):
    """sin²(2Δθ) × aspect_ratio_weight angle loss.
    
    Args:
        pred_bboxes:   (N_q, 5) predicted OBBs in (cx,cy,w,h,θ) format, θ∈[0,π)
        target_bboxes: (N_q, 5) target OBBs
        fg_mask:       (N_q,) bool mask of matched queries
        weight:        (N_q,) loss weight per query (from target_scores)
        target_scores_sum: scalar for normalization
        lambda_val:    sensitivity to aspect ratio (default 3.0)
    
    Returns:
        scalar angle_loss
    """
    w_gt, h_gt = target_bboxes[..., 2], target_bboxes[..., 3]
    pred_theta, target_theta = pred_bboxes[..., 4], target_bboxes[..., 4]
    
    # Aspect ratio weight: square objects get weak angle supervision
    log_ar = torch.log((w_gt + 1e-9) / (h_gt + 1e-9))
    scale_weight = torch.exp(-(log_ar ** 2) / (lambda_val ** 2))
    
    # Wrap angle difference to [-π/2, π/2]
    delta_theta = pred_theta - target_theta
    delta_theta_wrapped = delta_theta - torch.round(delta_theta / math.pi) * math.pi
    
    # sin²(2Δθ): zero at Δθ=0 and Δθ=±π/2 (w↔h symmetry)
    ang_loss = torch.sin(2 * delta_theta_wrapped[fg_mask]) ** 2
    
    ang_loss = scale_weight[fg_mask] * ang_loss * weight
    return ang_loss.sum() / target_scores_sum


def yolo_probiou_loss(pred_bboxes, target_bboxes, fg_mask, weight, 
                       target_scores_sum):
    """1 - ProbIoU box regression loss.
    
    Args:
        pred_bboxes:   (N_q, 5) predicted OBBs
        target_bboxes: (N_q, 5) target OBBs
        fg_mask:       (N_q,) bool mask
        weight:        (N_q,) per-query weight
        target_scores_sum: scalar normalizer
    
    Returns:
        scalar probiou_loss
    """
    iou = probiou(pred_bboxes[fg_mask], target_bboxes[fg_mask]).squeeze(-1)
    loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum
    return loss_iou
```

**依赖**：`from ..deim.obb_ops import probiou`

### 5.2 修改文件：`engine/deim/deim_criterion.py`

#### 5.2.1 `__init__` 新增参数

```python
# 在 __init__ 末尾新增：
self.use_yolo_angle = True       # 启用 YOLO-style sin²(2Δθ) 角度损失
self.use_yolo_probiou = True     # 启用 ProbIoU box loss（替代 L1）
self.angle_lambda = 3.0          # 宽高比敏感度
self.keep_kld = True             # 是否保留 KLD 作为辅助损失
```

#### 5.2.2 修改 `loss_boxes`（第 242-289 行）

当前 OBB 分支的计算逻辑（行 264-287）：

```python
# --- 删除以下代码 ---
elif self.box_mode == "obb":
    # FIXME: L1距离存在缺陷        ← 去掉这个分支的 L1 逻辑
    if self.periodic_angle_flag:
        spatial_l1 = F.l1_loss(src_boxes[..., :4], target_boxes[..., :4], reduction="none")
        angle_term = self.lambda_angle * periodic_angle_distance(...) / torch.pi
        loss_bbox = torch.cat([spatial_l1, angle_term], dim=-1)
    ...
    losses["loss_bbox"] = loss_bbox.sum() / num_boxes
    loss_kld = kld_loss(src_boxes, target_boxes, reduction="none")
    losses["loss_kld"] = loss_kld.sum() / num_boxes

# --- 替换为 ---
elif self.box_mode == "obb":
    weight = ...                  # 从 boxes_weight 获取，若为 None 则用 1.0
    fg_mask = torch.ones(src_boxes.shape[0], dtype=torch.bool, device=src_boxes.device)
    target_scores_sum = max(src_boxes.shape[0], 1)
    
    # 1. ProbIoU box loss (替代 L1)
    if self.use_yolo_probiou:
        from .yolo_obb_loss import yolo_probiou_loss
        losses["loss_probiou"] = yolo_probiou_loss(
            src_boxes, target_boxes, fg_mask, weight, target_scores_sum
        )
    else:
        # 保留原 L1 逻辑（向后兼容）
        ...
    
    # 2. YOLO-style angle loss (替代 periodic_angle_distance)
    if self.use_yolo_angle:
        from .yolo_obb_loss import yolo_angle_loss
        losses["loss_angle"] = yolo_angle_loss(
            src_boxes, target_boxes, fg_mask, weight, target_scores_sum,
            lambda_val=self.angle_lambda
        )
    
    # 3. KLD 辅助损失（可选保留）
    if self.keep_kld:
        loss_kld = kld_loss(src_boxes, target_boxes, reduction="none")
        losses["loss_kld"] = loss_kld.sum() / num_boxes
```

**关键设计决策**：

- `loss_bbox` key 不再使用（被 `loss_probiou` + `loss_angle` 替代）
- KLD 默认保留但权重可置 0 以消融
- `boxes_weight`（IoU 权重）通过 `get_loss_meta_info` 获取，用于加权 ProbIoU loss

#### 5.2.3 新增 `get_loss_meta_info` 支持

`get_loss_meta_info`（第 717-750 行）当前对 `loss in ("boxes",)` 才返回 `boxes_weight`。需要扩展：

```python
# 第 743 行修改：
- if loss in ("boxes",):
+ if loss in ("boxes", "probious",):
      meta = {"boxes_weight": iou}
```

同时修改 `forward()` 中的 `use_uni_set` 判断（第 571 行），确保 `loss_probiou` 也使用统一匹配集：

```python
# 第 571 行修改：
- use_uni_set = self.use_uni_set and (loss in ["boxes", "local"])
+ use_uni_set = self.use_uni_set and (loss in ["boxes", "local", "probiou"])
```

### 5.3 修改文件：`engine/deim/matcher.py`

#### 5.3.1 `__init__` 新增参数

```python
# matcher.py __init__
self.cost_angle = weight_dict.get("cost_angle", 0)  # 新增：角度 cost 权重
```

#### 5.3.2 修改 `forward` cost 计算（第 172-188 行）

当前 OBB 路径：
```python
if self.box_mode == "obb":
    spatial_cost = torch.cdist(out_bbox[..., :4], tgt_bbox[..., :4], p=1)
    angle_cost = periodic_angle_distance(...).squeeze(-1) / self.angle_factor
    cost_bbox = spatial_cost + self.lambda_angle * angle_cost
    cost_probiou = -batch_probiou(out_bbox, tgt_bbox, eps=1e-8)
    C = self.cost_bbox * cost_bbox + self.cost_class * cost_class 
        + self.cost_probiou * cost_probiou + ...

# --- 修改为 ---
if self.box_mode == "obb":
    cost_probiou = -batch_probiou(out_bbox, tgt_bbox, eps=1e-8)
    
    # YOLO-style angle cost: sin²(2Δθ) × AR_weight on the cost matrix
    if self.cost_angle > 0:
        from .yolo_obb_loss import compute_angle_cost_matrix
        cost_angle = compute_angle_cost_matrix(out_bbox, tgt_bbox)
    
    C = (self.cost_bbox * cost_bbox        # L1 cost (保留)
         + self.cost_class * cost_class
         + self.cost_probiou * cost_probiou
         + self.cost_angle * cost_angle    # 新增
         + ...)
```

#### 5.3.3 新增 `yolo_obb_loss.compute_angle_cost_matrix`

```python
def compute_angle_cost_matrix(pred_bboxes, tgt_bboxes, lambda_val=3.0):
    """Compute pairwise angle cost matrix for Hungarian matching.
    
    Args:
        pred_bboxes: (N_q, 5) predicted OBBs
        tgt_bboxes:  (N_gt, 5) target OBBs
    
    Returns:
        (N_q, N_gt) angle cost matrix, values in [0, 1]
    """
    N_q, N_gt = pred_bboxes.shape[0], tgt_bboxes.shape[0]
    
    # Extract angles
    pred_theta = pred_bboxes[:, 4:5]   # (N_q, 1)
    tgt_theta = tgt_bboxes[:, 4:5]     # (N_gt, 1)
    
    # Pairwise angle difference with π-periodic wrapping
    delta = pred_theta - tgt_theta.T    # (N_q, N_gt)
    delta = delta - torch.round(delta / math.pi) * math.pi
    
    # sin²(2Δθ): handles both π-periodicity and w↔h symmetry
    angle_cost = torch.sin(2 * delta) ** 2
    
    # Aspect ratio weighting: (N_gt,) → broadcast to (N_q, N_gt)
    tgt_w, tgt_h = tgt_bboxes[:, 2], tgt_bboxes[:, 3]
    log_ar = torch.log((tgt_w + 1e-9) / (tgt_h + 1e-9))
    ar_weight = torch.exp(-(log_ar ** 2) / (lambda_val ** 2))  # (N_gt,)
    ar_weight = ar_weight.unsqueeze(0)  # (1, N_gt) for broadcast
    
    return angle_cost * ar_weight
```

### 5.4 修改文件：YAML 配置文件

#### 5.4.1 `configs/custom_obb/deimv2_obb_common.yml`

```yaml
# 当前 (line 157-158):
DEIMCriterion:
  weight_dict: {loss_mal: 1, loss_bbox: 5, loss_kld: 2, loss_fgl: 0.15}
  losses: ['mal', 'boxes', 'local']

# 改为:
DEIMCriterion:
  weight_dict: {loss_mal: 1, loss_probiou: 5, loss_angle: 3, loss_kld: 1, loss_fgl: 0.15}
  losses: ['mal', 'boxes', 'local']
  use_yolo_probiou: True
  use_yolo_angle: True
  keep_kld: True
  matcher:
    weight_dict: {cost_class: 2, cost_probiou: 5, cost_angle: 3, cost_chamfer: 5}
```

#### 5.4.2 `configs/custom_obb/deimv2_obb_sp.yml`

同步更新 `weight_dict` 和 matcher `weight_dict`。

### 5.5 `weight_dict` key 映射变更

| Key (旧) | Key (新) | 含义 | 默认权重 |
|----------|----------|------|---------|
| `loss_bbox` | — | 删除（被以下两项替代） | — |
| — | `loss_probiou` | 1 - ProbIoU box 损失 | 5 |
| — | `loss_angle` | sin²(2Δθ) 角度损失 | 3 |
| `loss_kld` | `loss_kld` | KLD 辅助损失（保留，权重降低） | 1 |
| `loss_mal` | `loss_mal` | MAL 分类（不变） | 1 |
| `loss_fgl` | `loss_fgl` | FGL 精炼（不变） | 0.15 |

### 5.6 Matcher `weight_dict` key 映射变更

| Key (旧) | Key (新) | 含义 | 默认权重 |
|----------|----------|------|---------|
| `cost_bbox` | `cost_bbox` | L1 空间 cost（保留） | 5 |
| `cost_probiou` | `cost_probiou` | ProbIoU cost（不变） | 5 → 2 |
| `cost_chamfer` | `cost_chamfer` | Chamfer cost（不变） | 5 |
| — | `cost_angle` | sin²(2Δθ) 角度 cost（新增） | 3 |

**`cost_bbox` 保留原因**：L1 的 `torch.cdist` 提供全局搜索时的尺度基准，ProbIoU 在某些极端旋转下可能退化。保留 L1 cost 作为稳定基线。

**`cost_probiou` 权重从 5 降到 2 的原因**：现在 `loss_probiou` 已承担主回归角色，matcher 中 ProbIoU cost 的权重可以降低，避免匹配过度依赖单一度量。

## 6. 数据流

### 6.1 训练时 loss 计算流

```
DEIMCriterion.forward(outputs, targets)
  │
  ├─ HungarianMatcher.forward(outputs, targets)
  │     └─ cost_matrix = 5×cost_bbox + 2×cost_class 
  │                      + 2×cost_probiou + 3×cost_angle + 5×cost_chamfer
  │     └─ linear_sum_assignment → indices
  │
  ├─ loss_boxes(outputs, targets, indices, num_boxes, boxes_weight=IoU)
  │     └─ yolo_probiou_loss()     → loss_probiou
  │     └─ yolo_angle_loss()       → loss_angle
  │     └─ kld_loss()              → loss_kld (optional)
  │
  ├─ loss_labels_mal(outputs, targets, indices)    → loss_mal
  └─ loss_local(outputs, targets, indices)          → loss_fgl + loss_ddf
```

### 6.2 匹配与 Loss 度量对齐

```
                    Matching Cost           Training Loss
                    ─────────────           ─────────────
ProbIoU             cost_probiou            loss_probiou         ← SAME metric ✓
Angle               cost_angle              loss_angle           ← SAME metric ✓
Spatial L1          cost_bbox               — (removed)          ← matcher-only
Chamfer             cost_chamfer            — (matcher-only)     ← matcher-only
Classification      cost_class              loss_mal (MAL)       ← aligned via soft target
Distribution        —                       loss_fgl + loss_ddf  ← criterion-only
KLD                 —                       loss_kld             ← criterion-only
```

## 7. 向后兼容

### 7.1 HBB 路径

所有 `self.box_mode == "obb"` 分支独立，`self.box_mode == "hbb"` 路径的 `loss_boxes` 逻辑完全不变。

### 7.2 OBB 回退

通过配置开关支持回退到旧行为：

```yaml
DEIMCriterion:
  use_yolo_probiou: False   # 回退到 L1
  use_yolo_angle: False     # 回退到 periodic_angle_distance
  keep_kld: True            # 保留 KLD（旧行为）
```

当 `use_yolo_probiou=False` 且 `use_yolo_angle=False` 时，`loss_boxes` 降级为原始 L1 + KLD 逻辑。

### 7.3 旧 config 兼容

旧的 `weight_dict` key `loss_bbox` 在 criterion 中通过 `_handle_deprecated_keys` 自动映射：

```python
# deim_criterion.py __init__ 末尾新增
if "loss_bbox" in self.weight_dict and "loss_probiou" not in self.weight_dict:
    LOGGER.warning("'loss_bbox' is deprecated for OBB, auto-mapping to 'loss_probiou'")
    self.weight_dict["loss_probiou"] = self.weight_dict.pop("loss_bbox")
```

## 8. 测试策略

### 8.1 单元测试（新增 `tests/test_yolo_obb_loss.py`）

| 测试 | 预期 |
|------|------|
| `test_angle_loss_zero_when_same` | pred=target → loss=0 |
| `test_angle_loss_zero_when_w_h_swapped` | θ_pred = θ_target + π/2 → loss=0（w↔h 互换） |
| `test_angle_loss_max_when_perpendicular` | θ_pred = θ_target + π/4 → loss 最大 |
| `test_angle_loss_square_ignored` | ar≈1 时 loss 权重 ≈ 1.0 |
| `test_angle_loss_very_thin_suppressed` | ar<<1 时 loss 权重 → 0 |
| `test_probiou_loss_bounded` | loss ∈ [0, 1] |
| `test_probiou_loss_zero_perfect_match` | 完全重合 → loss=0 |

### 8.2 集成测试（修改 `tests/test_deim_criterion.py`）

| 测试 | 预期 |
|------|------|
| `test_obb_loss_keys` | `losses` dict 包含 `loss_probiou`, `loss_angle`, `loss_kld` |
| `test_obb_loss_no_bbox_key` | `losses` dict **不**包含 `loss_bbox` |
| `test_hbb_loss_unchanged` | HBB 路径输出与修改前一致 |
| `test_backward_compat` | `use_yolo_*=False` 时输出与旧行为一致 |

### 8.3 回归检查（训练 1 epoch）

- HBB 路径（`box_mode='hbb'`）：loss 数值与修改前误差 < 1e-6
- OBB 路径（`box_mode='obb'`）：loss 无 NaN，收敛趋势正常
- `train.py --test-only`：eval mAP 可计算、无崩溃

## 9. 风险与缓解

| 风险 | 概率 | 影响 | 缓解 |
|------|------|------|------|
| `loss_probiou` + `loss_kld` 同时存在导致梯度冲突 | 中 | 收敛变慢 | Kendall weighting 自动平衡；`keep_kld=False` 消融 |
| `cost_angle` 引入后匹配不稳定 | 低 | mAP 下降 | `cost_angle` 权重默认 3，可调至 0 禁用 |
| `loss_probiou` 在极小框上数值不稳定 | 低 | NaN loss | ProbIoU 内部已含 `eps` 和 `clamp` 保护 |
| 旧 config 文件的 `loss_bbox` key 丢失 | 低 | 训练崩溃 | `_handle_deprecated_keys` 自动映射 + warning |

## 10. Files Changed

| File | Change Type | Description |
|------|-------------|-------------|
| `engine/deim/yolo_obb_loss.py` | **新建** | `yolo_angle_loss`, `yolo_probiou_loss`, `compute_angle_cost_matrix` |
| `engine/deim/deim_criterion.py` | 修改 | `__init__` 新参数, `loss_boxes` OBB 分支重写, `get_loss_meta_info` 扩展 |
| `engine/deim/matcher.py` | 修改 | `__init__` 新参数 `cost_angle`, `forward` OBB cost 矩阵扩展 |
| `configs/custom_obb/deimv2_obb_common.yml` | 修改 | `weight_dict` 和 matcher `weight_dict` 更新 |
| `configs/custom_obb/deimv2_obb_sp.yml` | 修改 | 同上 |
| `tests/test_yolo_obb_loss.py` | **新建** | 角度损失和 ProbIoU 损失单元测试 |
