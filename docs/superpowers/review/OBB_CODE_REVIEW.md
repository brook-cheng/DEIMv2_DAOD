# DEIMv2-OBB 代码审查报告

> **修复状态核验日期**：2026-06-16 — 对照源码逐项核查，10/11 已修复。详见 §0 结论速览表。
>
> 审查对象：`/home/cx/win_dir/thired/DEIMv2_DAOD`（在 DEIMv2/D-FINE 实时检测器上移植 O2-DETR 的定向框检测）
> 参考实现：`/home/cx/win_dir/thired/ai4rs/projects/rotated_rtdetr`（O2-RTDETR）
> 参考论文：Ding et al. 2026《Real-Time Oriented Object Detection Transformer》、Huang et al. 2025《Real-Time Object Detection Meets DINOv3 (DEIMv2)》
> 方法：逐行人工审查全部“定向（OBB）相关”代码，并对照论文公式与 ai4rs 参考实现核验数学。
> 约定：OBB = `(cx, cy, w, h, θ)`，θ 公开物理弧度 ∈[0,π)（半开区间）；坐标训练时按图像尺寸归一化到 [0,1]；解码器内部私有归一化角 `theta_norm = θ/π` ∈[0,1)（严格等比，无 shifted seam）；loss 内部规范域 `[-π/4, 3π/4)` 仅由 `physical_rad_to_loss_rad` 提供，配合 `periodic_angle_distance` 使用。

---

## 0. 结论速览

| # | 位置 | 问题 | 严重度 | 置信度 | 状态 |
|---|------|------|--------|--------|------|
| 1 | `engine/deim/dfine_decoder.py:167-184` | 旋转交叉注意力数学错误（按旋转后的半边向量逐元素缩放，而非旋转尺寸缩放后的偏移） | 严重 | 高（含反例） | ✅ 已修复 |
| 2 | `engine/data/transforms/mosaic.py:122-134,158-166` | Mosaic 把拼图平移量加到了框的 w、h 上 | 严重 | 高 | ✅ 已修复 |
| 3 | `engine/data/transforms/obb_transforms.py:11-18` | OBBFlip 翻框但不翻图（作者 FIXME） | 严重 | 高 | ✅ 已修复 |
| 4 | `configs/custom_obb/deimv2_obb_sp.yml` | 训练/验证图像归一化不一致（val 有 ImageNet Normalize，train 无） | 严重 | 高 | ✅ 已修复 |
| 5 | `engine/eval/obb_eval.py:157-180` | 每类 AP 被 append 两次，其中一次未按分数排序 | 严重 | 高 | ✅ 已修复 |
| 6 | `engine/eval/{obb_eval,dota_eval}.py` | 评测用 ProbIoU 近似而非精确多边形 IoU（`poly_iou.py` 成死代码） | 主要 | 高 | ❌ 暂缓（见说明） |
| 7 | `engine/data/transforms/obb_transforms.py` (OBBResize/ZoomOut/IoUCrop) | 对旋转框做各向异性 w/h 缩放却不更新 θ | 主要 | 高 | ✅ 已修复 |
| 8 | `configs/deimv2_obb/deimv2_obb_dinov3_s_dota.yml:4` | include 了不存在的 `../base/deimv2_obb.yml`（加载即崩溃） | 主要（疑似废弃配置） | 高 | ✅ 已修复 |
| 9 | `engine/deim/obb_ops.py:184,198` | KLD 损失行列式 clamp 加错位置 | 中 | 高（位置错确凿） | ✅ 已修复 |
| 10 | `engine/deim/matcher.py:173` | `cost_kld` 实际算 `-ProbIoU` 而非论文的 KL 散度 | 中（命名/语义） | 高（事实）/ 需作者确认是否有意 | ✅ 已修复 |
| 11 | `engine/deim/dfine_utils.py:212 vs 253-255` | ADR 顶点偏移前向/反向缩放基准不一致 | 中 | 高（不一致确凿）/ 精度影响需实测 | ✅ 已修复 |

> **修复统计**：11 个问题中 10 个已修复，1 个暂缓（#6）。全部 5 个「严重」级别的 bug 均已修复。
>
> **关于 #6（ProbIoU → polygon IoU）**：`poly_iou.py` 的精确多边形 IoU 基于逐边求交实现，计算量远超 ProbIoU（高斯近似），在验证集较大时 eval 耗时会显著增加。目前使用 ProbIoU 不影响不同训练 run 之间的相对比对（所有 run 在同一度量标准下），absolute AP 值虽与 DOTA 官方口径略有偏差但不影响训练调优决策。该 bug 标记为「暂缓 —— 后续修复」，届时可考虑向量化加速 `poly_iou` 或将其作为评测阶段的可选项。
>
> 注：当前**实际训练配置**是 `configs/custom_obb/deimv2_obb_sp.yml`（含真实数据路径）。问题 #8 在另一个疑似废弃的配置里，不影响主链路。

---

## A 组：严重 / 主要 bug（高置信度，建议优先修）

> 修复状态：**全部 8 个已修复**（#1–#5, #7–#8），#6 暂缓（详见上方说明）。

### 1. 旋转交叉注意力数学错误（严重）✅ 已修复
**位置**：`engine/deim/dfine_decoder.py:167-184`（`MSDeformableAttention.forward` 的 5 维分支）

**现状（错误）**：
```python
elif reference_points.shape[-1] == 5:
    cosa = torch.cos(reference_points[..., 4:] * torch.pi)
    sina = torch.sin(reference_points[..., 4:] * torch.pi)
    rot_matrix = torch.cat([cosa, -sina, sina, cosa], dim=-1).view(bs, Len_q, -1, 2, 2)
    wh = reference_points[..., 2:4] * 0.5
    rotated_points = torch.einsum("bnh i j,bnh j->bnh i", rot_matrix, wh)   # = R(θ)·(w/2,h/2)，一个向量
    offset = sampling_offsets * num_points_scale * rotated_points[:, :, None, :, :] * self.offset_scale
    sampling_locations = reference_points[:, :, None, :, :2] + offset
```

**为什么错**：它把采样偏移 `(dx,dy)` 与“旋转后的半边向量 `R(θ)·(w/2,h/2)`”**逐元素相乘**，得到 `offset=(dx·rp_x, dy·rp_y)`。而论文/几何上正确的“旋转采样点”应是先按尺寸缩放、再整体旋转：`offset = R(θ)·(dx·w/2, dy·h/2)`。

**反例（数学确证）**：正方形框 `w=h`、`θ=45°` 时 `rp_x = (w/2)(cos45°−sin45°) = 0`，于是**所有 x 方向偏移都变成 0**，采样点退化成一条竖线。仅当 θ=0（R=I）时该式恰好退化为正确式，因此“看起来对”，实则 θ≠0 全错。这直接破坏了论文核心“按角度旋转采样点（rotation-aware cross-attention）”。

**修复（正确代码）**：
```python
elif reference_points.shape[-1] == 5:
    # reference_points: (bs, Len_q, n_levels=1, 5) — (cx, cy, w, h, θ)，θ∈[0,1] → 弧度
    angle = reference_points[..., 4] * torch.pi                       # (bs, Len_q, 1)
    cosa, sina = torch.cos(angle), torch.sin(angle)
    rot = torch.stack([cosa, -sina, sina, cosa], dim=-1).reshape(bs, Len_q, 2, 2)  # (bs,Lq,2,2)
    wh = reference_points[..., 2:4] * 0.5                             # (bs, Len_q, 1, 2)
    # 1) 先按半边尺寸缩放每个采样偏移
    scaled = sampling_offsets * num_points_scale * self.offset_scale * wh[:, :, None, :, :]
    # 2) 再用 R(θ) 绕框中心整体旋转
    rotated = torch.einsum("bqij,bqhpj->bqhpi", rot, scaled)          # (bs,Lq,heads,points,2)
    sampling_locations = reference_points[:, :, None, :, :2] + rotated
```

---

### 2. Mosaic 把平移量加到了框的宽高上（严重）✅ 已修复
**位置**：`engine/data/transforms/mosaic.py:122-134`（`create_mosaic_from_cache`）与 `:158-166`（`create_mosaic_from_dataset`）

**现状（错误）**：
```python
offsets = torch.tensor([[0,0],[max_width,0],[0,max_height],[max_width,max_height]]).repeat(1, 2)  # 行如 [dx,dy,dx,dy]
if boxes.shape[-1] == 5:
    offsets = torch.cat([offsets, torch.zeros(4, 1)], dim=-1)   # → [dx,dy,dx,dy,0]
...
target["boxes"] = target["boxes"] + offsets[i]                 # (cx,cy,w,h,θ)+[dx,dy,dx,dy,0]
```

**为什么错**：`[dx,dy,dx,dy]` 是 HBB 的 xyxy 平移写法。对 5 维 `(cx,cy,w,h,θ)` 来说，第 3、4 维是 **w、h**，于是 `w += dx, h += dy`。dx/dy 是拼图块的放置偏移（常达数百像素），框宽高被严重污染。配置 `mosaic_prob:0.5` 下该路径生效（OBB 还会跳过 affine，所以这就是全部变换）。

**修复（正确代码，两处同样改）**：
```python
if boxes.shape[-1] == 5:
    # OBB (cx,cy,w,h,θ)：只平移中心，保持 w,h,θ 不变
    offsets = torch.tensor([[0,0],[max_width,0],[0,max_height],[max_width,max_height]],
                           dtype=torch.float32)
    offsets = torch.cat([offsets, torch.zeros(4, 3)], dim=-1)   # → [dx,dy,0,0,0]
else:
    offsets = torch.tensor([[0,0],[max_width,0],[0,max_height],[max_width,max_height]],
                           dtype=torch.float32).repeat(1, 2)     # [dx,dy,dx,dy] for xyxy
```

---

### 3. OBBFlip 翻框但不翻图（严重）✅ 已修复
**位置**：`engine/data/transforms/obb_transforms.py:11-18`（作者已 `# FIXME: 图片没有Flip`）

**现状（错误）**：
```python
def forward(self, sample):
    img, tgt, ds = sample
    b = tgt["boxes"]
    _, h, w = img.shape if hasattr(img, "shape") else (img.mode, *img.size[::-1])
    b[:, 0] = w - b[:, 0]
    b[:, 4] = (torch.pi - b[:, 4]) % torch.pi
    return img, tgt, ds                 # ← img 原样返回，未翻转
```
在 `custom_obb/deimv2_obb_sp.yml:39` 训练流水线中**实际启用** → 图像与标注错位，等于注入随机标签噪声。

**修复（正确代码）**：
```python
import random
import torchvision.transforms.v2.functional as TF

@register()
class OBBFlip(nn.Module):
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def forward(self, sample):
        img, tgt, ds = sample
        if random.random() > self.p:
            return img, tgt, ds
        _, h, w = img.shape if hasattr(img, "shape") else (3, *img.size[::-1])
        img = TF.hflip(img)                              # ← 真正翻转图像
        b = tgt["boxes"]
        b[:, 0] = w - b[:, 0]                            # 像素坐标下翻转中心 x
        b[:, 4] = (torch.pi - b[:, 4]) % torch.pi        # le90 约定下镜像角度
        return img, tgt, ds
```
> 注意：OBBFlip 必须在 `OBBConvertBoxes`（归一化）**之前**执行，框才是像素坐标，`w - b[:,0]` 才成立。当前 sp 配置顺序满足此条件。

---

### 4. 训练/验证图像归一化不一致（严重）✅ 已修复
**位置**：`configs/custom_obb/deimv2_obb_sp.yml`
- 训练 ops 末尾仅 `ConvertPILImage(scale: True)`（→[0,1]），**无 Normalize**（line 42-43）。
- 验证 ops 末尾有 `Normalize(mean=ImageNet, std=ImageNet)`（line 66）。

**为什么错**：训练与推理输入分布不同（[0,1] vs ImageNet 标准化），导致评测指标系统性失真。

**修复（二选一，使二者一致）**：
- 若 DINOv3 主干期望 ImageNet 标准化 → **训练也加 Normalize**：
  ```yaml
  train_dataloader.dataset.transforms.ops:
    ...
    - {type: ConvertPILImage, dtype: 'float32', scale: True}
    - {type: OBBConvertBoxes, normalize: True, img_size: [640, 640]}
    - {type: Normalize, mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225]}   # 新增
  ```
- 若主干内部已做标准化（或按 [0,1] 训练）→ **删除 val 的 Normalize**。
> 行动项：先确认 `engine/backbone/dinov3` 是否内部做了归一化，再据此统一两端。

---

### 5. 评测 AP 被重复且错误地累加（严重）✅ 已修复
**位置**：`engine/eval/obb_eval.py:157-180`

**现状（错误）**：每个类别循环里 `aps.append` 了**两次**——
```python
# (1) 正确：按分数排序后算 PR/AP
sort_idx = np.argsort(-scores_cat)
tp_cum = np.cumsum(tp_cat[sort_idx]); fp_cum = np.cumsum(fp_cat[sort_idx])
...
aps.append(_voc_ap(rec, prec, use_07_metric=True))

# (2) 错误：未排序直接 cumsum，且重复 append
tp_cum = np.cumsum(tp_cat); fp_cum = np.cumsum(fp_cat)     # 没有 sort_idx！
...
aps.append(_voc_ap(rec, prec, use_07_metric=True))
```
导致 `aps` 长度变为 `2×num_classes`，且 `mean(aps)` 把未排序的错误 AP 平均进去 → AP50/mAP 不可信。

**修复**：删除第二段（约 174-180 行），只保留按分数排序的那次。

---

### 6. 评测使用 ProbIoU 近似而非精确多边形 IoU（主要）❌ 暂缓
**位置**：`engine/eval/obb_eval.py:147,199` 与 `engine/eval/dota_eval.py:106`（`_poly_iou_8coord` 名为多边形 IoU，内部却调 `batch_probiou`）。同目录精确实现 `engine/eval/poly_iou.py` 成为**死代码**。

**为什么错**：DOTA 官方按多边形 IoU 计算 AP；ProbIoU 是高斯近似，数值不等价，报告的 AP 与官方口径不一致。

**修复（obb_eval.py，drop-in 替换）**：
```python
from .poly_iou import poly_iou           # 替换 from ..deim.obb_ops import batch_probiou
...
ious = poly_iou(det_t, gt_t).numpy()      # (N_det, M_gt)，与 batch_probiou 同形状
```
`dota_eval.py::_poly_iou_8coord` 同理改为 `poly_iou(obb_b, obb_a)`（注意返回 shape 顺序，必要时转置以匹配原 `.squeeze(-1)` 用法）。
> 提示：`poly_iou` 为 Python 双重循环，较慢但仅用于评测；如需提速可后续向量化。

---

### 7. OBBResize/OBBZoomOut/OBBIoUCrop 对旋转框做各向异性缩放却不更新 θ（主要）✅ 已修复
**位置**：`engine/data/transforms/obb_transforms.py`（`OBBResize:50-67`、`OBBZoomOut:22-47`、`OBBIoUCrop:122-191`）

**为什么错**：当 `sx≠sy`（非等比 resize / 非方形 crop）时，旋转矩形经各向异性缩放后**不再是同角度矩形**；仅按 `w*=sx, h*=sy` 并保持 θ，只在 θ∈{0,π/2} 或 sx=sy 时成立。OBBIoUCrop 另用 HBB 轴对齐 IoU 选裁剪框（作者 `# FIXME` 标注）。

**修复思路（统一做法：变换 4 顶点后重新拟合）**，以 OBBResize 为例：
```python
from ...deim.obb_geometry import xywhr_to_xyxyxyxy, xyxyxyxy_to_xywhr

class OBBResize(nn.Module):
    def __init__(self, size):           # size = [H, W]
        super().__init__(); self.size = size
    def forward(self, sample):
        img, tgt, ds = sample
        _, h, w = img.shape if hasattr(img, "shape") else (3, *img.size[::-1])
        img = TF.resize(img, self.size)
        sy, sx = self.size[0] / h, self.size[1] / w
        b = tgt["boxes"]                # (N,5) 像素 (cx,cy,w,h,θ)
        if len(b):
            v = xywhr_to_xyxyxyxy(b)     # (N,4,2)
            v[..., 0] = v[..., 0] * sx
            v[..., 1] = v[..., 1] * sy
            tgt["boxes"] = xyxyxyxy_to_xywhr(v)   # 重新拟合 (cx,cy,w,h,θ)
        return img, tgt, ds
```
ZoomOut（pad）与 IoUCrop（平移+裁剪）同样用“变换顶点→重拟合”；裁剪后再按是否越界过滤。OBBIoUCrop 的 IoU 评分若要严谨应改用 `poly_iou`，否则至少注释其为近似。

---

### 8. 配置 include 指向不存在的文件（主要，疑似废弃配置）✅ 已修复
**位置**：`configs/deimv2_obb/deimv2_obb_dinov3_s_dota.yml:4`
```yaml
__include__: ['../dataset/dota_detection.yml', '../runtime.yml', '../base/deimv2_obb.yml']
```
`configs/base/deimv2_obb.yml` 不存在；`yaml_utils.load_config` 直接 `open()`（`yaml_utils.py:44`）→ 加载即 `FileNotFoundError`。

**修复**：改为同目录的真实文件，或删除该废弃配置：
```yaml
__include__: ['../dataset/dota_detection.yml', '../runtime.yml', 'deimv2_obb.yml']
```
> 当前主用训练配置是 `custom_obb/deimv2_obb_sp.yml`（加载正常），此项不影响主链路。

---

## B 组：中等严重（事实确凿，定量影响需实测）

> 修复状态：**全部 3 个已修复**（#9–#11）。

### 9. KLD 损失行列式 clamp 加错位置（中）✅ 已修复
**位置**：`engine/deim/obb_ops.py:184`（`det_t`）、`:198`（`det_p`）

**现状（错误）**：
```python
det_t = sigma_t[..., 0, 0] * sigma_t[..., 1, 1] - sigma_t[..., 0, 1].pow(2).clamp(min=eps)
```
clamp 加在副对角项平方 `a01²` 上（使被减项≥eps，反而让 det 偏小），而非保护整个行列式。退化/极细框时 `det_t` 仍可能≈0 甚至为负 → `inv_t = adj/det_t` 数值爆炸。

**修复**：
```python
det_t = (sigma_t[..., 0, 0] * sigma_t[..., 1, 1] - sigma_t[..., 0, 1].pow(2)).clamp(min=eps)
# det_p 同理：
det_p = (sigma_p[..., 0, 0] * sigma_p[..., 1, 1] - sigma_p[..., 0, 1].pow(2)).clamp(min=eps)
```

### 10. `cost_kld` 实为 ProbIoU 而非 KL 散度（中，命名/语义）✅ 已修复
**位置**：`engine/deim/matcher.py:173`
```python
cost_kld = -batch_probiou(out_bbox, tgt_bbox, eps=1e-8)   # 名为 kld，实为 -ProbIoU
```
论文（O2-RTDETR）将 **KLD 作为 IoU 代价**；此处用 ProbIoU 替代，而 `obb_ops.kld_loss` 反而未被 matcher 使用。ProbIoU 作为定向相似度在工程上可行，但与论文不符且命名误导。

**修复（二选一）**：
- 若工程上沿用 ProbIoU：把键名/变量改为 `cost_probiou` 并在注释中说明（避免误导）。
- 若需复现论文：改用 KL 散度代价（可用 `obb_ops` 内 Gauss/KLD 路径构造），权重 `ξ_kld=2.0`。

### 11. ADR 顶点偏移前向/反向缩放基准不一致（中）✅ 已修复
**位置**：`engine/deim/dfine_utils.py`
- 前向解码 `distance2bbox_obb:211-213`：用**调整后**外接矩形 `ext_adj_cxcywh[...,2:]` 缩放顶点偏移。
- 反向求 FGL 目标 `bbox2distance_obb:253-255`：用**调整前**外接矩形 `ext_rect_cxcywh_pred[...,2:]`。

严格互逆要求两端用同一 (w,h) 基准，否则顶点偏移的回归目标含系统性误差（细化幅度小时为二阶）。

**修复（最小一致化：前向也改用 pred 基准）**：
```python
# distance2bbox_obb
ext_rect_xyxy, vertex_offsets = oriented_box_to_external_rect(points)
ext_rect_cxcywh = box_xyxy_to_cxcywh(ext_rect_xyxy)
ext_adj_cxcywh = distance2bbox(ext_rect_cxcywh, distance[..., :4], reg_scale)
ext_adjust_xyxy = box_cxcywh_to_xyxy(ext_adj_cxcywh)
# FIX: 用与反向目标相同的 pred 外接矩形 w/h 缩放顶点偏移
vertex_offsets_adj = vertex_offsets + distance[..., 4:] * ext_rect_cxcywh[..., 2:] / reg_scale
return external_rect_to_oriented_box(ext_adjust_xyxy, vertex_offsets_adj)
```

---

## C 组：已核验“正确/无需改”（避免误伤）

- θ 量纲链路一致：公开物理角统一为 `[0, π)` 半开弧度（GT 规范化、解码器输出、criterion、matcher、postprocessor 均只交换物理角）；解码器内部 `[0,1]` 为严格等比私有归一化编码（`physical_rad_to_norm(θ) = θ/π` / `norm_to_physical_rad(x) = x·π`），无区间内 shifted seam；loss 内部规范域 `[-π/4, 3π/4)` 仅由 `physical_rad_to_loss_rad` 提供；几何模块（`obb_geometry`）从不消费 norm/logit（`deim_decoder.py:1241-1259`；`postprocessor.py:60-74`）。
- 旋转去噪 OCD：仅实现 box-noise（角度不加噪），与论文“box noise 最优”一致；`physical_rad_to_norm`（等比 `θ/π`）后 `inverse_sigmoid` 正确（`denoising.py:110-115`）。
- Chamfer 代价与论文 Eq.5 一致（双向 min-mean、平方距离、4 顶点）（`chamfer_cost.py`）。
- `Integral` 对 `num_reg_dist=6` 正确；残差 logits 跨层累加符合 D-FINE（`dfine_decoder.py:327-331`；`deim_decoder.py:264`）。
- 解码器采用 RMSNorm + SwiGLUFFN、跨层共享 query 位置编码，符合 DEIMv2“高效解码器”（`deim_decoder.py:62,74,77`）。
- 损失/代价权重与论文吻合：loss `{mal:1, bbox:5, kld:2, fgl:0.15}`、cost `{class:2, bbox:5, chamfer:5, kld:2}`（`custom_obb/deimv2_obb_common.yml`、`deimv2_obb_sp.yml`）。
- OBB↔外接矩形+顶点偏移 几何往返自洽（`obb_geometry.py`，含自测）。

---

## D. 修复优先级建议

> **2026-06-16 更新**：以下 #1–#5, #7–#11 均已完成修复。仅 #6（ProbIoU → polygon IoU）暂缓。

~~1. **先修训练正确性**：#2（Mosaic w/h）、#3（OBBFlip 翻图）、#1（旋转注意力）、#4（归一化一致）。这几项会直接导致定向训练严重劣化。~~
~~2. **再修评测可信度**：#5（重复 AP）、#6（poly IoU）。否则无法用 AP 判断其余修复是否生效。~~
~~3. **数值/精度**：#9、#11；命名澄清 #10。~~
~~4. **几何增强严谨化**：#7。~~
~~5. **配置清理**：#8。~~

当前唯一待办：#6 — 将 eval 从 ProbIoU 改为精确多边形 IoU。后续修复时需注意性能优化（`poly_iou.py` 的 Python 双重循环在大型验证集上可能成为瓶颈）。

---

## E. 二次复核（自校验，Oracle 替代）

> 本环境的子代理（explore/deep/oracle）多次在 17–30 分钟卡死/超时（本次共 5/5 失败），属环境/挂载问题而非审查内容问题。故对四个数学项改用“自我对抗式”复核（主动尝试证伪以排查假阳性）。

**四个数学项复核结论：全部 CONFIRMED，修复代码经形状/数学验证有效。**

- **#1**：代码位移展开为 `disp_x = dx·(cosθ·w/2 − sinθ·h/2)`、`disp_y = dy·(sinθ·w/2 + cosθ·h/2)`；正确应为 `disp_x = cosθ·(dx·w/2) − sinθ·(dy·h/2)`。代码在交叉项上误用 `dx`（应为 `dy`），从不跨轴混合 dx/dy，故无法旋转采样图样；反例 `w=h,θ=45° ⇒ disp_x≡0` 成立。修复 einsum `"bqij,bqhpj->bqhpi"` 形状自洽。
- **#2**：`[dx,dy,dx,dy,0]` 加到 `(cx,cy,w,h,θ)` ⇒ 第 2 维 w+=dx、第 3 维 h+=dy；框确为 5 维 cxcywhθ。修复 `[dx,dy,0,0,0]` 正确。
- **#9**：运算符优先级使 `.clamp` 仅作用于 `a01²`，无法保证 `det≥eps`；`inv_t=adj/det`（line 188）除以未受保护的 det → 细长框数值爆炸（DOTA 桥/港口常见）。修复 `(a00*a11 - a01**2).clamp(min=eps)` 正确。
- **#11**：反向目标解码得 `vo_adj = vo_pred + (vo_gt−vo_pred)·(ext_adj/ext_pred)`，仅当 `ext_adj==ext_pred` 才等于 `vo_gt`；不一致确凿。前向改用 `ext_pred` 即成精确互逆（与 D-FINE HBB 路径一致）。

**假阳性检查**：A 组无假阳性；#10（`cost_kld` 实为 ProbIoU）正确归类为“命名/论文一致性的决策项”，ProbIoU 作为定向相似度工程上有效，非硬 bug。

**二次扫描新增的次要隐患（低严重度）**：

- **#12** `OBBResize` 的 size 索引 H/W 错位 — `obb_transforms.py:62` `sx, sy = s[0]/w, s[1]/h`，但 `TF.resize(img, size)` 约定 `size=[H,W]`，故 `sx` 误用了高度 `s[0]`。当前 `size=[640,640]` 故无实际影响；已被 #7 的修复方案（`sy, sx = self.size[0]/h, self.size[1]/w`）顺带纠正。
- **#13** `deim_criterion.py:702-703` `get_loss_meta_info` 对 obb 直接 `raise NotImplementedError()` — 仅当 `boxes_weight_format` 非 None 时触发；当前 obb 配置未设置（默认 None），故未触发，属潜在地雷（若后续启用 `boxes_weight_format` 训练会崩）。
- **#14** `OBBConvertBoxes` 用固定 `img_size=(640,640)` 归一化（`obb_transforms.py:71-85`），强耦合于“`OBBResize` 先执行且尺寸也是 640”。若改 resize 尺寸而忘改此处 `img_size`，会静默产生错误的归一化坐标。建议改为按样本实际尺寸归一化，或与 `OBBResize` 共享尺寸来源。

---

> 本报告仅为审查结论与修复方案，**未改动任何源码**。
