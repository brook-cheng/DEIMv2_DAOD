# OBB ADR 分解 Loss 设计

Date: 2026-08-04

## 1. Purpose

为 rep0/rep2（使用 `(cx,cy,ext_w,ext_h,offset_w,offset_h)` 表示的旋转矩形）实现真正的 ADR 分解 loss：将 5D OBB 分解为外接矩形 + 顶点偏移，分别计算 loss，完全移除角度依赖的 loss 项，同时保留不受角度影响的 KLD loss。

**关键区别：** 当前 `_noangle` 配置只是在 5D OBB 上关闭角度项；本设计是**结构性分解**——外接矩形用非旋转 HBB loss，偏移用 L1。

## 2. 背景

当前 `deim_criterion.py:294-340` 的 OBB loss 路径：

```
loss_bbox     = canonical_side_l1_loss(src_5D, tgt_5D)   ← 仍在 5D OBB 上
loss_probiou  = probiou(src_5D, tgt_5D)                   ← 仍在 5D OBB 上
loss_angle    = yolo_angle_loss(src_5D, tgt_5D)           ← 角度项（可关）
loss_kld      = kld_loss(src_5D, tgt_5D)                  ← 保留
```

没有任何 OBB→外接矩形+偏移 的分解。

## 3. 设计目标

1. 新增 `loss_boxes_adr` 路径：OBB 分解为外接矩形（4D）+ 偏移（2D），分别计算。
2. 外接矩形部分使用**非旋转 HBB loss**（L1 + GIoU）。
3. 偏移部分使用 **L1 loss**。
4. 保留 KLD（在 5D OBB 上，几何感知、不受角度参数影响）。
5. 完全移除角度 loss 项。
6. 通过配置开关控制，便于消融。

## 4. 非目标

- 不修改 decoder 内部 FDR 逻辑（`distance2bbox_obb` / `bbox2distance_obb` 保持不变）。
- 不修改 HBB 路径。
- 不修改 matcher（匹配仍用原 OBB 成本）。
- 不修改 FGL/DDF loss。

## 5. 架构

```
src_5D (cx,cy,w,h,θ) ──oriented_box_to_external_rect──→ (ext_rect_4D, offsets_2D)
tgt_5D (cx,cy,w,h,θ) ──oriented_box_to_external_rect──→ (ext_rect_4D, offsets_2D)

                      │
                      ├─ loss_extrect_l1 = L1(src_ext_4D, tgt_ext_4D)
                      ├─ loss_extrect_giou = 1 - GIoU(src_ext_xyxy, tgt_ext_xyxy)
                      ├─ loss_offset_l1 = L1(src_offsets_2D, tgt_offsets_2D)
                      ├─ loss_kld = kld_loss(src_5D, tgt_5D)        ← 保留
                      └─ loss_angle = 移除                          ← 无角度
```

### 5.1 分解函数

使用现有 `engine/deim/obb_geometry.py:oriented_box_to_external_rect`：

```python
ext_rect_src, offsets_src = oriented_box_to_external_rect(src_boxes)
ext_rect_tgt, offsets_tgt = oriented_box_to_external_rect(target_boxes)
```

- `ext_rect`: `(N, 4)` — `(x1, y1, x2, y2)`（外接矩形像素坐标）
- `offsets`: `(N, 2)` — `(ε, η)`（顶点偏移）

### 5.2 外接矩形 loss

外接矩形是轴对齐的，用标准 HBB loss：

```python
# L1 on (cx,cy,ext_w,ext_h) — 先把 xyxy 转 cxcywh 或直接用 xyxy L1
ext_src_cxcywh = box_xyxy_to_cxcywh(ext_rect_src)
ext_tgt_cxcywh = box_xyxy_to_cxcywh(ext_rect_tgt)
loss_extrect_l1 = F.l1_loss(ext_src_cxcywh, ext_tgt_cxcywh, reduction="none").sum(-1)

# GIoU on xyxy
loss_extrect_giou = 1 - torch.diag(generalized_box_iou(ext_rect_src, ext_rect_tgt))
```

### 5.3 偏移 loss

```python
loss_offset_l1 = F.l1_loss(offsets_src, offsets_tgt, reduction="none").sum(-1)
```

### 5.4 KLD 保留

```python
loss_kld = kld_loss(src_boxes, target_boxes, reduction="none")
```

### 5.5 总 loss

```python
loss_bbox = loss_extrect_l1 + loss_extrect_giou + loss_offset_l1
```

各分量在 weight_dict 中分别可配（便于消融）：
```python
weight_dict: {
    loss_extrect_l1: 5,
    loss_extrect_giou: 2,
    loss_offset_l1: 1,
    loss_kld: 2,
    ...
}
```

## 6. 接口设计

### 6.1 新 criterion 参数

`DEIMCriterion.__init__` 新增：

```python
adr_loss: bool = False,   # True = 启用 ADR 分解 loss
```

当 `adr_loss=True` 时：
- 进入 `loss_boxes_adr` 路径
- `loss_extrect_l1` / `loss_extrect_giou` / `loss_offset_l1` 必须出现在 weight_dict
- `loss_angle` 不要求（不存在）
- `keep_kld=True` 时 `loss_kld` 保留

### 6.2 配置示例

```yaml
DEIMCriterion:
  adr_loss: true
  keep_kld: true
  weight_dict: {
    loss_extrect_l1: 5,
    loss_extrect_giou: 2,
    loss_offset_l1: 1,
    loss_kld: 2,
    loss_mal: 1,
    loss_fgl: 0.15,
    loss_ddf: 1.5
  }
```

### 6.3 loss 分发

`loss_boxes` 方法中：

```python
elif self.box_mode == "obb":
    if self.adr_loss:
        # ADR 分解路径
        ...
    elif self.use_yolo_probiou or self.use_yolo_angle:
        # 原有 yolo 路径
        ...
    else:
        # 原有 periodic/非periodic 路径
        ...
```

## 7. 消融设计

通过 `adr_loss` 开关 + weight_dict 组合，支持以下消融：

| 配置 | `adr_loss` | 说明 |
|---|---|---|
| baseline | false | 当前 yolo 路径（Proposal 3 `_noangle` 基础上） |
| adr-l1 | true | 外接矩形 L1 + 偏移 L1 + KLD |
| adr-l1-giou | true | 外接矩形 L1+GIoU + 偏移 L1 + KLD |
| adr-nokld | true | 关闭 KLD（`keep_kld=false`） |
| adr-giou-only | true | weight_dict 中 loss_extrect_l1=0（GIoU 主导） |

### 7.1 配置文件

基于 `sp_fz_rep0_nloss.yml` 创建：
- `sp_fz_rep0_nloss_adr.yml` — 基础 ADR（l1 + giou + offset + kld）
- `sp_fz_rep0_nloss_adr_nokld.yml` — 无 KLD 变体

### 7.2 关键消融问题

1. **外接矩形 loss 是否足够驱动位置收敛？** 对比 `adr-l1` vs `baseline`。
2. **GIoU 是否比纯 L1 更好？** 对比 `adr-l1-giou` vs `adr-l1`。
3. **偏移 L1 是否提供了朝向信息？** 对比 `adr-l1-giou` vs 移除偏移项的变体。
4. **KLD 是否仍然必要？** 对比 `adr-l1-giou` vs `adr-nokld`。

## 8. 边界情况与错误处理

| 场景 | 处理 |
|---|---|
| `adr_loss=True` 但 `loss_extrect_*` 不在 weight_dict | `__init__` 抛 ValueError（仿现有 required_keys 检查） |
| 空标注（无 boxes） | `oriented_box_to_external_rect` 对空 tensor 应返回空；loss 为 0 |
| `keep_kld=False` 且 `adr_loss=True` | 不计算 KLD，只算外接矩形 + 偏移 |
| 数值稳定 | 外接矩形退化（w=0/h=0）时 GIoU 为 0，L1 正常 |

## 9. 测试计划

### 9.1 单元测试

- `oriented_box_to_external_rect` 分解正确性（5D → 4D+2D）
- `loss_extrect_l1` / `loss_extrect_giou` / `loss_offset_l1` 数值正确性
- `adr_loss=True` 时 weight_dict 缺失 key 报错
- 空标注安全
- 角度不变时（仅 w/h 变化），`loss_angle` 无贡献（本质不计算）

### 9.2 消融测试

- `adr-l1` vs `baseline` 的训练 loss 曲线对比（需训练，可选）
- 纯单元测试验证 loss 分量数值

## 10. 验收标准

1. `adr_loss=True` 时，`loss_boxes` 走分解路径，产出 `loss_extrect_l1` / `loss_extrect_giou` / `loss_offset_l1`。
2. 5D OBB 分解为 4D 外接矩形 + 2D 偏移正确。
3. 外接矩形用 HBB loss（L1+GIoU），偏移用 L1。
4. KLD 保留（`keep_kld=True`）。
5. 无角度 loss 项（`loss_angle` 不出现）。
6. 默认 `adr_loss=False` 时行为与当前完全一致。
7. 消融配置创建完成。
