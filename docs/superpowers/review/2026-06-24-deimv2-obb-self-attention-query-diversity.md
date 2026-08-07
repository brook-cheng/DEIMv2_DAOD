# Self-Attention Query Diversity Analysis

> 代码审查：DEIMv2-OBB decoder 的 300 个 query 进入自注意力前是否具备足够区分度

## 审查范围

| 文件 | 行号 | 检查内容 |
|------|------|---------|
| `engine/deim/deim_decoder.py` | 518-522 | `_reset_parameters` — 参数初始化 |
| `engine/deim/deim_decoder.py` | 631-676 | `_generate_anchors` — anchor 格点生成 |
| `engine/deim/deim_decoder.py` | 700-735 | `_get_decoder_input` — encoder top-k 选 query |
| `engine/deim/deim_decoder.py` | 737-755 | `_select_topk` — top-k 选择逻辑 |
| `engine/deim/deim_decoder.py` | 451-453 | `query_pos_head` — 位置编码 |
| `engine/deim/deim_decoder.py` | 57-62 | `self_attn` — 自注意力模块 |
| `engine/deim/deim_decoder.py` | 94-99 | 自注意力 forward（q/k 构造） |

---

## 结论

**300 个 decoder query 在初始化时角度维度（θ）完全同质化，空间维度仅由 encoder 分数驱动无多样性约束，导致自注意力在训练早期缺乏区分 query 的信号。**

---

## 详细证据

### 1. θ 初始化全为 0.5

`_reset_parameters()` 中 encoder bbox head 末层 bias 置零：

```python
# engine/deim/deim_decoder.py:521-522
init.constant_(self.enc_bbox_head.layers[-1].weight, 0)
init.constant_(self.enc_bbox_head.layers[-1].bias, 0)
```

`enc_bbox_head` 是 3 层 MLP（`deim_decoder.py:447-449`），末层权重和 bias 均为 0。forward 时输出 `enc_bbox_head(x) + anchors`，其中 encoder head 贡献为零，全靠 anchors。`_generate_anchors()`（`deim_decoder.py:633-658`）中 OBB anchor 的 θ 固定为 0.5：

```python
# engine/deim/deim_decoder.py:656-658 (obb path)
grid_xy = torch.stack([grid_x, grid_y], dim=-1)
wh = torch.ones_like(grid_xy) * 0.05
theta = torch.ones((*grid_xy.shape[:-1], 1)) * 0.5
anchors = torch.cat([grid_xy, wh, theta], dim=-1)
```

最终 `enc_topk_bbox_unact ≈ anchors`，sigmoid 后所有 query 的 **θ 初始值均为 0.5**。

### 2. Encoder Top-K 选择无空间约束

`_select_topk()` 纯按 `enc_score_head` 输出的最高分类分数取 top-300：

```python
# engine/deim/deim_decoder.py:744-745
if self.query_select_method == "default":
    _, topk_ind = torch.topk(outputs_logits.max(-1).values, topk, dim=-1)
```

没有空间去偏、没有 NMS、没有角度多样性约束。若相邻像素的 encoder 分数均高，就会选出多个**空间位置接近 + θ 相同**的 query。

### 3. Query 位置嵌入角度区分度低

`query_pos_head` 映射 `(cx,cy,w,h,θ)` → 256 维位置嵌入：

```python
# engine/deim/deim_decoder.py:451-452
self.query_pos_head = MLP(
    self._num_box_dof, hidden_dim, hidden_dim, 3, act=mlp_act
)
```

`(cx,cy,w,h)` 的值域是 `[0, 1]`（归一化坐标），`θ` 固定为 0.5。当两个 query 的空间坐标接近时，`(cx,cy,w,h,θ)` 整体几乎相同 → `query_pos_embed` 几乎相同。

### 4. 自注意力 key/query 缺乏区分信号

自注意力 forward 中 q 和 k 均来自 `target + query_pos_embed`：

```python
# engine/deim/deim_decoder.py:95-97
q = k = self.with_pos_embed(target, query_pos_embed)
target2, _ = self.self_attn(q, k, value=target, attn_mask=attn_mask)
```

由于 `query_pos_embed` 相似（见上），多个 query 在 attention space 中映射到邻近位置 → 自注意力无法通过 key 相似度学习"抑制"——每个 query 对所有其他 query 的关注度是近似均匀的。

### 5. 训练初期 query content 同质化

Query content 来自两方面：
- 非 CDN 模式：`content = enc_topk_memory.detach()`（`deim_decoder.py:725`）— 从 encoder 特征图中按 top-k 位置采样
- CDN 模式：`content = torch.concat([denoising_logits, content], dim=1)`（`deim_decoder.py:733`）— 前半部分为 CDN query

非 CDN 的 300 个 query content 是 encoder 特征的稀疏采样。若 encoder 特征缺乏细粒度差异（deformable attention 在训练初期未收敛的典型情况），content 维度也同质化。

---

## 影响

| 阶段 | 现象 | 后果 |
|------|------|------|
| 训练初期 | 300 个 query 在 attention space 中聚类 | 自注意力无法学习到 query 之间的抑制关系 |
| 训练中期 | CDN 的 attention mask 提供部分区分 | 非 CDN query 之间仍缺乏区分 |
| 推理时 | 无 CDN mask | 多个 query 可能聚合到同一目标区域 → 重复检测 |

这解释了"高召回（多个 query 都检测到目标）但低精度（同一目标被多个 query 重复预测，在 IoU 评估中被计为 FP）"的现象。

---

## 未涉及的方面

- 本文档仅审查 query **初始化**阶段的多样性，不分析训练收敛后的实际 attention 矩阵
- 不涉及 CDN attention mask 的正确性（由 C2 测试覆盖）
- 不提出修改建议（本次测试计划的约束）
