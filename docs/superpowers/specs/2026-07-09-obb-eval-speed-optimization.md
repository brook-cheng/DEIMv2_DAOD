# OBB 评估速度优化记录

Date: 2026-07-09

## 背景

训练过程中 `obb_evaluate` 是每 N 个 checkpoint 调用一次的在线评估函数，位于 `engine/eval/obb_eval.py`。IoU 计算使用 `batch_probiou`（Gaussian ProbIoU），位于 `engine/deim/obb_ops.py`。

## 当前瓶颈

### 瓶颈 1：逐图串行 + CPU↔GPU 反复搬运

`obb_evaluate` 主循环（line 185-227）中，每张图：
- `res["boxes"].cpu().numpy()` — GPU→CPU
- `res["scores"].cpu().numpy()` — GPU→CPU
- `torch.tensor(pred_boxes[:, :5])` — CPU→GPU（又搬回去）
- `torch.tensor(gt_boxes)` — CPU→GPU
- `batch_probiou(gt_t, det_t)` — GPU 计算
- `match_predictions(...)` — 内部又转 numpy

每张图 **6 次数据搬运**。

### 瓶颈 2：`batch_probiou` 每图独立调用，小矩阵开销大

每张图调用一次 `batch_probiou(gt_t, det_t)`，矩阵通常 `M×300`（M<20 GT, 300 pred）。对于小矩阵，kernel launch 开销占比大。验证集 500 张图 = 500 次小矩阵调用。

### 瓶颈 3：没有置信度预过滤

`postprocessor` 默认输出 300 个预测（`num_top_queries: 300`），大部分置信度极低，但全部参与 `batch_probiou` 计算。

## 优化方案

### 优化 1：置信度预过滤（最大收益，改动最小）

在 `batch_probiou` 调用前过滤低置信度预测：

```python
conf_mask = pred_scores > 0.001
pred_boxes = pred_boxes[conf_mask]
pred_scores = pred_scores[conf_mask]
pred_labels = pred_labels[conf_mask]
```

- 改动量：~5 行
- 预期收益：300 个预测通常只有 10-50 个置信度 > 0.001，`batch_probiou` 矩阵从 `M×300` 缩小到 `M×20`，IoU 计算量降低 **10-15 倍**
- 优先级：**必做**

### 优化 2：避免 CPU↔GPU 搬运，全程 GPU

保持 tensor 在 GPU 上，只在最终 `all_tp.append()` 时才 `.cpu().numpy()`。需要 GPU 版本的 `match_predictions`。

- 改动量：~30 行（主循环重写 + GPU match 函数）
- 预期收益：**2-3x** 整体加速
- 优先级：**推荐**

### 优化 3：`match_predictions` 改为 GPU 版本

当前 `match_predictions`（line 44-71）全程 numpy，内部做 `iou.cpu().numpy()`。改为 GPU tensor 操作，只在 dedup 步骤用 numpy（或纯 GPU 实现）。

- 改动量：~20 行
- 预期收益：**1.5x** 匹配加速
- 优先级：可选

### 优化 4：跨图批量 IoU（改动较大）

把所有图的预测和 GT 收集起来一次性算 IoU。需要处理"不同图的 pred/gt 不应交叉计算 IoU"的问题（可用 block-diagonal mask 或分段计算）。

- 改动量：~50 行
- 预期收益：~1.2x（验证集图片数 <1000 时收益有限）
- 优先级：不推荐

## 推荐实施顺序

1. 优化 1（5 行，10-15x IoU 加速）
2. 优化 2（30 行，2-3x 整体加速）
3. 优化 3（可选，配合优化 2）

## 涉及文件

- `engine/eval/obb_eval.py` — 主评估循环、`match_predictions`
- `engine/deim/obb_ops.py` — `batch_probiou`（无需改动，只是调用方优化）
