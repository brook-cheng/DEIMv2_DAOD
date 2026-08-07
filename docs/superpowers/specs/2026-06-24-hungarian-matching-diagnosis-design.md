# 匈牙利匹配正确性诊断实验设计

> 创建日期：2026-06-24
> 目的：验证 DEIMv2-OBB 的匈牙利匹配是否正常工作，诊断分类精度低是否由匹配质量导致
> 状态：设计阶段，待评审

---

## 1. 背景

在 density_020 合成数据集上训练后，观察到：分类分数整体偏低且无跨 IoU 的区分力、存在重复预测。这与 DETR "一个 query 对应一个 GT" 的设计直觉矛盾。

需要验证：
- Q1: 匈牙利匹配是否做到了真正的一对一？
- Q2: 匹配代价函数是否有足够的区分力？
- Q3: 分类分数是否与匹配质量（IoU）正相关？

---

## 2. 实验方案

**方案 A（首选）**：在 inference 阶段复现匹配 —— 加载 trained checkpoint，对验证集做前向推理，用 matcher 对 pred ↔ GT 做匈牙利匹配，统计 assignment 质量。不修改训练代码。

**方案 B（备选）**：若方案 A 无法充分定位问题，改为在训练过程中 hook matcher，记录每个 batch 的匹配矩阵和代价分布。需要修改 criterion 代码。

---

## 3. 数据收集

对 density_020 验证集（100 张图，每图 20 GT），每张图收集：

### 3.1 从 matcher 获取
- 匹配矩阵：300 query → GT assignment（或标记为未匹配）
- 代价矩阵：300 × N_GT 的 4 维代价（class + bbox + chamfer + probiou）
- 匹配对 ProbIoU

### 3.2 从模型输出获取
- 每个 query 的分类 logits（sigmoid 后为 scores）
- 每个 query 的预测 OBB `(cx, cy, w, h, θ)`
- 类别预测

### 3.3 从 GT 获取
- 真实 OBB 和类别

---

## 4. 诊断指标与可视化

### 4.1 Q1：一对多检测

**问题**：一个 GT 是否被多个 query 抢占？是否存在完全未被匹配的 GT？

| 输出 | 说明 |
|------|------|
| per-GT query 数分布柱状图 | X=分配给该 GT 的 query 数（0,1,2,3+），Y=GT 数量 |
| 匹配可视化 | 选 4 张典型图，绿=GT + 匹配数标注，红=匹配预测框（深浅=IoU），黄虚线=未匹配预测框（透明度=分数） |

**判断标准**：理想情况每个 GT 恰好 1 个匹配 query。偏差超过 10% 则 flag。

### 4.2 Q2：代价函数区分度

**问题**：matcher 的 4 维代价能否有效区分"真正的匹配"和"碰巧的匹配"？

| 输出 | 说明 |
|------|------|
| 代价分量对比直方图 | 4 子图（class/bbox/chamfer/probiou），各含二分布（蓝=匹配对 / 红=未匹配最优候选），X=代价，Y=频数 |
| 代价矩阵热力图 | 选 1 张典型图，X=GT(20 列)，Y=top-50 query(行)，颜色=总代价，白星=最优匹配。看最优匹配是否孤立 |

**判断标准**：匹配对代价应显著低于未匹配对。若分布重叠>30%，则 matcher 区分力不足。

### 4.3 Q3：分数-质量相关性

**问题**：分类分数能否反映匹配的 IoU 质量？

| 输出 | 说明 |
|------|------|
| IoU-分数散点图 | X=ProbIoU，Y=分类分数。叠加线性回归 + Pearson r |
| IoU 分桶箱线图 | X=IoU 桶(0-0.2,...,0.8-1.0)，Y=分类分数。显示 mean/median/IQR |

**判断标准**：正相关（r>0.3 或显著单调递增趋势）。平线=分类分数与质量无关。

### 4.4 全局汇总

输出 `matching_report.txt`：

```
=== 匹配质量汇总 (100 images) ===

Q1: One-to-Many
  Total GT: 2000
  GT with 0 matches: N (N%)
  GT with 1 match:  N (N%)
  GT with 2+ matches: N (N%)

Q2: Cost Discriminability
  Matched cost mean ± std:   X ± Y
  Unmatched cost mean ± std: X ± Y
  Cost separation ratio:     Xx

Q3: Score-IoU Correlation
  Pearson r = X
  Spearman ρ = Y

CONCLUSION: ...
```

---

## 5. 脚本结构

**文件**：`scripts/diagnose_hungarian_matching.py`

**功能模块**：
1. `load_model()` — 加载 checkpoint + config
2. `run_matching()` — 前向推理 + 调用 matcher 获取 assignment/代价
3. `analyze_q1()` — 一对多统计 + 可视化
4. `analyze_q2()` — 代价区分度直方图 + 热力图
5. `analyze_q3()` — 分数-IoU 散点图 + 箱线图
6. `generate_report()` — 全局汇总

**输出目录**：`test/outputs/matching_diag/`

**依赖**：复用 `engine/deim/matcher.py` 和 `engine/deim/obb_ops.py`

---

## 6. 判定逻辑

```
Q1: 一对一匹配率 < 85% → Flag: 匹配有问题
Q2: 代价分离比 < 2x → Flag: matcher 区分力不足
Q3: Pearson r < 0.2 → Flag: 分类分数与匹配质量无关

若 Q1-Q3 均正常 → 匹配合格，问题在 decoder 耦合
若 Q2/Q3 异常 → 匹配是精度瓶颈，需先修复匹配再考虑解耦
```

---

## 7. 自检

- [x] Q1-Q3 均有明确的量化判定标准
- [x] 每张可视化输出有对应的判断标准
- [x] 方案 A/B 有明确的适用场景和切换条件
- [x] 依赖文件已确认存在（matcher.py, obb_ops.py, checkpoint）
- [x] 无 TBD 或模糊描述
