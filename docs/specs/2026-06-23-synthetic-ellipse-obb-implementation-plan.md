# 合成椭圆 OBB 数据集 — 实施计划

> 基于：`docs/specs/2026-06-23-synthetic-ellipse-obb-dataset-design.md`
> 创建日期：2026-06-23
> 状态：待执行

---

## 阶段〇：数据样例生成与审查（闸门）

**目标**：生成 50 张覆盖全部密度的样例，截图审查通过后再进入大批量生成。

### Step 0.1 — 编写样例生成脚本

**文件**：`scripts/generate_synthetic_ellipse_samples.py`

**职责**：
- 读取密度列表 `[1, 2, 5, 10, 20, 50, 100]`
- 每个密度生成 ~7 张图 → 共 ~50 张
- 每张图输出三个文件：
  - `{name}.png` — 原始合成图
  - `{name}.txt` — DOTA 格式标注（8 顶点 + 类别名 + difficulty）
  - `{name}_viz.png` — 可视化叠加图（灰色背景 + 红色椭圆 + 绿色 OBB 外接框 + 蓝色顶点 + 类别标签）

**关键实现细节**：
- 椭圆绘制：`cv2.ellipse(img, center=(cx,cy), axes=(a,b), angle=theta_deg, startAngle=0, endAngle=360, color=rgb, thickness=-1)`
- OBB 外接框顶点计算：见 spec 3.6.1 伪代码
- DOTA 标注写入：8 个顶点用空格分隔，浮点数保留 1 位小数
- 可视化：在同一个 `np.array` 上画椭圆 + 叠加 OBB 框（绿色 polyline）+ 顶点圆点（蓝色）+ 类别名（白色文字）
- 碰撞检测：OBB 四顶点 → `shapely.geometry.Polygon` → `.intersection(其他多边形).area / .union(其他多边形).area < 0.05`

**输出目录**：`synthetic_ellipse/samples/`

### Step 0.2 — 运行样例生成

```bash
python scripts/generate_synthetic_ellipse_samples.py
```

### Step 0.3 — 用户审查

审查要点：
- [ ] 椭圆颜色可区分（红/绿/蓝饱和度足够）
- [ ] OBB 框紧密贴合椭圆
- [ ] 无重叠椭圆（高密度场景下碰撞检测生效）
- [ ] 高密度（50, 100 GT/图）场景不过度拥挤
- [ ] DOTA 标注格式正确（8 顶点 + 类别名 + 0）

**闸门**：审查通过 → 进入阶段一。不通过 → 调整生成参数 → 重新 Step 0.1–0.3。

---

## 阶段一：完整数据集生成

**目标**：生成 7 组 × 500 张 = 3500 张的完整数据集。

### Step 1.1 — 编写完整生成脚本

**文件**：`scripts/generate_synthetic_ellipse.py`

**职责**：
- 遍历 7 个密度梯度，每梯度生成 `train/400 + val/100`
- 输出结构：
  ```
  synthetic_ellipse/
    classes.txt
    density_001/
      train/   (400 png + 400 txt)
      val/     (100 png + 100 txt)
    density_002/ ...
    ...
    density_100/ ...
  ```
- 文件名：`{density_编号:G04d}_{index:06d}.png` / `.txt`

### Step 1.2 — 编写验证脚本

**文件**：`scripts/validate_synthetic_dataset.py`

**职责**：
- 校验每个密度梯度的 train/val 图片数 = 400/100
- 校验每张图的 GT 数量 ∈ [目标数-20%, 目标数+20%]（碰撞检测可能导致少量 GT 丢弃）
- 校验所有 OBB 无重叠（IoU < 0.05）
- 校验无越界标注（4 顶点全在 [0, 255] 内）
- 输出 `dataset_stats.json`：
  ```json
  {
    "density_001": {
      "train": {"images": 400, "total_gt": 398, "avg_gt": 0.99, "min": 1, "max": 1},
      "val":   {"images": 100, "total_gt": 99,  "avg_gt": 0.99, "min": 1, "max": 1}
    },
    ...
  }
  ```

### Step 1.3 — 运行生成

```bash
python scripts/generate_synthetic_ellipse.py && python scripts/validate_synthetic_dataset.py
```

---

## 阶段二：批量短训练（7 组对照实验）

**目标**：在 7 个密度的合成数据集上分别做 30 epoch 短训练。

### Step 2.1 — 编写配置生成脚本

**文件**：`scripts/generate_synthetic_configs.py`

**职责**：从 `configs/custom_obb/deimv2_obb_sp.yml` 模板生成 7 个配置：

```yaml
# configs/custom_obb/synthetic_exp_001.yml
__include__: ['../runtime.yml', '../base/dataloader.yml', '../base/optimizer.yml',
              '../dataset/dota_detection.yml', './dataset_common.yml', './deimv2_obb_common.yml']
output_dir: ./outputs/synthetic_exp_001
# ... 其他超参
train_dataloader:
  dataset:
    img_folder: synthetic_ellipse/density_001/train
    ann_folder: synthetic_ellipse/density_001/train
    classes_file: synthetic_ellipse/classes.txt
```

每组配置的关键差异：

| 配置 | 密度 | 数据集路径 |
|------|------|-----------|
| `synthetic_exp_001.yml` | 1 | `density_001` |
| `synthetic_exp_002.yml` | 2 | `density_002` |
| `synthetic_exp_005.yml` | 5 | `density_005` |
| `synthetic_exp_010.yml` | 10 | `density_010` |
| `synthetic_exp_020.yml` | 20 | `density_020` |
| `synthetic_exp_050.yml` | 50 | `density_050` |
| `synthetic_exp_100.yml` | 100 | `density_100` |

**共享参数**（7 组一致）：

```yaml
epoches: 30
flat_epoch: 18
no_aug_epoch: 3
eval_spatial_size: [256, 256]
Mosaic: {output_size: 128, probability: 0.5}  # 256/2 适配缩小一半
OBBResize: {size: [256, 256]}
```

### Step 2.2 — 训练执行

每实验：

```bash
python train.py --config configs/custom_obb/synthetic_exp_{density}.yml --comet-project deimv2_obb_synthetic
```

并行策略：1 GPU 串行 7 组，或 2 GPU 各跑一半。

监控指标（Comet）：
- `loss_mal`、`loss_fgl`（每 100 step）
- `val_precision`、`val_recall`、`val_AP50`、`val_mAP`（每 epoch）

### Step 2.3 — 收集中间 checkpoint

每实验保留：
- `last.pth`（30 epoch 完成后的权重）
- Comet experiment key（用于后续对比分析）

---

## 阶段三：推理诊断与交叉分析

**目标**：对 7 组实验分别运行诊断脚本，汇总分数分布对比。

### Step 3.1 — 批量推理诊断

**脚本**：`scripts/run_infer_diag_batch.sh`

每实验：

```bash
python test/test_infer_diag.py \
  --ckpt outputs/synthetic_exp_{density}/last.pth \
  --config configs/custom_obb/synthetic_exp_{density}.yml \
  --num 20 \
  --conf 0.1
```

输出目录：`test/outputs/infer_diag/density_{density}/`

收集 `score_dist.txt`。

### Step 3.2 — 交叉分析脚本

**文件**：`scripts/analyze_density_experiment.py`

**职责**：
1. 读取 7 组 `score_dist.txt`
2. 输出：
   - **分数分布热力图**：7 行（密度）× 20 bin（分数区间），颜色编码预测数量
   - **密度-分数分位数曲线**：多线图（P10, P25, P50, P75, P90, P95）
   - **密度-mAP 对比图**：散点图，X=GT密度，Y=mAP
   - **损失收敛对比**：7 条 `loss_fgl` 曲线叠加
3. 自动判断：输出第一组"分数分布出现 >0.3 明显峰值"的密度阈值

### Step 3.3 — 结论输出

脚本输出 `experiment_report.md` 包含：
- 以上所有图表
- 密度阈值判定结果
- 对照结论（H1 确认 / 排除 / 部分成立）

---

## 文件清单

| 文件 | 用途 | 阶段 |
|------|------|------|
| `scripts/generate_synthetic_ellipse_samples.py` | 样例生成（含可视化） | 〇 |
| `scripts/generate_synthetic_ellipse.py` | 完整数据集生成 | 一 |
| `scripts/validate_synthetic_dataset.py` | 数据集校验 | 一 |
| `scripts/generate_synthetic_configs.py` | 训练配置生成 | 二 |
| `scripts/run_infer_diag_batch.sh` | 批量推理诊断 | 三 |
| `scripts/analyze_density_experiment.py` | 交叉分析 + 报告生成 | 三 |
| `synthetic_ellipse/samples/` | 50 张样例（审查用） | 〇 |
| `synthetic_ellipse/density_*/` | 7 组完整数据集 | 一 |
| `configs/custom_obb/synthetic_exp_*.yml` | 7 个训练配置 | 二 |
| `test/outputs/infer_diag/density_*/` | 7 组推理诊断结果 | 三 |
| `experiment_report.md` | 最终分析报告 | 三 |

---

## 风险与缓解

| 风险 | 缓解 |
|------|------|
| 高密度场景碰撞检测失败率过高，无法生成足够 GT | 放宽碰撞 IoU 阈值 0.05→0.10，或在样例审查中评估是否需要缩小椭圆尺寸 |
| 30 epoch 不够分离分数（收敛速度慢） | 改用 50 epoch，或在分析脚本中标注"可能需要更长训练" |
| 合成数据集太简单导致分数分离过度（过拟合），无法迁移到真实数据 | 这不影响 H1 验证——H1 只关心"密集数据能否让分数分离"，过拟合是后续问题 |

---

> **下一步**：按阶段〇 → 一 → 二 → 三顺序执行。每阶段完成后人工确认再进入下一阶段。
