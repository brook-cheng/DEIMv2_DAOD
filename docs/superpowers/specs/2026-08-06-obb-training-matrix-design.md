# DEIMv2 OBB Stage 1/2 功能稳定性训练矩阵设计

日期：2026-08-06
状态：待审阅
前置文档：`docs/superpowers/specs/2026-08-06-obb-decoder-repair-design.md`（两阶段设计，Stage 1/2 已完成）

## 1. 目标与非目标

### 目标
以 `synthetic_exp_020_anrep0_offset_per.yml` 为基线，设计覆盖本轮 DEIMv2 OBB Stage 1/2 全部功能分支的训练实验组，由用户执行。**只验证功能稳定性**：
- 完整 train / resume / test-only eval / OBB infer 链路无崩溃
- 全程 loss 无 NaN/Inf
- checkpoint 落盘、加载、续训契约合法
- 输出契约（PostProcessor 5D OBB、DOTA 导出 8 坐标）合法

### 非目标
- 不设 AP / 损失性能回归硬门槛（仅要求 `mAP50_95 > 0` 作 sanity）
- 不做多 seed 性能复现（R2-RESUME 兼职 R2 的第二个 seed）
- 不涉及 `use_focal_loss=False` 训练（见 §7 排除项）

## 2. 验收标准（功能稳定性）

| 门控 | 通过条件 |
|---|---|
| Gate 1 预检（1 epoch） | 无异常退出；≥1 条有限 loss（无 NaN/Inf）；`last.pth` 落盘；无 CUDA OOM |
| Gate 2 完整训练（80 epoch） | 全程无异常；loss 全程有限（抽查日志）；`last.pth` + `checkpoint0079.pth` + `best_stg2.pth` 落盘；`mAP50_95 > 0` |
| Gate 3 链路 | `--test-only` eval 正常产出指标；infer 对全部图像导出合法 DOTA txt（8 坐标 + 类别 + [0,1] 置信度） |
| resume | 从 epoch-40 checkpoint 续训无异常；`last_epoch` 正确恢复；续训至 80 完成 |

NaN/Inf 处理：任一 run 出现 → 该配置重跑 1 次以区分偶发与缺陷；持续存在 → 上报，不掩盖。

## 3. 基线（已核实的配置事实）

`configs/custom_obb/synthetic_configs/synthetic_exp_020_anrep0_offset_per.yml`：
- 80 epochs（YAML 键为 `epoches`）、输入 256×256、`total_batch_size: 12`
- AMP GradScaler enabled、EMA enabled（decay 0.9998）、`checkpoint_freq: 10`
- `angle_rep: 0`、`offset_scale_source: "pre"`（DEIMTransformer + DEIMCriterion 两处一致）
- `num_denoising: 100`、`box_noise_scale: 0.5`、`use_focal_loss: true`
- `num_queries: 300`、`num_classes: 3`、`box_mode: obb`、`obbox_rep_dim: 6`
- `change_matcher: true`、`matcher_change_epoch: 60`
- 数据：`density_020` 合成椭圆 train/val

已核实的 CLI 与机制：
- `train.py`：`-c` 配置、`-r` resume、`--test-only`（`solver.val()`）、`-u a.b=c` 覆盖、`--seed`
- checkpoint 命名：每 epoch 末写 `last.pth`；`(epoch+1) % checkpoint_freq == 0` 时写 `checkpoint{epoch:04}.pth`（freq=10 → 0009/0019/0029/0039/0049/0059/0069/0079）
- epoch 语义：`last_epoch` 默认 -1（`engine/core/_config.py:58`）→ `start_epoch = last_epoch + 1 = 0` → 循环 `range(start_epoch, epoches)` 为 0-indexed，`epoches: 80` 恰训练 80 个 epoch（epoch 0..79）；`checkpoint0039.pth` = epoch 39 结束 = 已训 40 epoch
- resume：`state_dict` 含 `last_epoch` 与 optimizer/lr_scheduler/ema/scaler 状态；`load_resume_state` 恢复后从 `last_epoch` 续训
- `best_stg2.pth`：`mAP50_95` 超历史最佳且 epoch ≥ stop_epoch(50) 时写
- infer 工具：`test/tool_deimv2_obb_infer.py` 的 `infer_obb_and_export(img_dir, ckpt, config, output_dir, classes_txt, imgsz, max_det, score_threshold, device)`，输出每图 DOTA txt（`x1 y1 x2 y2 x3 y3 x4 y4 class confidence`）

## 4. 运行矩阵（8 runs × 80 epoch）

| # | 名称 | 配置文件 | 与基线差异 | 覆盖分支 | output_dir |
|---|---|---|---|---|---|
| 1 | R0 | `synthetic_exp_020_anrep0_offset_per.yml`（已有） | 无（基线） | rep0 主路径回归锚 | `./outputs/synthetic_exp_020_anrep0_offset_per` |
| 2 | R1 | `synthetic_exp_020_anrep1_offset_per.yml`（已有） | `angle_rep: 1` | rep1 | `./outputs/synthetic_exp_020_anrep1_offset_per` |
| 3 | R2 | `synthetic_exp_020_anrep2_offset_per.yml`（已有） | `angle_rep: 2` | **rep2 核心**（Stage 2 主路径） | `./outputs/synthetic_exp_020_anrep2_offset_per` |
| 4 | R3 | `synthetic_exp_020_anrep3_offset_per.yml`（已有） | `angle_rep: 3` | rep3 | `./outputs/synthetic_exp_020_anrep3_offset_per` |
| 5 | R2-DN0 | 新建 `synthetic_exp_020_anrep2_dn0.yml` | + `num_denoising: 100 → 0` | denoising off 分支 | `./outputs/synthetic_exp_020_anrep2_dn0` |
| 6 | R2-BN0 | 新建 `synthetic_exp_020_anrep2_bn0.yml` | + `box_noise_scale: 0.5 → 0` | 零噪声分支（Stage 2 Task 5 修复点） | `./outputs/synthetic_exp_020_anrep2_bn0` |
| 7 | R2-POST | 新建 `synthetic_exp_020_anrep2_offset_post.yml` | + `offset_scale_source: "pre" → "post"`（DEIMTransformer + DEIMCriterion 两处） | offset post 分支 | `./outputs/synthetic_exp_020_anrep2_offset_post` |
| 8 | R2-RESUME | 复用 R2 配置 | 训练至 epoch 40 中断，`-r checkpoint0039.pth` 续训至 80 | checkpoint save/resume 链路；兼职 R2 第二 seed | `./outputs/synthetic_exp_020_anrep2_offset_per`（同 R2） |

已有配置（R0–R3）已核实：`angle_rep` 分别为 0/1/2/3，`output_dir` 各自唯一，其余字段与基线一致。

## 5. 新建配置文件（3 个）

复制 `synthetic_exp_020_anrep2_offset_per.yml` 并仅修改以下字段：

1. `synthetic_exp_020_anrep2_dn0.yml`：
   - `DEIMTransformer.num_denoising: 0`
   - `output_dir: ./outputs/synthetic_exp_020_anrep2_dn0`
2. `synthetic_exp_020_anrep2_bn0.yml`：
   - `DEIMTransformer.box_noise_scale: 0`
   - `output_dir: ./outputs/synthetic_exp_020_anrep2_bn0`
3. `synthetic_exp_020_anrep2_offset_post.yml`：
   - `DEIMTransformer.offset_scale_source: "post"`
   - `DEIMCriterion.offset_scale_source: "post"`
   - `output_dir: ./outputs/synthetic_exp_020_anrep2_offset_post`

遵循仓库「一配置一文件」惯例，保证 eval/resume 命令零 `-u` 漂移。

## 6. 三层门控与命令清单

所有命令在 `deimv2_daod/` 根目录执行。`<CFG>` 指 §4 表中对应配置文件路径。

### Gate 1 预检（8 个配置 × 1 epoch）
```bash
python train.py -c <CFG> -u epoches=1 print_freq=10 output_dir=./outputs/preflight/<name>
```
- `print_freq=10` 强制高频打印 loss 以便 NaN/Inf 检测；scratch output_dir 避免污染正式目录
- 8 个 `name`：anrep0 / anrep1 / anrep2 / anrep3 / anrep2_dn0 / anrep2_bn0 / anrep2_offset_post

### Gate 1b resume 冒烟（R2-RESUME 专属）
resume 是独立功能路径，需在正式中断前预检：
```bash
# 1. 预检 R2（产出 scratch last.pth）
python train.py -c .../synthetic_exp_020_anrep2_offset_per.yml -u epoches=1 print_freq=10 output_dir=./outputs/preflight/anrep2
# 2. 从 scratch last.pth 续训 1 epoch（预检后 last_epoch=0 → resume 后 start_epoch=1 → epoches=2 时 range(1,2) 恰训 epoch 1）
python train.py -c .../synthetic_exp_020_anrep2_offset_per.yml -u epoches=2 print_freq=10 output_dir=./outputs/preflight/anrep2 -r ./outputs/preflight/anrep2/last.pth
```
- PASS = 两步均无异常、`last_epoch` 从 0 续训（start_epoch=1）、续训后 loss 有限

### Gate 2 完整训练（8 runs）
```bash
# R0–R3（已有配置）
python train.py -c configs/custom_obb/synthetic_configs/synthetic_exp_020_anrep0_offset_per.yml
python train.py -c configs/custom_obb/synthetic_configs/synthetic_exp_020_anrep1_offset_per.yml
python train.py -c configs/custom_obb/synthetic_configs/synthetic_exp_020_anrep2_offset_per.yml
python train.py -c configs/custom_obb/synthetic_configs/synthetic_exp_020_anrep3_offset_per.yml
# R2-DN0 / R2-BN0 / R2-POST（新建配置）
python train.py -c configs/custom_obb/synthetic_configs/synthetic_exp_020_anrep2_dn0.yml
python train.py -c configs/custom_obb/synthetic_configs/synthetic_exp_020_anrep2_bn0.yml
python train.py -c configs/custom_obb/synthetic_configs/synthetic_exp_020_anrep2_offset_post.yml
```

### Gate 3a resume（R2-RESUME，在 R2 完整训练中断后）
```bash
python train.py -c configs/custom_obb/synthetic_configs/synthetic_exp_020_anrep2_offset_per.yml \
  -r ./outputs/synthetic_exp_020_anrep2_offset_per/checkpoint0039.pth
```
- 首选 `checkpoint0039.pth`（epoch 39 结束 = 已训 40 epoch）；若不存在回退最近的 `checkpoint0029/0019/0009.pth`；全部缺失则记录并从头重训
- resume 复用 R2 的 `output_dir`（`last_epoch` 由 checkpoint 状态恢复）

### Gate 3b test-only eval（4 个代表终检）
```bash
python train.py -c <CFG> -r <output_dir>/last.pth --test-only
```
- R0、R2、R2-DN0、R2-POST 各一次（R2-RESUME 的 `last.pth` 即最终 checkpoint，可加测）

### Gate 3c OBB infer（2 个代表）
编辑 `test/tool_deimv2_obb_infer.py` 的 `infoes`，对 R2（+R0）终检 `last.pth` 调用 `infer_obb_and_export`：
- `img_dir`: `//mnt/d/project_data/model_test/deimv2_obb_train_data/synthetic_ellipse/density_020/val`
- `classes_txt`: `//mnt/d/project_data/model_test/deimv2_obb_train_data/synthetic_ellipse/classes.txt`
- `imgsz`: `(256, 256)`（匹配训练尺寸）、`max_det: 300`、`score_threshold: 0.2`、`device: cuda:0`

## 7. 明确排除项：`use_focal_loss=False`

核对传播链后确认 non-focal **不能**作为完整训练配置，也不设 eval-only 项：
- `deim_decoder.py:707`：`dec_score_head = Linear(hidden_dim, num_classes)` → 3 通道，**无背景通道**
- `postprocessor.py:80` non-focal 路径：`F.softmax(logits, dim=-1)[:, :, :-1]` → 假定 `num_classes+1` 通道，3 通道下会切片丢真实类别
- `deim_criterion.py`：仅 focal 路径（`sigmoid_focal_loss`，`one_hot(num_classes+1)[..., :-1]`）
- 覆盖已由 Stage 2 Task 6 单元测试完成（postprocessor 相关回归 147 passed）

## 8. 结果记录

每 run 记录并归档（建议 `docs/superpowers/records/2026-08-06-obb-matrix-results.md`）：

| 字段 | 说明 |
|---|---|
| run / 配置 | 名称 + 配置文件路径 |
| Gate 1 结果 | pass/fail + 首条 loss 值 |
| Gate 2 结果 | 完成状态、末 epoch loss、`mAP50_95`（不设门槛）、checkpoint 文件清单、异常摘要 |
| Gate 3 结果 | eval 指标、infer 成功图数 / 总图数、输出样例合法性 |
| NaN/Inf 事件 | 发生位置、重跑结果、结论（偶发 / 缺陷） |

## 9. 风险与回退

- R2-RESUME 断点缺失 → §6 Gate 3a 回退链
- 任一完整训练中途崩溃（非 NaN）→ 按 resume 机制从最近 checkpoint 续训，记录事件
- 预检即失败 → 阻止该配置进入 Gate 2，上报并排查（区分配置错误与代码缺陷）
- infer 工具为既有工具（dlzdt 流程已使用），仅替换参数，不做代码改动

## 10. 范围与后续

- 本设计仅覆盖 Stage 1/2 相关功能分支；Stage 2 集成门已 PASS（196 tests）
- 后续（独立任务）：YAML 配置清理、`clamp_offsets=True` 边界守护（来自 repair design 文档后续项）
