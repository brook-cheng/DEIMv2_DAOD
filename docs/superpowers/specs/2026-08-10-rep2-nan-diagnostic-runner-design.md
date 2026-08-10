# DEIMv2-OBB rep2 NaN 远程诊断 runner 设计

日期：2026-08-10

状态：设计已确认；默认单 GPU，兼容完整训练 checkpoint 与 weights-only checkpoint。

## 1. 背景

`angle_rep=2` 与 `angle_rep=2 + use_gate_fusion=true` 两次训练分别在 epoch 88 和
epoch 115 的 `loss.backward()` 后出现全链路非有限梯度。现有日志只能证明 forward、
criterion loss 与 total loss 在异常 step 前仍为有限值，无法定位第一个产生 NaN 的
backward op。训练服务器与当前开发环境分离，因此需要一个可移植的单 GPU 诊断脚本，
在远程服务器复现异常并完整保存现场。

当前最强候选是 rep2 外接矩形偏移解码中的退化边进入 `atan2(0, 0)`：该输入前向结果
有限、反向梯度为 NaN。但本设计只负责收集证据，不预设或修改根因。

## 2. 目标与非目标

### 2.1 目标

1. 从指定 YAML 与 checkpoint 构建 rep2 训练环境，执行有界的单 GPU 短程训练。
2. 自动识别完整训练 checkpoint 与 weights-only checkpoint，并记录恢复完整度。
3. 通过 autograd anomaly detection、梯度全扫描和 rep2 几何探针定位首个异常算子。
4. 异常退出前持久化触发 batch、targets、outputs、loss、梯度统计、几何统计、模型状态、
   optimizer 状态、环境信息与完整 traceback。
5. 日常 step 仅写轻量 JSONL；只有失败 step 保存完整张量，限制磁盘占用。
6. 产物目录可以整体打包带回当前环境离线分析。

### 2.2 非目标

1. 不修改正式训练循环 `engine/solver/det_engine.py`。
2. 不尝试修复 rep2 数值问题。
3. 不执行验证、Comet 上报、EMA 指标比较或多 GPU DDP。
4. weights-only 恢复只用于数值扫描，不宣称精确复现原训练轨迹。
5. 不在每 step 保存完整模型或 batch。

## 3. 文件与接口

新增：

- `test/tool_diagnose_rep2_nan.py`：远程诊断 runner。
- `test/test_rep2_nan_diagnostic.py`：纯 CPU 单元测试。

CLI：

```bash
python test/tool_diagnose_rep2_nan.py \
  --config configs/custom_obb/dlzdt/ablation/abl_rep2.yml \
  --checkpoint /path/to/checkpoint.pth \
  --output-dir /path/to/rep2_nan_diagnostic \
  --device cuda:0 \
  --max-epochs 10 \
  --detect-anomaly
```

必填参数：

- `--config`：训练 YAML。
- `--checkpoint`：完整或 weights-only checkpoint。
- `--output-dir`：本次诊断的独立输出目录；已存在非空目录默认拒绝覆盖。

可选参数：

- `--device`，默认 `cuda:0`。
- `--max-epochs`，默认 `10`。
- `--max-steps-per-epoch`，默认不限制。
- `--start-epoch`：完整 checkpoint 自动推断；weights-only 未提供时默认 `0` 并记录告警。
- `--seed`：未指定时沿用配置 seed。
- `--detect-anomaly/--no-detect-anomaly`，默认启用。
- `--save-every-steps`，默认 `50`，只写轻量统计与进度 marker。
- `--overwrite`：显式允许覆盖既有输出目录。

## 4. Checkpoint 恢复语义

脚本首先读取 checkpoint 顶层键并分类：

### 4.1 完整训练 checkpoint

若存在 model/EMA、optimizer、lr_scheduler（或项目等价键）与 epoch：

- 恢复训练 model 权重（优先非 EMA model；若仅有 EMA 则使用 EMA 并记录）。
- 恢复 optimizer、scheduler 与起始 epoch。
- 若某一状态加载失败，不静默降级；记录具体错误并将恢复级别降为 `partial`。
- `recovery_fidelity` 为 `full` 或 `partial`。

### 4.2 Weights-only checkpoint

- 支持顶层 state dict、`model`、`ema.module` 等项目既有形态。
- optimizer/scheduler 由 YAML 重建。
- 起始 epoch 使用 `--start-epoch`，未提供则为 `0`。
- `recovery_fidelity=weights_only`，报告明确说明结果不等同原训练轨迹复现。

`run_manifest.json` 必须列出 checkpoint 顶层键、每个状态的恢复结果、缺失键和
`load_state_dict` missing/unexpected keys。

## 5. 诊断训练循环

脚本复用 YAMLConfig 构建：

- model
- criterion
- optimizer
- scheduler
- train dataloader

但不调用 `solver.fit()`；脚本拥有独立、最小的 step 循环：

1. 把 samples/targets 移至指定设备。
2. 按正式训练一致的 BF16 autocast 执行 model forward。
3. 把浮点 outputs 递归转为 FP32，关闭 autocast 计算 criterion。
4. 检查 outputs、每项 loss 与 total loss 是否有限。
5. 开启 anomaly detection 执行 backward。
6. 扫描**全部**参数梯度并记录所有非有限参数；不在首个参数处停止。
7. 仅在梯度全部有限时 clip、optimizer.step 和 scheduler.step。
8. 每 step 向 `events.jsonl` 写一条记录。

异常 step 绝不执行 optimizer.step。

## 6. 几何探针

### 6.1 运行前自检

脚本启动时运行 CPU/GPU probe：

```text
atan2(0, 0): forward finite, backward non-finite
```

结果写入 manifest，用于确认远程 PyTorch/CUDA 版本的实际行为。

### 6.2 训练时统计

在不改变生产张量的前提下，通过 forward hook 或对输出的只读派生统计记录：

- `pred_boxes`/`ref_points`/`pred_corners` 的 finite 比例、min/max/absmax。
- 预测 w/h 的最小值、接近零数量。
- rep2 外接矩形 `(cx,cy,w,h,epsilon,eta)` 的 w/h、epsilon/eta 范围。
- 由 rep2 顶点构造出的两条连续边的长度最小值。
- `edge_ab == (0,0)`、`edge_bc == (0,0)` 和低于阈值的近退化边数量。
- `atan2` 输入 `(w_dx,w_dy)` 同时为零/近零的数量。

探针统计统一 detach 后执行，不参与反向，不修改模型行为。

## 7. 产物目录

```text
output-dir/
  run_manifest.json
  command.txt
  stdout.log
  events.jsonl
  progress.json
  geometry_probe.json
  failure/
    traceback.txt
    failure_summary.json
    trigger_batch.pt
    outputs.pt
    losses.pt
    geometry_snapshot.pt
    gradients_summary.json
    model_state.pt
    optimizer_state.pt
```

### 7.1 `run_manifest.json`

记录：时间、hostname、OS、Python/PyTorch/CUDA/cuDNN、GPU 型号、git commit/diff 状态、
config 路径和 SHA256、checkpoint 路径和 SHA256、CLI、seed、恢复完整度、恢复状态清单、
数据集路径、dataloader 长度、AMP dtype 与 anomaly detection 状态。

### 7.2 `events.jsonl`

每 step 一行：epoch、step、global_step、LR、loss 总值和分项、输出有限性统计、几何摘要、
grad norm、step 时长、显存占用。不得写完整 tensor。

### 7.3 failure 现场

- 所有 tensor 保存前 detach、移到 CPU。
- `trigger_batch.pt` 包含 samples、targets 和可用的数据集索引/图片路径。
- `gradients_summary.json` 扫描全部参数，记录每个异常梯度的 NaN/Inf 数、形状、dtype、
  finite 部分 min/max/norm；不保存完整梯度 tensor。
- model/optimizer 状态只在 rank 0（本设计固定单进程）保存。
- 任一产物保存失败不得覆盖原 traceback；将次级保存错误追加到 failure summary。

## 8. 退出状态

- `0`：达到运行上限且未发现非有限值。
- `2`：成功捕获非有限 forward/loss/backward/gradient，并完整或部分保存 failure 现场。
- `3`：配置、checkpoint、数据集或恢复阶段失败。
- `4`：CUDA OOM 或其他运行时异常；仍保存 traceback 与已获取现场。

## 9. 测试

纯 CPU 测试不依赖真实数据集/checkpoint：

1. `atan2(0,0)` probe 证明 forward finite、backward non-finite。
2. Tensor 统计正确区分 finite/NaN/+Inf/-Inf。
3. 梯度全扫描能一次报告多个异常参数，不在第一个参数停止。
4. checkpoint 分类覆盖完整、partial、weights-only 与非法格式。
5. artifact writer 在模拟 backward 异常时生成规定文件，tensor 已移到 CPU。
6. 非空 output-dir 默认拒绝覆盖，`--overwrite` 显式放行。
7. runner dry-run 使用小型 toy model/criterion/dataloader，有限路径退出 0、NaN backward
   路径退出 2。

## 10. 远程运行与回传

优先从中断前最近完整 checkpoint 执行：

```bash
CUDA_VISIBLE_DEVICES=0 python test/tool_diagnose_rep2_nan.py \
  --config configs/custom_obb/dlzdt/ablation/abl_rep2_fused.yml \
  --checkpoint outputs/.../checkpoint0110.pth \
  --output-dir diagnostics/rep2_fused_e110 \
  --device cuda:0 \
  --max-epochs 10 \
  --detect-anomaly \
  2>&1 | tee diagnostics/rep2_fused_e110/launcher.log
```

完成后打包整个目录：

```bash
tar -czf rep2_fused_e110_diagnostic.tar.gz diagnostics/rep2_fused_e110
```

若只有 weights-only checkpoint，必须保留 manifest 中的恢复告警，并避免把未复现解读为
根因不存在。

## 11. 成功判据

诊断工具成功不要求复现 NaN；满足以下任一条件即可：

1. 捕获异常并得到 autograd traceback、触发 batch 和完整梯度/几何摘要；或
2. 在限定窗口内未复现，但提供可审计的环境、恢复完整度和逐 step 证据，足以决定下一次
   更接近原状态的运行条件。
