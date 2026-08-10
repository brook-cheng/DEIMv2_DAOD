# DEIMv2-OBB rep2 稳定 atan2 反向传播设计

日期：2026-08-10

状态：设计已确认；采用 forward 不变、仅稳定 backward 分母的最小修复。

## 1. 背景与诊断结论

`angle_rep=2 + use_gate_fusion=true` 的完整 checkpoint 已由
`test/tool_diagnose_rep2_nan.py` 在单 GPU 上恢复并复现异常。诊断现场位于
`test/data/outputs/diagnose_rep2_nan`，关键证据如下：

- checkpoint 以 `fidelity=full` 恢复，model 与 optimizer 均加载成功，起始 epoch 为 115；
- epoch 115 的 step 0-9 完成 forward、FP32 criterion、backward、梯度检查和 optimizer step；
- step 10 / global step 59810 的公开 forward outputs 与全部 loss 均为有限值；
- `loss.backward()` 在 `Atan2Backward0` 抛出非有限梯度异常；
- forward 调用路径为
  `DEIMTransformer.forward -> enc_topk_bboxes_list ->
  external_xywh_rect_to_oriented_box -> external_xyxy_rect_to_oriented_box ->
  torch.atan2(w_dy, w_dx)`；
- 失败现场的最小边长约为 `3.2e-5`，并存在严格为零的单轴分量；
- 独立自检证明 `atan2(0, 0)` forward 有限但 backward 梯度为 NaN。

因此，根因是 rep2 外接矩形偏移解码在极小或退化宽边向量上执行原生
`torch.atan2`，其 backward 分母 `x^2 + y^2` 在 BF16/FP32 混合训练图中可能为零或
过小，产生 NaN 或不可接受的梯度放大。该问题发生在 decoder forward 构建的训练图中，
不是 criterion、optimizer 或公开 forward 输出首先变为非有限。

## 2. 目标与非目标

### 2.1 目标

1. 保持正常和退化输入上的 angle forward 值与当前 `torch.atan2` 一致。
2. 保证 `(0, 0)`、轴对齐极小边和双轴极小边的 backward 梯度均为有限值。
3. 只修改 rep2 几何反解中的角度梯度，不修改 box、offset、loss 或 decoder 结构。
4. 使用已保存的失败 batch 定点验证修复，并越过原失败 global step 59810。
5. 保持 rep0、rep1、rep3 的训练和推理行为不变。

### 2.2 非目标

1. 不重新设计 rep2 六维表示。
2. 不在训练路径 clamp `epsilon`、`eta` 或外接矩形。
3. 不改变 longer-edge-as-width、角度范围 `[0, pi)` 或现有 roundtrip 约定。
4. 不把修复扩大为通用数学库或修改所有项目中的 `torch.atan2`。
5. 不因本修复直接启动完整 200 epoch 重训。

## 3. 方案比较与选择

### 3.1 采用：forward 原样、backward 分母下限

新增私有稳定算子。Forward 直接返回：

```python
torch.atan2(y, x)
```

Backward 使用 atan2 的解析导数，但对平方半径设置下限：

```text
r2_safe = max(x^2 + y^2, eps)
dL/dy = dL/dtheta * x / r2_safe
dL/dx = -dL/dtheta * y / r2_safe
```

选择理由：

- forward 不发生偏移，正常 OBB 与推理结果不变；
- 修改只作用于病态区域的梯度；
- `(0, 0)` 得到零梯度，而不是 NaN；
- `set eps=1e-9` 与现有边长 `sqrt(... + eps)` 稳定项保持同一尺度；
- 失败现场最小边长约等于 `sqrt(1e-9)`，该阈值直接覆盖已观测奇点。

### 3.2 拒绝：向 x 或 y 添加 epsilon

`atan2(y, x + eps)` 或用 `where` 替换零分量会改变轴对齐和象限边界处的 forward
角度，并可能引入系统性方向偏差。`torch.where(valid, atan2(...), fallback)` 也不能可靠
避免无效分支的 `Atan2Backward0` 被构图和求导。

### 3.3 拒绝：训练路径 clamp offsets

`clamp_offsets=True` 会改变 loss-bearing rep2 表示及越界 offset 的梯度，属于算法行为
变化。该选项继续只用于 detached/eval-safe decode，不作为本次数值修复。

## 4. 稳定算子设计

### 4.1 文件与接口

修改：

- `engine/deim/obb_geometry.py`
- `test/test_obb_adr_geometry.py`

新增私有实现，不增加公共配置项：

私有接口固定为：

- `_StableAtan2.forward(ctx, y: Tensor, x: Tensor, eps: float) -> Tensor`
- `_StableAtan2.backward(ctx, grad_output: Tensor) -> tuple[Tensor, Tensor, None]`
- `_stable_atan2(y: Tensor, x: Tensor, eps: float) -> Tensor`

`external_xyxy_rect_to_oriented_box()` 的公开签名保持不变，现有 `eps=1e-9` 同时用于：

- `len_ab`、`len_bc` 的平方根稳定项；
- `_stable_atan2` backward 的 `r2_safe` 下限。

### 4.2 Forward 合约

`_StableAtan2.forward()` 必须直接调用 `torch.atan2(y, x)`，不得预处理、clamp 或偏移输入。
对正常有限输入，输出应与原生 `torch.atan2` 使用相同 dtype、device、shape 和数值。

后续继续执行：

```python
theta = torch.remainder(theta, torch.pi)
```

因此外部 OBB 角度范围和象限语义不变。

### 4.3 Backward 合约

Backward 中，若保存的输入或上游梯度为 FP16/BF16，计算临时提升到 FP32：

```python
calc_dtype = torch.float32 if input dtype is float16/bfloat16 else input dtype
```

然后：

1. 计算 `r2 = x^2 + y^2`；
2. 使用 `r2.clamp_min(eps)`；
3. 按解析公式计算 `grad_y`、`grad_x`；
4. 将梯度转换回对应输入 dtype；
5. `eps` 不需要梯度。

行为约定：

- `r2 >= eps`：梯度与原生 atan2 解析梯度一致；
- `0 < r2 < eps`：保持方向，限制梯度放大；
- `x == 0 and y == 0`：两个梯度均为零；
- 非有限输入不做掩盖，继续由现有 forward/loss 有限性检查捕获。

本修复不承诺二阶梯度；当前训练链路只需要一阶 backward。

## 5. 单元测试

### 5.1 原生失败锁定

先添加测试证明当前实现会在 `(0, 0)` 上产生非有限 backward，确保测试能锁定真实缺陷。
随后改为断言 `_stable_atan2` 的两个输入梯度均有限。

### 5.2 必测输入

参数化覆盖 FP32，并在 CUDA 可用时覆盖 BF16 autocast：

- `(x, y) = (0, 0)`；
- `(0, 1e-8)`、`(1e-8, 0)`；
- `(0, -1e-8)`、`(-1e-8, 0)`；
- `(1e-8, 1e-8)`；
- 正常四象限输入；
- 当前失败尺度附近 `r ~= sqrt(1e-9)` 的输入。

所有退化输入必须满足：

- forward 有限；
- backward 不抛 anomaly；
- `x.grad`、`y.grad` 有限。

### 5.3 Forward 等价

对正常输入和退化输入分别比较：

```python
actual = _stable_atan2(y, x, eps)
expected = torch.atan2(y, x)
```

要求 dtype、shape 一致，数值使用 `rtol=0, atol=0` 比较。若特定设备内核不支持位级比较，
只允许退化为该 dtype 的最小合理容差，并在测试中明确原因。

### 5.4 几何函数回归

为 `external_xyxy_rect_to_oriented_box()` 增加：

- 极小外接矩形 backward 有限；
- 零尺寸外接矩形 backward 有限；
- 正常 roundtrip 输出不变；
- `clamp_offsets=False` 仍不修改或截断 offset；
- `clamp_offsets=True` 的既有 eval-safe 行为不变。

## 6. 失败现场回放

新增独立工具：

- `test/tool_replay_rep2_nan_failure.py`
- `test/test_rep2_nan_failure_replay.py`

不继续扩展已有 900 行诊断 runner。Replay 工具只负责加载一次保存现场并执行一个 step。

CLI：

```bash
python test/tool_replay_rep2_nan_failure.py \
  --config configs/custom_obb/dlzdt/ablation/abl_rep2_fused.yml \
  --failure-dir test/data/outputs/diagnose_rep2_nan/failure \
  --device cuda:0 \
  --detect-anomaly
```

数据流：

1. 使用 YAMLConfig 构建 model、criterion、optimizer；
2. 加载 `model_state.pt`、`optimizer_state.pt`；
3. 加载 `trigger_batch.pt`；
4. BF16 autocast model forward；
5. 递归将浮点 outputs 转 FP32；
6. 在 autocast 外执行 criterion；
7. 检查 outputs、loss 和 total loss 有限；
8. 执行 `loss.backward()`；
9. 扫描全部参数梯度；
10. 默认不执行 optimizer step；显式 `--step-optimizer` 时才 clip 和 step。

Replay 输出到 stdout，并返回：

- `0`：forward/loss/backward/gradient 全部有限；
- `2`：捕获数值异常；
- `3`：配置或现场加载失败；
- `4`：OOM 或其他运行时异常。

第一轮验收只执行 backward；第二轮增加 `--step-optimizer`，并验证更新后的参数与 optimizer
浮点状态全部有限。

## 7. Checkpoint 短程验证

### 7.1 100-step 验证

```bash
python test/tool_diagnose_rep2_nan.py \
  --config configs/custom_obb/dlzdt/ablation/abl_rep2_fused.yml \
  --checkpoint outputs/deimv2_obb_dlzdt_sp_fz_ablation/abl_rep2_fused/last.pth \
  --output-dir test/data/outputs/diagnose_rep2_nan_fixed_100 \
  --max-epochs 1 \
  --max-steps-per-epoch 100 \
  --detect-anomaly
```

必须满足：

- 越过原失败 step 10 / global step 59810；
- `events.jsonl` 含 100 条有限 step 记录；
- 无 `failure/` 目录；
- 所有 loss、grad norm 和参数梯度有限；
- 退出码为 0。

### 7.2 完整 epoch 验证

100-step 通过后，运行完整 520 steps：

```bash
python test/tool_diagnose_rep2_nan.py \
  --config configs/custom_obb/dlzdt/ablation/abl_rep2_fused.yml \
  --checkpoint outputs/deimv2_obb_dlzdt_sp_fz_ablation/abl_rep2_fused/last.pth \
  --output-dir test/data/outputs/diagnose_rep2_nan_fixed_epoch \
  --max-epochs 1 \
  --detect-anomaly
```

必须满足：

- `progress.json` 含 `done=true`；
- `events.jsonl` 含 520 条记录；
- 无 `failure/` 目录；
- 退出码为 0。

### 7.3 后续训练止损线

完整 epoch 通过后，只建议继续 5-10 epoch 观察 loss 与验证指标。如果出现新的首个异常算子，
停止添加补丁并重新诊断。如果数值稳定但 rep2 指标仍显著弱于 rep0，不启动完整 200 epoch
重训；本修复的主要价值是修复共享几何路径的数值缺陷，而不是证明 rep2 是最佳表示。

## 8. 完成标准

本工作只有在以下条件全部满足时完成：

1. `_stable_atan2` 的 forward 与原生 atan2 等价；
2. 所有规定退化输入的一阶 backward 有限；
3. 既有 OBB geometry、ADR、criterion 与 diagnostic runner 测试通过；
4. 保存的 step 10 失败 batch replay 通过；
5. 原 checkpoint 的 100-step 和完整 epoch 验证通过；
6. 未修改 rep2 offset 合约、loss 权重、decoder 结构或其他 angle representation；
7. 新异常若出现，作为独立根因重新诊断，不以额外防御代码掩盖。
