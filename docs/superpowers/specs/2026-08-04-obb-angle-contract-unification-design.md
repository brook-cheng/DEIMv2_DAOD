# DEIMv2 OBB 角度量纲统一设计

日期：2026-08-04

状态：设计已确认，等待用户审核规格文档

## 1. 目标

修正 DEIMv2 OBB 全链路中角度语义和量纲混用的问题，并建立可长期维护的统一契约。

本设计遵循以下原则：

1. 跨模块传递的标准 OBB 只使用一种物理角度语义。
2. 网络优化编码与物理角度显式分离。
3. `angle_rep` 只影响 decoder 内部的 head 与 residual 表示，不泄漏到 criterion、matcher、postprocessor 或 eval。
4. 所有角度转换集中在统一 helper 和 Codec 中，禁止调用方通过裸 `* pi`、`/ pi` 或 `+/- 0.25` 推断语义。
5. 四种 `angle_rep` 都保留，但最终必须汇入相同的标准 5D OBB 边界。

## 2. 非目标

本设计不改变：

- OBB 的 pi 周期几何定义；
- 模型的类别预测结构；
- HBB 路径；
- postprocessor 的空间尺度变换；
- eval 使用的几何度量；
- 四种 `angle_rep` 的特征提取或 head 架构差异。

本设计也不要求为 tensor 引入 dataclass 或自定义 Tensor 子类，以避免影响训练、导出、TorchScript 和 ONNX 链路。

## 3. 核心决策：双域角度契约

### 3.1 公开物理域

跨模块标准 OBB 定义为：

```text
(cx, cy, w, h, theta_phys_rad)
```

其中：

```text
theta_phys_rad in [0, pi)
```

该契约用于：

- dataset 输出；
- transforms 输入与输出；
- decoder 的公开输出；
- criterion 和 matcher 输入；
- postprocessor 输入与输出；
- eval 与 export 输出；
- 所有 polygon、external rectangle 和 OBB geometry API。

选择 `[0, pi)` 的原因是：它是修改前已稳定使用的几何交换格式，保留它可以降低 geometry、postprocessor、eval 和导出链路的改动范围。对正确处理 pi 周期的几何运算而言，它与其他宽度为 pi 的 canonical 区间等价。

### 3.2 网络优化域

decoder reference、sigmoid 输出与 denoising 使用无量纲归一化角：

```text
theta_norm in [0, 1)
```

该编码对应优化 canonical 区间：

```text
theta_opt_rad in [-pi/4, 3*pi/4)
```

它只用于网络优化，不作为公开 OBB 角度。

转换公式：

```text
theta_opt_rad = remainder(theta_phys_rad + pi/4, pi) - pi/4
theta_norm = (theta_opt_rad + pi/4) / pi

theta_opt_rad = (theta_norm - 0.25) * pi
theta_phys_rad = remainder(theta_opt_rad, pi)
```

这将 `[0, pi)` 物理角映射到以下网络编码：

| 物理方向 | `theta_norm` |
|---|---:|
| 0 度 | 0.25 |
| 45 度 | 0.5 |
| 90 度 | 0.75 |
| 135 度 | 0 或 1 的周期边界 |

主要优化收益是将高频水平和垂直方向移出编码边界，减少普通标量回归、denoising 和迭代 refinement 在 0 度附近的周期跳变。

### 3.3 网络角度 logit

```text
theta_logit in (-inf, +inf)
```

它仅允许出现在：

- `inverse_sigmoid(theta_norm)` 的结果；
- denoising query；
- 未激活 reference。

`theta_logit` 不得进入 polygon、external rectangle、ProbIoU 或其他几何 API。

### 3.4 周期角残差

角度 residual 统一使用弧度：

```text
delta_theta_rad in [-pi/2, pi/2)
```

定义为从 reference 到 target 的最短有符号 pi 周期旋转：

```text
periodic_delta_rad(target, ref)
  = remainder(target - ref + pi/2, pi) - pi/2
```

绝对物理角和角度 residual 使用相同单位（弧度），但不能使用相同区间：绝对角是 pi 周期方向的 canonical 代表值；residual 是围绕 0 对称的最短旋转。

FGL target、5D residual decode 和角度相关 loss 必须直接使用 `delta_theta_rad`。禁止额外乘 pi。

## 4. 统一转换 API

角度转换只允许通过统一 helper 完成。实现必须提供以下纯函数：

```text
canonicalize_phys_rad(theta_rad) -> theta_phys_rad [0, pi)
physical_rad_to_norm(theta_phys_rad) -> theta_norm [0, 1)
norm_to_physical_rad(theta_norm) -> theta_phys_rad [0, pi)
physical_rad_to_logit(theta_phys_rad) -> theta_logit
logit_to_physical_rad(theta_logit) -> theta_phys_rad [0, pi)
periodic_delta_rad(target_phys_rad, ref_phys_rad) -> [-pi/2, pi/2)
apply_delta_rad(ref_phys_rad, delta_theta_rad) -> theta_phys_rad [0, pi)
```

要求：

- helper 是 angle conversion 的唯一实现位置；
- 调用方不得重新实现相同公式；
- helper 支持任意 batch shape；
- helper 保持 dtype 与 device；
- helper 不修改输入 tensor；
- 边界遵守半开区间，不通过额外 epsilon 改变语义。

## 5. 模块公开契约

### 5.1 Dataset 与 transforms

- polygon 到 OBB 的 canonical 输出为 `[0, pi)` 物理角。
- 每个 transform 接收和返回相同的标准 OBB 契约。
- flip、resize、crop、mosaic 和 affine 不得临时改变公开角度范围。
- polygon refit 返回后必须调用 `canonicalize_phys_rad`。

### 5.2 Denoising

输入为标准物理 OBB。

编码链路：

```text
theta_phys_rad
  -> physical_rad_to_norm
  -> inverse_sigmoid
  -> theta_logit
```

进入 decoder 后：

```text
theta_logit
  -> sigmoid
  -> theta_norm
```

任何 geometry 转换必须先通过 `norm_to_physical_rad`。禁止将 logit 或 `theta_norm` 直接传给 geometry。

### 5.3 Decoder

decoder 是唯一允许感知 `angle_rep` 的主要业务模块。

decoder 每层遵循：

```text
standard physical ref OBB
  -> physical_rad_to_norm
  -> attention/head private representation
  -> raw residual
  -> Codec.decode
  -> standard physical OBB
```

每层 decode 后立即得到标准物理 OBB。下一层需要 `theta_norm` 时必须重新显式编码，禁止层间传递语义不明的 5D angle tensor。

decoder 的公开输出包括：

- `pred_boxes`；
- `ref_points`；
- `pre_boxes`；
- auxiliary boxes；
- denoising boxes。

以上所有 5D OBB 的角度都必须是 `theta_phys_rad in [0, pi)`。

### 5.4 Criterion 与 matcher

- 输入只接受标准物理 OBB。
- matcher 的 ProbIoU、KLD 和周期 angle cost 使用物理弧度。
- criterion 的普通角度比较必须调用 `periodic_delta_rad`。
- criterion 不允许根据 `angle_rep` 选择角度换算公式。
- FGL target 由注入 criterion 的 Codec 生成；criterion 只调用 Codec 的公开 target-encoding 接口。
- 模型装配层必须向 decoder 与 criterion 注入同一种 Codec。criterion 不读取 `angle_rep`，也不依据 residual 维度猜测 Codec。

### 5.5 Postprocessor、eval 与 export

- postprocessor 只缩放 `cx`、`cy`、`w`、`h`；角度保持不变。
- eval 与 export 只接收和输出 `[0, pi)` 物理角。
- 这些模块不允许包含 `angle_rep` 分支。

## 6. Codec 设计

四种 `angle_rep` 收敛为两个几何 Codec。

### 6.1 ExternalRectOffsetCodec（6D）

使用者：rep0、rep2。

原始 residual：

```text
(alpha, beta, gamma, delta, epsilon, eta)
```

职责：

- `decode(ref_phys_obb, raw_residual) -> standard_phys_obb`；
- `encode_target(ref_phys_obb, target_phys_obb) -> six_dim_target`；
- 保证 encode 与 decode 使用相同 external-rectangle size scaling；
- 所有 external rectangle geometry 只接收物理角；
- Codec 内不得通过 `theta_norm * pi` 构造物理角。

rep0 与 rep2 可以保留不同的 head、feature fusion 和 angle-first 架构，但使用相同的 6D 几何编码、解码语义。

### 6.2 DirectAngleCodec（5D）

使用者：rep1、rep3。

原始 residual：

```text
(alpha, beta, gamma, delta, delta_theta_rad)
```

职责：

- `decode(ref_phys_obb, raw_residual) -> standard_phys_obb`；
- `encode_target(ref_phys_obb, target_phys_obb) -> five_dim_target`；
- angle target 使用 `periodic_delta_rad`；
- angle decode 使用 `apply_delta_rad`；
- FGL target 与 decoder decode 均使用弧度；
- rep1 和 rep3 都不得对 angle residual 额外乘 pi。

rep1 与 rep3 的差异只保留在 head 和 feature architecture，不得改变 5D Codec 的数学语义。

### 6.3 Codec 选择

构造期固定映射：

| `angle_rep` | Codec |
|---:|---|
| 0 | `ExternalRectOffsetCodec` |
| 1 | `DirectAngleCodec` |
| 2 | `ExternalRectOffsetCodec` |
| 3 | `DirectAngleCodec` |

模型装配层根据该映射构造 Codec，并将同一种 Codec 分别注入 decoder 与 criterion。decoder 调用 `decode`，criterion 调用 `encode_target`。criterion 不读取 `angle_rep`，也不根据 tensor shape 动态选择数学公式。

## 7. 错误处理

### 7.1 生产环境结构检查

始终启用：

- Standard OBB 最后一维必须为 5；
- 6D Codec residual 最后一维必须为 6；
- 5D Codec residual 最后一维必须为 5；
- `angle_rep` 与 Codec 的映射必须在构造期确定；
- 不兼容功能组合在构造期抛出 `ValueError`，不得延迟到 forward。

### 7.2 Debug/test 语义检查

仅 debug 或测试启用：

- `theta_phys_rad` 在容差内属于 `[0, pi)`；
- `theta_norm` 在容差内属于 `[0, 1)`；
- `delta_theta_rad` 在容差内属于 `[-pi/2, pi/2)`；
- geometry API 输入必须满足物理角契约；
- logit 只检查 finite，不检查范围。

### 7.3 Canonical 边界

- 所有公开物理角输出调用 `canonicalize_phys_rad`；
- 所有 residual target 调用 `periodic_delta_rad`；
- 输入恰好位于周期边界时遵守半开区间；
- 不增加用于改变归属区间的 epsilon 分支。

## 8. 验收测试

### 8.1 Helper 数学测试

- `physical_rad_to_norm` 与 `norm_to_physical_rad` 几何互逆；
- `physical_rad_to_logit` 与 `logit_to_physical_rad` 在 clamp 语义内互逆；
- `periodic_delta_rad` 对加减任意整数倍 pi 不变；
- `apply_delta_rad(ref, periodic_delta_rad(target, ref))` 等价于 target；
- dtype、device、shape 和输入不变性测试。

### 8.2 Codec 数学不变量

每个 Codec 必须满足：

1. 零残差不变性：

   ```text
   decode(ref, zero_continuous_residual) == ref
   ```

   `zero_continuous_residual` 指 Integral/Codec 边界接收的连续 residual 全零，不是 head logits 全零。测试同时比较物理参数和顶点集合，不允许固定 `+pi/4` 旋转。

2. encode/decode 互逆：

   ```text
   decode(ref, encode_target(ref, target)) == target
   ```

3. 表示一致性：给定同一组 `ref_phys_obb` 与 `target_phys_obb`，分别执行两个 Codec 的 `encode_target` 与 `decode`，最终输出必须与同一 target 几何等价。测试不要求两种 Codec 的 raw residual 数值相同。

### 8.3 角度边界测试

覆盖：

- `0 +/- epsilon`；
- `pi - epsilon` 与 `0 + epsilon`；
- `pi/2 +/- epsilon`；
- 优化编码切口 `-pi/4 +/- epsilon` 与 `3*pi/4 +/- epsilon`；
- `w` 与 `h` 接近时的 long-edge 交换；
- 非零角 residual 的真实增量，防止额外乘 pi 回归。

### 8.4 组件契约测试

- dataset 与每个 transform 输出 `[0, pi)`；
- denoising 的 `phys -> norm -> logit -> norm -> phys` 闭环；
- attention 对同一物理方向的 pi 周期等价输入产生一致采样几何；
- decoder 每层和每类输出的 `ref_points`、`pred_boxes` 均为标准物理 OBB；
- criterion、matcher、postprocessor 和 eval 无 `angle_rep` 条件分支；
- postprocessor 角度 bitwise 不变；
- export 输出角度为 `[0, pi)`。

### 8.5 防回归源码检查

以下模式不得出现在统一 helper/Codec 之外：

```text
distance[..., 4] *= pi
theta[..., 4] * pi
theta[..., 4] / pi
theta[..., 4] +/- 0.25
oriented_box_to_external_rect(theta_logit_tensor)
geometry_fn(theta_norm_tensor)
```

源码检查用于提示潜在隐式转换，最终判定仍由数学不变量和组件契约测试完成。

## 9. 迁移顺序约束

后续实施计划必须按以下依赖顺序迁移：

1. 新增统一 helper 和测试；
2. 恢复 dataset、transforms 和 geometry 的 `[0, pi)` 物理角契约；
3. 实现两个 Codec 及其数学测试；
4. 改造 decoder 局部表示和逐层边界；
5. 改造 denoising；
6. 改造 FGL target、criterion 和 matcher；
7. 核对 postprocessor、eval 和 export；
8. 执行四 rep 端到端训练前 smoke、backward 和回归测试；
9. 删除旧转换和陈旧 `[−pi/4, 3*pi/4)` 公开契约文档。

实施过程中不得同时保留两套公开 OBB 契约；每个迁移阶段必须通过测试锁定边界后再迁移下一个消费者。

## 10. 成功标准

设计完成实施后，应满足：

- 跨模块 5D OBB 只有 `[0, pi)` 物理角一种语义；
- `theta_norm` 和 `theta_logit` 不离开 decoder/denoising 优化边界；
- residual angle 全部使用弧度，不存在 rep-specific pi 缩放；
- geometry 不接收 norm 或 logit；
- criterion、matcher、postprocessor 和 eval 不感知 `angle_rep`；
- rep0/rep1/rep2/rep3 的零残差、非零残差和 encode/decode 测试全部通过；
- 当前审计确认的 rep0 `+pi/4`、rep1 residual `*pi`、rep2 ref `*pi` 与 denoising logit/geometry 混用问题均由不变量测试覆盖。
