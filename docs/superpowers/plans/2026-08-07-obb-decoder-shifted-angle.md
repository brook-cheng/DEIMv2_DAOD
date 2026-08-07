# DEIMv2 OBB decoder 私有 shifted 角度编码 — 实现计划

> **For agentic workers:** 本计划遵循项目固定执行流程：**AI Test Gate → User Implementation Gate → AI Green/Review Gate**。每个 Review Unit 内，AI 先写 RED 测试并验证预期失败，用户修改指定 production 文件，AI 再运行定向/相关/全量 GREEN 测试并做代码 review。可用 `subagent-driven-development` 或 `executing-plans` 辅助跟踪勾选框进度。

**Goal:** 为 DEIMv2 OBB decoder 引入可配置的私有角度编码 `decoder_angle_encoding`（`proportional` | `shifted`），shifted 编码将 decoder 内部绝对角 reference 的 sigmoid 参数化从边界饱和区移到中心（seam 移至 135°），同时保持公开物理角契约 `[0, π)` 与 `proportional` 模式逐位不变。

**Architecture:** 新增两个纯函数 `physical_rad_to_shifted_norm` / `shifted_norm_to_physical_rad`（`obb_angle_contract.py`）；在 `TransformerDecoder` / `DEIMTransformer` / `TransformerDecoderLayer` / `MSDeformableAttention` / `get_contrastive_denoising_training_group` 逐层传递 `decoder_angle_encoding` / `angle_encoding` 参数；7 个转换点（anchor 生产、denoising 入口、geometry decode 前后、注意力旋转、encoder 辅助输出、公开输出）全部由 `_resolved_angle_encoding() == "shifted"` 守卫。rep2 在所有模式下强制解析为 `proportional`。

**Tech Stack:** PyTorch + DEIMv2（`deimv2_daod/`）、`engine/deim/` 下的 decoder/attention/denoising 模块、`pytest`。

## Global Constraints

- 所有命令在 `deimv2_daod/` 根目录执行。
- **`proportional` 模式与现状逐位一致**：任何未由 shifted 守卫的生产改动都会破坏该保证；所有转换点（spec §7 站点 1/1b/2/3/4/5/6/7）必须成对（生产者/还原者）且同守卫。
- **rep2 硬性规则**：`_resolved_angle_encoding()` 对 `angle_rep == 2` 恒返回 `"proportional"`；rep2 的全部代码路径（anchor 6D、ε/η 流、`_obb_denoising_unact_to_rep2_unact`、`external_xywh_rect_to_oriented_box`）不做任何改动。
- 非法 `decoder_angle_encoding` 值必须在 `TransformerDecoder.__init__` 与 `DEIMTransformer.__init__` 构造阶段抛 `ValueError`（`_VALID_DECODER_ANGLE_ENCODINGS = ("proportional", "shifted")`），不允许静默回退。
- **禁止**修改 `deim_criterion.py`、`deim_postprocessor.py`、`obb_geometry.py`、`dfine_utils.py`、matcher、eval、export（公开契约不变）。
- `dfine_decoder.py` 与 `rtdetrv2_decoder.py` 中 `MSDeformableAttention` 的既有实例（不传 `angle_encoding`）必须因默认值 `"proportional"` 而逐位不受影响。
- `get_contrastive_denoising_training_group` 新增参数必须带默认值 `"proportional"`，使 `dfine_decoder.py:960`、`rtdetrv2_decoder.py:574` 既有调用点不受影响。
- git commit 仅在用户明确指示时执行；每个 Unit 末尾的 commit 为建议边界。
- 不得删除、重命名或改动既有测试的断言（既有 OBB 定向测试 277 项 + 全量套件必须全绿）。
- 本计划全部生产代码由**用户**编写（AI 不直接修改 production 文件）；AI 只写测试并验证。

---

## Review Unit 0: 提交 spec 简化（前置）

**Files:**
- Modify（已改未提交）: `docs/superpowers/specs/2026-08-05-obb-decoder-shifted-angle-design.md`

**Interfaces:**
- Consumes: 工作区已有未提交的 spec 简化改动（删除 `shifted_direct`、scope A/B，双模式配置面，`git diff` 已验证 24+/22-）
- Produces: 干净的 spec 基线，后续 Unit 的验收以该 spec §5/§6/§7/§12 为唯一依据

- [ ] **Step 1: 确认 diff 只含 spec 简化**

```bash
cd /mnt/d/cx/thired/deimv2_daod
git diff --stat docs/superpowers/specs/2026-08-05-obb-decoder-shifted-angle-design.md
git diff docs/superpowers/specs/2026-08-05-obb-decoder-shifted-angle-design.md | grep -E "shifted_direct|scope [AB]" | head
```

Expected: `git diff --stat` 显示 1 个文件；grep 无残留（简化已完成）。

- [ ] **Step 2: 用户提交 spec 简化**

```bash
git add docs/superpowers/specs/2026-08-05-obb-decoder-shifted-angle-design.md
git commit -m "docs: 简化 shifted-angle 设计为双模式（proportional|shifted），移除 shifted_direct 与 scope A/B"
```

Expected: 提交成功，`git status` 干净。

---

## Review Unit 1: `obb_angle_contract.py` — shifted 编解码纯函数

**Files:**
- Modify: `engine/deim/obb_angle_contract.py`（新增两个函数，不动既有函数）
- Test: `test/test_obb_angle_contract.py`

**Interfaces:**
- Consumes: 既有 `physical_rad_to_norm` / `norm_to_physical_rad` / `physical_rad_to_loss_rad`（仅作参照，不改）
- Produces: `physical_rad_to_shifted_norm(theta_phys_rad: Tensor) -> Tensor`（`[0,π)` → `[0,1)`）、`shifted_norm_to_physical_rad(theta_shift: Tensor) -> Tensor`（逆）。后续所有 Unit 的还原点/生产点都 import 这两个函数。

### AI Test Gate（RED）

- [ ] **Step 1: 写 failing test**

在 `test/test_obb_angle_contract.py` 顶部 import 区追加：

```python
from engine.deim.obb_angle_contract import (
    canonicalize_phys_rad,
    physical_rad_to_norm,
    norm_to_physical_rad,
    physical_rad_to_logit,
    logit_to_physical_rad,
    physical_rad_to_loss_rad,
    physical_rad_to_shifted_norm,
    shifted_norm_to_physical_rad,
)
```

在文件末尾追加以下测试（spec §12.1）：

```python
# ---------------------------------------------------------------------------
# 7. shifted 编解码: physical_rad_to_shifted_norm / shifted_norm_to_physical_rad
#    (spec 2026-08-05-obb-decoder-shifted-angle-design.md §5, §12.1)
# ---------------------------------------------------------------------------


def test_physical_rad_to_shifted_norm_mapping():
    """spec §5/§12.1 边界映射: 0→0.25, π/2→0.75, 3π/4→0, π⁻→0.25⁻。"""
    theta = torch.tensor([0.0, PI / 2, 3 * PI / 4, PI - 1e-4])
    shift = physical_rad_to_shifted_norm(theta)
    expected = torch.tensor([0.25, 0.75, 0.0, 0.25 - 1e-4 / PI])
    assert torch.allclose(shift, expected, atol=1e-6), f"got {shift}"
    # 域 [0, 1)
    assert (shift >= 0).all() and (shift < 1).all()


def test_shifted_norm_to_physical_rad_mapping():
    """spec §5 逆映射: 0.25→0, 0.75→π/2, 0→3π/4, 1⁻→(3π/4)⁻。"""
    shift = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0 - 1e-4])
    phys = shifted_norm_to_physical_rad(shift)
    expected = torch.tensor(
        [3 * PI / 4, 0.0, PI / 4, PI / 2, 3 * PI / 4 - 1e-4 * PI]
    )
    assert torch.allclose(phys, expected, atol=1e-6), f"got {phys}"
    # 域 [0, π)
    assert (phys >= 0).all() and (phys < PI).all()


def test_shifted_roundtrip_random():
    """phys→shift→phys 恒等; shift→phys→shift 恒等（remainder 数值稳定）。"""
    torch.manual_seed(42)
    theta = torch.rand(2000) * PI                       # [0, π)
    rt = shifted_norm_to_physical_rad(physical_rad_to_shifted_norm(theta))
    assert torch.allclose(rt, theta, rtol=1e-6), (
        f"phys→shift→phys max err {(rt - theta).abs().max():.2e}"
    )

    shift = torch.rand(2000)                            # [0, 1)
    rt2 = physical_rad_to_shifted_norm(shifted_norm_to_physical_rad(shift))
    assert torch.allclose(rt2, shift, rtol=1e-6), (
        f"shift→phys→shift max err {(rt2 - shift).abs().max():.2e}"
    )


def test_shifted_algebraic_relation_with_loss_rad():
    """spec §5 代数关系: theta_shift = physical_rad_to_loss_rad(θ)/π + 0.25 (mod 1)。"""
    torch.manual_seed(7)
    theta = torch.rand(500) * PI
    loss = physical_rad_to_loss_rad(theta)
    algebraic = torch.remainder(loss / PI + 0.25, 1.0)
    shift = physical_rad_to_shifted_norm(theta)
    assert torch.allclose(shift, algebraic, atol=1e-6), (
        f"max err {(shift - algebraic).abs().max():.2e}"
    )


def test_shifted_pi_periodicity_preserved():
    """shifted 编码保持 π 周期性: θ 与 θ+π 映射到同一 theta_shift。"""
    torch.manual_seed(11)
    theta = torch.rand(500) * PI
    a = physical_rad_to_shifted_norm(theta)
    b = physical_rad_to_shifted_norm(theta + PI)
    assert torch.allclose(a, b, atol=1e-6), "shifted 必须保持 π 周期等价"
```

- [ ] **Step 2: 运行验证预期失败**

```bash
python -m pytest test/test_obb_angle_contract.py -q -k "shifted"
```

Expected: 失败，`ImportError: cannot import name 'physical_rad_to_shifted_norm'`。

### User Implementation Gate

- [ ] **Step 3: 用户实现**

在 `engine/deim/obb_angle_contract.py` 中 `physical_rad_to_loss_rad` 之后追加（保持文件内中文 docstring 风格），并在文件顶部速查表补一行 `theta_shift` 定义：

```python
def physical_rad_to_shifted_norm(theta_phys_rad: Tensor) -> Tensor:
    """将物理弧度角编码为 decoder 私有 shifted 归一化角。

    Args:
        theta_phys_rad: 输入物理角，单位为**弧度**，范围 ``[0, pi)``。

    Returns:
        ``theta_shift``：输出 shifted 归一化角，**无量纲**，范围 ``[0, 1)``，
        即 ``remainder(theta_phys_rad / pi + 0.25, 1)``。0→0.25, π/2→0.75,
        3π/4→0（seam 移至 135°）。
    """
    return torch.remainder(theta_phys_rad / torch.pi + 0.25, 1.0)


def shifted_norm_to_physical_rad(theta_shift: Tensor) -> Tensor:
    """将 decoder 私有 shifted 归一化角还原为物理弧度角。

    这是 :func:`physical_rad_to_shifted_norm` 的逆转换。

    Args:
        theta_shift: 输入 shifted 归一化角，**无量纲**，范围 ``[0, 1)``。

    Returns:
        ``theta_phys_rad``：输出标准物理角，单位为**弧度**，范围
        ``[0, pi)``。
    """
    return torch.remainder(theta_shift - 0.25, 1.0) * torch.pi
```

### AI Green/Review Gate

- [ ] **Step 4: 运行定向 GREEN**

```bash
python -m pytest test/test_obb_angle_contract.py -q -k "shifted"
```

Expected: 5 passed。

- [ ] **Step 5: 运行相关 GREEN（既有契约不得回归）**

```bash
python -m pytest test/test_obb_angle_contract.py -q
```

Expected: 34 passed（原 29 + 新 5）。

- [ ] **Step 6: 代码 review checklist**

- 两个函数仅依赖 `torch.remainder`，无 clamp、无 inplace、无副作用；
- 与 `physical_rad_to_loss_rad` 独立实现，未复用（spec §5 要求避免语义耦合）；
- docstring 数值域与 `obb_angle_contract.py` 顶部速查表风格一致；`theta_shift` 已加入顶部速查表。

- [ ] **Step 7: 提交（用户决定）**

```bash
git add engine/deim/obb_angle_contract.py test/test_obb_angle_contract.py
git commit -m "feat: 新增 shifted 角度编解码纯函数（decoder 私有编码）"
```

---

## Review Unit 2: `dfine_decoder.py` — `MSDeformableAttention.angle_encoding` 参数 + 站点 5

**Files:**
- Modify: `engine/deim/dfine_decoder.py`（`MSDeformableAttention.__init__` + forward 5D 分支）
- Test: `test/test_deimv2_obb_smoke.py`

**Interfaces:**
- Consumes: Unit 1 的 `shifted_norm_to_physical_rad`
- Produces: `MSDeformableAttention(..., angle_encoding: str = "proportional")` 构造参数；`self.angle_encoding` 属性；5D reference 分支按编码解码物理角。默认值保证 `dfine_decoder.py` 自身与既有实例零改动。

### AI Test Gate（RED）

- [ ] **Step 1: 写 failing test**

在 `test/test_deimv2_obb_smoke.py` 顶部 import 区追加：

```python
from engine.deim.obb_angle_contract import (
    norm_to_physical_rad,
    physical_rad_to_norm,
    physical_rad_to_shifted_norm,
    shifted_norm_to_physical_rad,
)
```

在 `test_msdeform_attn_decouple_angle_reference_consumes_theta` 之后追加（spec §12.3）：

```python
# ---------------------------------------------------------------------------
# Test 1b: MSDeformableAttention 站点 5 — shifted 编码的注意力旋转
# ---------------------------------------------------------------------------


def test_msdeform_attn_shifted_equiv_to_proportional_same_phys():
    """spec §12.3: 同一物理角，shifted 与 proportional 编码的 5D reference
    在注意力中必须产生相同输出（站点 5 的 shifted 分支经
    shifted_norm_to_physical_rad 还原后与 ×π 等价的物理角一致）。

    Given: 两个同权重 MSDeformableAttention（proportional / shifted）。
    When:  对同一物理角 theta_phys，分别用 theta_norm 与 theta_shift 编码
            5D reference 前向。
    Then:  输出必须 allclose（shifted 分支不得忽略/错解 θ）。
    """
    torch.manual_seed(0)
    embed_dim, num_heads, num_levels, num_points = 32, 4, 2, 2
    spatial_shapes = [(4, 4), (2, 2)]
    bs, n_queries, n_ref_levels = 1, 5, 1

    attn_prop = MSDeformableAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_levels=num_levels,
        num_points=num_points,
        method="default",
        angle_encoding="proportional",
    )
    attn_shift = MSDeformableAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_levels=num_levels,
        num_points=num_points,
        method="default",
        angle_encoding="shifted",
    )
    # 同权重: MSDeformableAttention 初始化完全确定（constant/arange），
    # 但显式复制 state_dict 以消除任何不确定来源。
    attn_shift.load_state_dict(attn_prop.state_dict())
    attn_prop.eval()
    attn_shift.eval()

    query = torch.randn(bs, n_queries, embed_dim)
    value = _make_msdeform_value(bs, num_heads, embed_dim // num_heads, spatial_shapes)

    centers = 0.3 + 0.4 * torch.rand(bs, n_queries, n_ref_levels, 2)
    wh = 0.1 + 0.2 * torch.rand(bs, n_queries, n_ref_levels, 2)
    theta_phys = torch.rand(bs, n_queries, n_ref_levels, 1) * math.pi

    ref_prop = torch.cat([centers, wh, physical_rad_to_norm(theta_phys)], dim=-1)
    ref_shift = torch.cat([centers, wh, physical_rad_to_shifted_norm(theta_phys)], dim=-1)

    out_prop = attn_prop(query, ref_prop, value, spatial_shapes)
    out_shift = attn_shift(query, ref_shift, value, spatial_shapes)
    assert torch.isfinite(out_shift).all(), "shifted 5D ref 输出含 NaN"
    assert torch.allclose(out_prop, out_shift, atol=1e-5), (
        "同一物理角的 shifted 与 proportional 注意力输出必须一致；"
        "站点 5 未正确还原 shifted 编码"
    )


def test_msdeform_attn_shifted_90deg_rotation_axis():
    """spec §12.3: 90° reference（theta_shift=0.75）在 shifted 模式下改变
    θ 会改变输出（θ 被消费，非忽略），且输出有限。
    """
    torch.manual_seed(0)
    embed_dim, num_heads, num_levels, num_points = 32, 4, 2, 2
    spatial_shapes = [(4, 4), (2, 2)]
    bs, n_queries, n_ref_levels = 1, 3, 1

    attn_shift = MSDeformableAttention(
        embed_dim=embed_dim, num_heads=num_heads, num_levels=num_levels,
        num_points=num_points, method="default", angle_encoding="shifted",
    )
    attn_shift.eval()

    query = torch.randn(bs, n_queries, embed_dim)
    value = _make_msdeform_value(bs, num_heads, embed_dim // num_heads, spatial_shapes)
    centers = torch.full((bs, n_queries, n_ref_levels, 2), 0.5)
    wh = torch.full((bs, n_queries, n_ref_levels, 2), 0.2)

    ref_90 = torch.cat([centers, wh, torch.full((bs, n_queries, n_ref_levels, 1), 0.75)], dim=-1)
    out_90 = attn_shift(query, ref_90, value, spatial_shapes)
    assert torch.isfinite(out_90).all()

    # θ 被消费: 0.75(90°) 与 0.25(0°) 输出不同
    ref_0 = torch.cat([centers, wh, torch.full((bs, n_queries, n_ref_levels, 1), 0.25)], dim=-1)
    out_0 = attn_shift(query, ref_0, value, spatial_shapes)
    assert not torch.allclose(out_90, out_0, atol=1e-6), (
        "shifted 模式 θ 通道被忽略（90° 与 0° 输出相同）"
    )
```

- [ ] **Step 2: 运行验证预期失败**

```bash
python -m pytest test/test_deimv2_obb_smoke.py -q -k "shifted"
```

Expected: 失败，`TypeError: __init__() got an unexpected keyword argument 'angle_encoding'`。

### User Implementation Gate

- [ ] **Step 3: 用户实现**

`engine/deim/dfine_decoder.py`：

1. 顶部 import 区追加（在 `from .obb_geometry import external_xywh_rect_to_oriented_box` 之后）：

```python
from .obb_angle_contract import shifted_norm_to_physical_rad
```

2. `MSDeformableAttention.__init__` 签名（第 48-56 行）追加参数并存储：

```python
    def __init__(
        self,
        embed_dim=256,
        num_heads=8,
        num_levels=4,
        num_points=4,
        method="default",
        offset_scale=0.5,
        angle_encoding="proportional",
    ):
```

在 `self.method = method` 之后追加：

```python
        self.angle_encoding = angle_encoding
```

3. forward 5D 分支（第 177-178 行）改为按编码解码：

```python
            else:
                # 站点 5: 5D OBB 分支的 θ 通道按编码解码为物理角。
                # proportional: theta_norm * π; shifted: shifted_norm_to_physical_rad。
                if self.angle_encoding == "shifted":
                    angle = shifted_norm_to_physical_rad(reference_points[..., 4:5])
                else:
                    angle = reference_points[..., 4:5] * torch.pi
```

注意：6D 分支（rep2 路径，`external_xywh_rect_to_oriented_box` 后 θ 已是物理角）**不修改**；rep2 恒 proportional，其 6D 路径不经过此 5D 分支。

### AI Green/Review Gate

- [ ] **Step 4: 运行定向 GREEN**

```bash
python -m pytest test/test_deimv2_obb_smoke.py -q -k "shifted"
```

Expected: 2 passed。

- [ ] **Step 5: 运行相关 GREEN（既有注意力测试不回归）**

```bash
python -m pytest test/test_deimv2_obb_smoke.py -q -k "msdeform_attn or shifted"
```

Expected: 3 passed（原 1 + 新 2）。

- [ ] **Step 6: 代码 review checklist**

- `angle_encoding` 默认 `"proportional"`，`dfine_decoder.py:246` 与 `rtdetrv2_decoder.py:186` 既有实例不传该参数 → 走默认分支，逐位不变；
- 5D 分支的 shifted 还原与 proportional `* π` 数学等价（`shifted_norm_to_physical_rad(physical_rad_to_shifted_norm(θ)) == θ`）；
- 6D 分支零改动；`self.angle_encoding` 未用于 6D 分支（rep2 排除，符合设计）。

- [ ] **Step 7: 提交（用户决定）**

```bash
git add engine/deim/dfine_decoder.py test/test_deimv2_obb_smoke.py
git commit -m "feat: MSDeformableAttention 支持 angle_encoding，站点 5 按编码解码物理角"
```

---

## Review Unit 3: `deim_decoder.py` — 配置传播与校验

**Files:**
- Modify: `engine/deim/deim_decoder.py`（`TransformerDecoder.__init__`、`TransformerDecoderLayer.__init__`、`DEIMTransformer.__init__`、模块级常量与辅助）
- Test: `test/test_deimv2_obb_smoke.py`

**Interfaces:**
- Consumes: Unit 1 函数（本 Unit 不直接使用，仅后续 Unit 用）
- Produces:
  - 模块级 `_VALID_DECODER_ANGLE_ENCODINGS = ("proportional", "shifted")` 与 `_resolve_angle_encoding(angle_rep, decoder_angle_encoding)` 辅助
  - `TransformerDecoder(..., decoder_angle_encoding: str = "proportional")`：构造期校验 + `self.decoder_angle_encoding` + `_resolved_angle_encoding()` 方法
  - `TransformerDecoderLayer(..., angle_encoding: str = "proportional")`：透传给 `MSDeformableAttention`
  - `DEIMTransformer(..., decoder_angle_encoding: str = "proportional")`：构造期校验 + `self.decoder_angle_encoding` + `_resolved_angle_encoding()`；构建两个 layer 时传解析后的 `angle_encoding`，构建 `TransformerDecoder` 时传原始配置值

### AI Test Gate（RED）

- [ ] **Step 1: 写 failing test**

在 `test/test_deimv2_obb_smoke.py` 的 `_make_rep2_model_with_denoising` 附近追加通用模型构造 helper（供后续 Unit 复用）：

```python
def _make_obb_model(
    angle_rep,
    use_angle_first=False,
    decoder_angle_encoding="proportional",
    num_denoising=0,
    angle_step=0.0,
):
    """最小 DEIMTransformer（obb），供 shifted/proportional 矩阵测试复用。

    零初始化头（_reset_parameters）保证确定性；dropout=0、BN=Identity
    （feat_channels == hidden_dim），forward 在 no_grad 下逐位可复现。
    """
    torch.manual_seed(0)
    return DEIMTransformer(
        num_classes=5,
        hidden_dim=32,
        num_queries=4,
        feat_channels=[32, 32],
        feat_strides=[4, 8],
        num_levels=2,
        num_points=2,
        nhead=4,
        num_layers=3,
        dim_feedforward=64,
        dropout=0.0,
        activation="relu",
        num_denoising=num_denoising,
        learn_query_content=False,
        eval_spatial_size=(16, 16),
        eval_idx=-1,
        eps=1e-2,
        aux_loss=True,
        cross_attn_method="default",
        query_select_method="default",
        reg_max=4,
        reg_scale=4.0,
        layer_scale=1,
        mlp_act="relu",
        use_gateway=True,
        share_bbox_head=False,
        share_score_head=False,
        box_mode="obb",
        angle_rep=angle_rep,
        offset_scale_source="pre",
        angle_step=angle_step,
        use_angle_first=use_angle_first,
        decoder_angle_encoding=decoder_angle_encoding,
    )
```

追加以下测试（spec §6）：

```python
# ---------------------------------------------------------------------------
# Test 3b: decoder_angle_encoding 配置传播与校验（spec §6）
# ---------------------------------------------------------------------------


def test_decoder_angle_encoding_invalid_raises():
    """非法 decoder_angle_encoding 必须在构造期抛 ValueError，不静默回退。"""
    with pytest.raises(ValueError, match="decoder_angle_encoding"):
        _make_obb_model(angle_rep=0, decoder_angle_encoding="bogus")
    with pytest.raises(ValueError, match="decoder_angle_encoding"):
        DEIMTransformer(
            num_classes=5, hidden_dim=32, num_queries=4,
            feat_channels=[32, 32], feat_strides=[4, 8], num_levels=2,
            num_points=2, nhead=4, num_layers=3, dim_feedforward=64,
            dropout=0.0, activation="relu", num_denoising=0,
            learn_query_content=False, eval_spatial_size=(16, 16),
            eval_idx=-1, eps=1e-2, aux_loss=False,
            cross_attn_method="default", query_select_method="default",
            reg_max=4, reg_scale=4.0, layer_scale=1, mlp_act="relu",
            use_gateway=True, share_bbox_head=False, share_score_head=False,
            box_mode="obb", angle_rep=2, offset_scale_source="pre",
            use_angle_first=False, decoder_angle_encoding="bogus",
        )


def test_decoder_angle_encoding_default_proportional():
    """默认值 proportional; 传递链: DEIMTransformer → TransformerDecoder →
    TransformerDecoderLayer → MSDeformableAttention 全部为 proportional。"""
    model = _make_obb_model(angle_rep=0)
    assert model.decoder_angle_encoding == "proportional"
    assert model.decoder.decoder_angle_encoding == "proportional"
    assert model.decoder.layers[0].cross_attn.angle_encoding == "proportional"
    assert model.decoder.layers[-1].cross_attn.angle_encoding == "proportional"


def test_resolved_angle_encoding_rep0_shifted():
    """rep0 + shifted: _resolved_angle_encoding() == 'shifted'，传递到注意力。"""
    model = _make_obb_model(angle_rep=0, decoder_angle_encoding="shifted")
    assert model._resolved_angle_encoding() == "shifted"
    assert model.decoder._resolved_angle_encoding() == "shifted"
    assert model.decoder.layers[0].cross_attn.angle_encoding == "shifted"


def test_resolved_angle_encoding_rep2_forced_proportional():
    """rep2 硬性规则: 配置 shifted 也强制解析为 proportional（含注意力）。"""
    model = _make_obb_model(angle_rep=2, decoder_angle_encoding="shifted")
    assert model._resolved_angle_encoding() == "proportional"
    assert model.decoder._resolved_angle_encoding() == "proportional"
    assert model.decoder.layers[0].cross_attn.angle_encoding == "proportional"
```

- [ ] **Step 2: 运行验证预期失败**

```bash
python -m pytest test/test_deimv2_obb_smoke.py -q -k "decoder_angle_encoding or resolved_angle"
```

Expected: 失败，`TypeError: __init__() got an unexpected keyword argument 'decoder_angle_encoding'`。

### User Implementation Gate

- [ ] **Step 3: 用户实现**

`engine/deim/deim_decoder.py`：

1. 顶部 import 区（第 36 行）改为：

```python
from .obb_angle_contract import (
    norm_to_physical_rad,
    physical_rad_to_norm,
    shifted_norm_to_physical_rad,
    physical_rad_to_shifted_norm,
)
```

2. 模块级常量与解析辅助（放在 `TransformerDecoderLayer` 类之前）：

```python
_VALID_DECODER_ANGLE_ENCODINGS = ("proportional", "shifted")


def _resolve_angle_encoding(angle_rep, decoder_angle_encoding):
    """rep2 硬性规则：其私有 reference 不含显式 θ，恒为 proportional。

    rep0/1/3 返回配置值。非法配置值在构造期已被拒绝，此处不再校验。
    """
    if angle_rep == 2:
        return "proportional"
    return decoder_angle_encoding
```

3. `TransformerDecoderLayer.__init__`（第 59-72 行）追加参数：

```python
        cross_attn_method="default",
        layer_scale=None,
        use_gateway=False,
        angle_encoding="proportional",
```

并在 `self.cross_attn = MSDeformableAttention(...)`（第 88-90 行）透传：

```python
        self.cross_attn = MSDeformableAttention(
            d_model,
            n_head,
            n_levels,
            n_points,
            method=cross_attn_method,
            angle_encoding=angle_encoding,
        )
```

4. `TransformerDecoder.__init__`（第 155-174 行）追加参数 `decoder_angle_encoding="proportional"`（放在 `use_angle_first=False` 之后），在 `self.use_angle_first = use_angle_first` 之后追加校验与存储：

```python
        if decoder_angle_encoding not in _VALID_DECODER_ANGLE_ENCODINGS:
            raise ValueError(
                "decoder_angle_encoding must be 'proportional' or 'shifted', "
                f"got {decoder_angle_encoding!r}"
            )
        self.decoder_angle_encoding = decoder_angle_encoding
```

并在该类中新增方法：

```python
    def _resolved_angle_encoding(self) -> str:
        """rep2 恒 proportional；其余返回配置值。"""
        return _resolve_angle_encoding(self.angle_rep, self.decoder_angle_encoding)
```

5. `DEIMTransformer.__init__`（第 529-566 行）追加参数 `decoder_angle_encoding="proportional"`（放在 `use_angle_first=False` 之后），在 `self.use_angle_first = use_angle_first` 之后追加校验与存储（同上 `if ... raise ValueError` + `self.decoder_angle_encoding = decoder_angle_encoding`）。

6. 两个 `TransformerDecoderLayer` 构造（第 636-646、648-659 行）追加 `angle_encoding=self._resolved_angle_encoding()`：

```python
        decoder_layer = TransformerDecoderLayer(
            hidden_dim,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            num_levels,
            num_points,
            cross_attn_method=cross_attn_method,
            use_gateway=use_gateway,
            angle_encoding=self._resolved_angle_encoding(),
        )

        decoder_layer_wide = TransformerDecoderLayer(
            hidden_dim,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            num_levels,
            num_points,
            cross_attn_method=cross_attn_method,
            layer_scale=layer_scale,
            use_gateway=use_gateway,
            angle_encoding=self._resolved_angle_encoding(),
        )
```

7. `TransformerDecoder` 构造（第 661-679 行）追加 `decoder_angle_encoding=self.decoder_angle_encoding`：

```python
        self.decoder = TransformerDecoder(
            hidden_dim,
            decoder_layer,
            decoder_layer_wide,
            num_layers,
            nhead,
            self.num_reg_dist,
            reg_max,
            self.reg_scale,
            self.up,
            eval_idx,
            layer_scale,
            act=activation,
            box_mode=self.box_mode,
            angle_rep=self.angle_rep,
            offset_scale_source=self.offset_scale_source,
            use_gate_fusion=self.use_gate_fusion,
            use_angle_first=self.use_angle_first,
            decoder_angle_encoding=self.decoder_angle_encoding,
        )
```

8. `DEIMTransformer` 类中新增方法（放在 `_generate_anchors` 之前）：

```python
    def _resolved_angle_encoding(self) -> str:
        """rep2 恒 proportional；其余返回配置值。"""
        return _resolve_angle_encoding(self.angle_rep, self.decoder_angle_encoding)
```

### AI Green/Review Gate

- [ ] **Step 4: 运行定向 GREEN**

```bash
python -m pytest test/test_deimv2_obb_smoke.py -q -k "decoder_angle_encoding or resolved_angle"
```

Expected: 4 passed。

- [ ] **Step 5: 运行相关 GREEN（既有模型构造/前向不回归）**

```bash
python -m pytest test/test_deimv2_obb_smoke.py test/test_deimv2_obb_rep2_eval.py -q
```

Expected: 全绿（smoke 原 12 + 新 4；rep2_eval 1）。

- [ ] **Step 6: 代码 review checklist**

- 两处校验（`TransformerDecoder` / `DEIMTransformer`）消息一致、含 `decoder_angle_encoding` 字样；
- `TransformerDecoderLayer` 仅透传，无逻辑；`angle_encoding` 默认 `"proportional"`；
- 解析辅助 `_resolve_angle_encoding` 只对 `angle_rep == 2` 特判，与 spec §6 一致；
- 构造链顺序：`DEIMTransformer` 先建 layer（用解析值）再建 decoder（用原始值），rep2 两处都得到 proportional；
- 未改动 `rtdetrv2_decoder.py` 的独立 `MSDeformableAttention` 类。

- [ ] **Step 7: 提交（用户决定）**

```bash
git add engine/deim/deim_decoder.py test/test_deimv2_obb_smoke.py
git commit -m "feat: decoder_angle_encoding 配置传播与构造期校验（rep2 强制 proportional）"
```

---

## Review Unit 4: `deim_decoder.py` — 站点 1/1b（anchor 生产）+ 站点 6（encoder 辅助输出）

**Files:**
- Modify: `engine/deim/deim_decoder.py`（`_generate_anchors` 第 1011-1045 行；`_get_decoder_input` 第 1137-1150 行）
- Test: `test/test_deimv2_obb_smoke.py`

**Interfaces:**
- Consumes: Unit 3 的 `_resolved_angle_encoding()`（`DEIMTransformer`）、Unit 1 的 `shifted_norm_to_physical_rad`
- Produces: shifted 模式 anchor θ 默认 `0.5`（45° 于 sigmoid 中心）、`angle_step` 候选 `remainder(r+0.25, 1)`、encoder 辅助输出物理域 `[0, π)`。`proportional` 逐位不变。

### AI Test Gate（RED）

- [ ] **Step 1: 写 failing test**

在 `test/test_deimv2_obb_smoke.py` 追加（spec §7 站点 1/1b、§12.2）：

```python
# ---------------------------------------------------------------------------
# Test 4a: 站点 1/1b — anchor 角度生产（spec §7）
# ---------------------------------------------------------------------------


def test_anchor_default_r_shifted_is_half():
    """spec 站点 1: shifted 默认 anchor θ=0.5（45° 于 sigmoid 中心）；
    proportional 保持 0.25。二者物理角一致（0.25π）。"""
    model_shift = _make_obb_model(angle_rep=0, decoder_angle_encoding="shifted")
    model_prop = _make_obb_model(angle_rep=0, decoder_angle_encoding="proportional")
    for model, expected in [
        (model_shift, 0.5),
        (model_prop, 0.25),
    ]:
        anchors_unact, _ = model._generate_anchors([[4, 4], [2, 2]], device="cpu")
        anchors = torch.sigmoid(anchors_unact)
        assert torch.allclose(
            anchors[..., 4], torch.full_like(anchors[..., 4], expected), atol=1e-6
        ), f"anchor θ 应为 {expected}, got {anchors[0, 0, 4].item():.6f}"


def test_anchor_angle_step_shifted_candidates():
    """spec 站点 1b: shifted 模式 angle_step 候选 = remainder(r + 0.25, 1)。

    angle_step=0.2 → 候选 arange(5)*0.2={0,0.2,0.4,0.6,0.8}，shifted 后
    {0.25,0.45,0.65,0.85,0.05}（全部 > eps 有效，与 proportional 的
    {0.2,0.4,0.6,0.8} 不同集）—— 该用例可判别 shifted 与 proportional。
    """
    model = _make_obb_model(
        angle_rep=0, decoder_angle_encoding="shifted", angle_step=0.2
    )
    anchors_unact, valid_mask = model._generate_anchors([[4, 4]], device="cpu")
    anchors = torch.sigmoid(anchors_unact)
    valid = valid_mask.squeeze(-1)
    theta_vals = anchors[valid, 4].round(decimals=4).unique().tolist()
    expected = {0.05, 0.25, 0.45, 0.65, 0.85}
    assert set(theta_vals) == expected, f"got {set(theta_vals)}, want {expected}"


def test_anchor_angle_step_proportional_unchanged():
    """spec 站点 1b: proportional 模式 angle_step 候选 = arange(n)*step（逐位不变）。

    angle_step=0.2 → {0, 0.2, 0.4, 0.6, 0.8}；θ=0.0 被 valid_mask(eps) 剔除。
    """
    model = _make_obb_model(
        angle_rep=0, decoder_angle_encoding="proportional", angle_step=0.2
    )
    anchors_unact, valid_mask = model._generate_anchors([[4, 4]], device="cpu")
    anchors = torch.sigmoid(anchors_unact)
    valid = valid_mask.squeeze(-1)
    theta_vals = anchors[valid, 4].round(decimals=4).unique().tolist()
    assert set(theta_vals) == {0.2, 0.4, 0.6, 0.8}, f"got {set(theta_vals)}"


def test_encoder_aux_theta_known_answer_shifted():
    """spec 站点 6 判别性 known-answer：零初始化下 encoder 辅助 θ 必为 π/4。

    zero-init enc_bbox_head → enc_topk_bbox_unact == anchors 精确；shifted
    anchor θ_shift=0.5 ↔ prop θ_norm=0.25，二者物理角均为 π/4。若站点 6
    未还原 shifted（仍 norm_to_physical_rad），shifted 会得 0.5π ≠ π/4。

    Given: rep0 + zero-init 头，两个模型（proportional / shifted）。
    When:  _get_decoder_input（train 模式）。
    Then:  enc_topk_bboxes_list[0] θ ≈ π/4，两模式一致。
    """
    torch.manual_seed(0)
    feats = [torch.randn(1, 32, 8, 8), torch.randn(1, 32, 4, 4)]
    model_prop = _make_obb_model(angle_rep=0, decoder_angle_encoding="proportional")
    model_shift = _make_obb_model(angle_rep=0, decoder_angle_encoding="shifted")
    model_prop.train()
    model_shift.train()

    out = {}
    for enc, model in [("proportional", model_prop), ("shifted", model_shift)]:
        memory, spatial_shapes = model._get_encoder_input(feats)
        with torch.no_grad():
            _, _, enc_list, _ = model._get_decoder_input(memory, spatial_shapes)
        theta = enc_list[0][..., 4]
        assert torch.isfinite(theta).all(), f"{enc} encoder 辅助 θ 含 NaN"
        out[enc] = theta

    assert torch.allclose(
        out["shifted"], torch.full_like(out["shifted"], math.pi / 4), atol=1e-4
    ), f"shifted encoder 辅助 θ 应 ≈ π/4, got mean={out['shifted'].mean():.4f}"
    assert torch.allclose(out["proportional"], out["shifted"], atol=1e-4), (
        "两模式 encoder 辅助 θ 应一致（同为 π/4）"
    )
```

- [ ] **Step 2: 运行验证预期失败**

```bash
python -m pytest test/test_deimv2_obb_smoke.py -q -k "anchor_default_r_shifted or anchor_angle_step_shifted or encoder_aux_theta"
```

Expected: 失败 — shifted anchor θ 仍为 0.25（断言 0.5 失败）；`test_encoder_aux_theta_known_answer_shifted` 得 0.5π ≠ π/4（站点 6 未还原）。

### User Implementation Gate

- [ ] **Step 3: 用户实现**

`engine/deim/deim_decoder.py`：

1. `_generate_anchors` 的 `angle_rep != 2` 分支（第 1011-1045 行）：

`angle_step > 0` 分支（第 1022-1034 行）中，在 `angle_candidates = torch.arange(n_angles, dtype=dtype) * self.angle_step` 之后追加：

```python
                        if self._resolved_angle_encoding() == "shifted":
                            # 站点 1b: shifted 候选 = remainder(r + 0.25, 1)
                            angle_candidates = torch.remainder(
                                angle_candidates + 0.25, 1.0
                            )
```

`else` 分支（第 1035-1044 行）改为：

```python
                    else:
                        # 站点 1: 默认 anchor θ。proportional: 0.25（45°）;
                        # shifted: 0.5（45°，sigmoid 中心）。
                        default_r = (
                            0.5
                            if self._resolved_angle_encoding() == "shifted"
                            else 0.25
                        )
                        r = default_r * torch.ones(
                            *grid_xy.shape[:-1],
                            1,
                            dtype=grid_xy.dtype,
                            device=grid_xy.device,
                        )
                        lvl_anchors = torch.concat([grid_xy, wh, r], dim=-1).reshape(
                            -1, h * w, self._num_box_dof
                        )
```

2. `_get_decoder_input` 的 encoder 辅助输出（第 1137-1150 行）改为：

```python
            if self.box_mode == "obb":
                if self.angle_rep != 2:
                    if self._resolved_angle_encoding() == "shifted":
                        # 站点 6: 内部 θ_shift 还原为物理角 [0, π)
                        enc_topk_bboxes = torch.cat(
                            [
                                enc_topk_bboxes[..., :4],
                                shifted_norm_to_physical_rad(
                                    enc_topk_bboxes[..., 4:]
                                ),
                            ],
                            dim=-1,
                        )
                    else:
                        # 角度量纲 [0,1]->[0, pi)
                        enc_topk_bboxes = torch.cat(
                            [
                                enc_topk_bboxes[..., :4],
                                norm_to_physical_rad(enc_topk_bboxes[..., 4:]),
                            ],
                            dim=-1,
                        )
                else:
                    # 使用偏移量替代角度表示
                    enc_topk_bboxes = enc_topk_bboxes
```

### AI Green/Review Gate

- [ ] **Step 4: 运行定向 GREEN**

```bash
python -m pytest test/test_deimv2_obb_smoke.py -q -k "anchor or encoder_aux_theta"
```

Expected: 4 passed（含既有 `test_anchor_default_r_is_pi_over_4`、`test_rep2_generated_anchors_are_valid_external_rect_offsets`）。
- [ ] **Step 5: 运行相关 GREEN**

```bash
python -m pytest test/test_deimv2_obb_smoke.py -q
```

Expected: 全绿。

- [ ] **Step 6: 代码 review checklist**

- 站点 1/1b 均在 `angle_rep != 2` 分支内（rep2 6D anchor 不受影响）；
- `default_r` 仅两值（0.25 / 0.5），shifted 0.5 与 proportional 0.25 物理角一致（0.25π）；
- 站点 6 的 shifted 守卫与既有 `angle_rep != 2` 条件嵌套正确；rep2 直通分支零改动；
- `valid_mask` 判定在 shifted 下 θ=0.0 候选仍会被 `eps` 剔除（`torch.inf` 掩码），与 proportional 语义一致。

- [ ] **Step 7: 提交（用户决定）**

```bash
git add engine/deim/deim_decoder.py test/test_deimv2_obb_smoke.py
git commit -m "feat: anchor 与 encoder 辅助输出支持 shifted 编码（站点 1/1b/6）"
```

---

## Review Unit 5: `denoising.py` — 站点 2（GT θ 入口编码）

**Files:**
- Modify: `engine/deim/denoising.py`（`get_contrastive_denoising_training_group` 第 12-21 行签名 + 第 112-115 行）；`engine/deim/deim_decoder.py`（`DEIMTransformer.forward` 第 1220-1232 行调用点）
- Test: `test/test_deimv2_obb_smoke.py`

**Interfaces:**
- Consumes: Unit 1 的 `physical_rad_to_shifted_norm`；Unit 3 的 `_resolved_angle_encoding()`
- Produces: `get_contrastive_denoising_training_group(..., box_mode="hbb", angle_encoding="proportional")`；GT θ 在 shifted 模式下编码为 `theta_shift`。默认值保证 `dfine_decoder.py:960` 与 `rtdetrv2_decoder.py:574` 既有调用点逐位不变。

### AI Test Gate（RED）

- [ ] **Step 1: 写 failing test**

在 `test/test_deimv2_obb_smoke.py` 的 `_run_denoising` helper 附近追加 shifted 版本并追加测试（spec §7 站点 2）：

```python
def _run_denoising_shifted(gt_theta_rad, num_classes=5, hidden_dim=8, angle_encoding="shifted"):
    """调用 denoising 函数（shifted 编码），返回 sigmoid 还原后的 θ_shift。"""
    target = {
        "labels": torch.tensor([0], dtype=torch.int64),
        "boxes": torch.tensor(
            [[0.5, 0.5, 0.3, 0.2, gt_theta_rad]], dtype=torch.float32
        ),
    }
    class_embed = nn.Embedding(num_classes + 1, hidden_dim)
    _, dn_bbox_unact, _, _ = get_contrastive_denoising_training_group(
        targets=[target],
        num_classes=num_classes,
        num_queries=4,
        class_embed=class_embed,
        num_denoising=10,
        label_noise_ratio=0.0,
        box_noise_scale=1.0,
        box_mode="obb",
        angle_encoding=angle_encoding,
    )
    return torch.sigmoid(dn_bbox_unact[..., 4])


@pytest.mark.parametrize(
    "gt_theta, expected_shift",
    [
        (0.0, 0.25),
        (math.pi / 4, 0.5),
        (math.pi / 2, 0.75),
        (3 * math.pi / 4, 0.0),   # 0.75+0.25=1.0 → mod 1 → 0.0（seam）
    ],
)
def test_denoising_theta_shifted(gt_theta, expected_shift):
    """spec 站点 2: shifted 模式 GT θ → θ_shift ∈ [0,1)，无 clip。"""
    torch.manual_seed(0)
    theta_shift = _run_denoising_shifted(gt_theta)
    assert torch.allclose(
        theta_shift, torch.full_like(theta_shift, expected_shift), atol=1e-4
    ), (
        f"GT θ={gt_theta:.4f}: 期望 θ_shift={expected_shift}, "
        f"got min={theta_shift.min():.6f} max={theta_shift.max():.6f}"
    )


def test_denoising_default_angle_encoding_proportional():
    """站点 2: 未传 angle_encoding（默认 proportional）时 θ_norm 行为不变。"""
    torch.manual_seed(0)
    theta_norm = _run_denoising_shifted(math.pi / 4, angle_encoding="proportional")
    assert torch.allclose(
        theta_norm, torch.full_like(theta_norm, 0.25), atol=1e-4
    ), f"默认 proportional 应得 θ_norm=0.25, got {theta_norm.min():.6f}"
```

- [ ] **Step 2: 运行验证预期失败**

```bash
python -m pytest test/test_deimv2_obb_smoke.py -q -k "denoising_theta_shifted or denoising_default_angle"
```

Expected: 失败，`TypeError: ... unexpected keyword argument 'angle_encoding'`。

### User Implementation Gate

- [ ] **Step 3: 用户实现**

`engine/deim/denoising.py`：

1. import 区（第 9 行）改为：

```python
from .obb_angle_contract import physical_rad_to_norm, physical_rad_to_shifted_norm
```

2. 函数签名（第 12-21 行）追加 `angle_encoding="proportional"`：

```python
def get_contrastive_denoising_training_group(
    targets,
    num_classes,
    num_queries,
    class_embed,
    num_denoising=100,
    label_noise_ratio=0.5,
    box_noise_scale=1.0,
    box_mode="hbb",
    angle_encoding="proportional",
):
```

3. `box_mode == "obb"` 分支（第 112-115 行）改为：

```python
    elif box_mode == "obb":
        # [0,pi) → decoder 私有编码 [0,1)
        if angle_encoding == "shifted":
            input_query_bbox[..., 4] = physical_rad_to_shifted_norm(
                input_query_bbox[..., 4]
            )
        else:
            input_query_bbox[..., 4] = physical_rad_to_norm(input_query_bbox[..., 4])
        input_query_bbox = torch.cat([noise_spatial, input_query_bbox[..., 4:]], dim=-1)
```

`engine/deim/deim_decoder.py` 的 `DEIMTransformer.forward` 调用点（第 1220-1232 行）追加：

```python
                get_contrastive_denoising_training_group(
                    targets,
                    self.num_classes,
                    self.num_queries,
                    self.denoising_class_embed,
                    num_denoising=self.num_denoising,
                    label_noise_ratio=self.label_noise_ratio,
                    box_noise_scale=self.box_noise_scale,
                    box_mode=self.box_mode,
                    angle_encoding=self._resolved_angle_encoding(),
                )
```

### AI Green/Review Gate

- [ ] **Step 4: 运行定向 GREEN**

```bash
python -m pytest test/test_deimv2_obb_smoke.py -q -k "denoising"
```

Expected: 全绿（原 3 + 新 5 参数化 + 1 = 9）。

- [ ] **Step 5: 运行相关 GREEN（rep2 denoising 不回归）**

```bash
python -m pytest test/test_deimv2_obb_smoke.py -q -k "denoising or rep2"
```

Expected: 全绿（rep2 走 proportional 编码，`_obb_denoising_unact_to_rep2_unact` 不受影响）。

- [ ] **Step 6: 代码 review checklist**

- `angle_encoding` 默认 `"proportional"` → `dfine_decoder.py:960`、`rtdetrv2_decoder.py:574` 调用点零改动；
- 调用点传入 `self._resolved_angle_encoding()`（rep2 → proportional，GT θ 走等比编码，与 `_obb_denoising_unact_to_rep2_unact` 内部 `norm_to_physical_rad` 一致）；
- 角度不加噪（`noise_spatial` 仅含 xywh），θ 通道编码在噪声之后、`inverse_sigmoid` 之前。

- [ ] **Step 7: 提交（用户决定）**

```bash
git add engine/deim/denoising.py engine/deim/deim_decoder.py test/test_deimv2_obb_smoke.py
git commit -m "feat: denoising GT 角支持 shifted 编码（站点 2）"
```

---

## Review Unit 6: `deim_decoder.py` — 站点 3/4（geometry decode 前后）+ 站点 7（公开输出）+ 前向矩阵

**Files:**
- Modify: `engine/deim/deim_decoder.py`（`TransformerDecoder.forward` 第 482-497 行；`DEIMTransformer.forward` 第 1275-1296 行）
- Test: `test/test_deimv2_obb_smoke.py`、`test/test_deimv2_obb_rep2_eval.py`

**Interfaces:**
- Consumes: Unit 1 的 `shifted_norm_to_physical_rad` / `physical_rad_to_shifted_norm`；Unit 3 的 `_resolved_angle_encoding()`（两个类）
- Produces: 站点 3（geometry 前 θ_shift → 物理角）、站点 4（geometry 后物理角 → θ_shift）、站点 7（`out_bboxes`/`out_refs`/`pre_bboxes` 的 θ 公开还原）。`proportional` 逐位不变；rep2 全部直通。

### AI Test Gate（RED）

- [ ] **Step 1: 写 failing test**

在 `test/test_deimv2_obb_smoke.py` 追加前向矩阵测试（spec §12.2）：

```python
# ---------------------------------------------------------------------------
# Test 6a: 前向矩阵 4 表示 × 2 配置 — 公开 θ 域 + rep2 golden
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("angle_rep", "use_angle_first"),
    [(0, False), (1, False), (2, False), (3, True)],
)
@pytest.mark.parametrize("decoder_angle_encoding", ["proportional", "shifted"])
def test_forward_matrix_public_theta_domain(
    angle_rep, use_angle_first, decoder_angle_encoding
):
    """spec §12.2: 4 表示 × 2 配置，公开输出 θ ∈ [0, π)、无 NaN。

    覆盖站点 3/4/7：shifted 模式下 geometry decode 与公开输出都必须正确
    还原 θ_shift → 物理角。
    """
    torch.manual_seed(0)
    model = _make_obb_model(
        angle_rep=angle_rep,
        use_angle_first=use_angle_first,
        decoder_angle_encoding=decoder_angle_encoding,
    )
    model.train()
    feats = [torch.randn(1, 32, 8, 8), torch.randn(1, 32, 4, 4)]
    with torch.no_grad():
        outputs = model(feats)

    for name, tensor in [
        ("pred_boxes", outputs["pred_boxes"][..., 4]),
        ("ref_points", outputs["ref_points"][..., 4]),
        ("pre_bboxes", outputs["pre_outputs"]["pred_boxes"][..., 4]),
    ]:
        assert torch.isfinite(tensor).all(), (
            f"rep={angle_rep} enc={decoder_angle_encoding} {name} 含 NaN"
        )
        assert (tensor >= 0).all() and (tensor < math.pi).all(), (
            f"rep={angle_rep} enc={decoder_angle_encoding} {name} θ 应 ∈ [0, π), "
            f"got min={tensor.min():.4f} max={tensor.max():.4f}"
        )


def test_rep2_forward_bitwise_identical_across_modes():
    """spec §12.2: rep2 在 proportional 与 shifted 配置下输出逐位一致（golden）。

    rep2 私有 reference 不含显式 θ，_resolved_angle_encoding() 恒为
    proportional → 两种配置构造的模型完全同权，forward 必须逐位相同。
    """
    torch.manual_seed(0)
    feats = [torch.randn(1, 32, 8, 8), torch.randn(1, 32, 4, 4)]

    model_prop = _make_obb_model(angle_rep=2, decoder_angle_encoding="proportional")
    model_shift = _make_obb_model(angle_rep=2, decoder_angle_encoding="shifted")
    model_prop.train()
    model_shift.train()
    with torch.no_grad():
        out_prop = model_prop(feats)
        out_shift = model_shift(feats)

    for key in ["pred_boxes", "pred_logits", "pred_corners", "ref_points"]:
        assert torch.equal(out_prop[key], out_shift[key]), (
            f"rep2 {key} 在两种编码配置下必须逐位一致"
        )


def test_shifted_public_theta_matches_proportional_same_anchor():
    """spec §12.2 判别性 known-answer：零初始化下 shifted 与 proportional 的
    公开 θ 必须逐位接近（同为 anchor 物理角 45° → π/4）。

    zero-init 使 pre_bboxes == anchor 值：prop θ_norm=0.25 ↔ shifted
    θ_shift=0.5，二者物理角均为 π/4。若站点 3/4/7 任一未正确还原 shifted，
    shifted 公开 θ 会偏离 ~π/4（0.785），与 proportional 不一致。

    Given: rep0 + zero-init 头（_reset_parameters），同 seed 两个模型。
    When:  forward（train + no_grad）。
    Then:  pred_boxes / ref_points / pre_bboxes 的 θ 通道 allclose(1e-4)。
    """
    torch.manual_seed(0)
    feats = [torch.randn(1, 32, 8, 8), torch.randn(1, 32, 4, 4)]

    model_prop = _make_obb_model(angle_rep=0, decoder_angle_encoding="proportional")
    model_shift = _make_obb_model(angle_rep=0, decoder_angle_encoding="shifted")
    model_prop.train()
    model_shift.train()
    with torch.no_grad():
        out_prop = model_prop(feats)
        out_shift = model_shift(feats)

    for key, tensor_of in [
        ("pred_boxes", lambda o: o["pred_boxes"][..., 4]),
        ("ref_points", lambda o: o["ref_points"][..., 4]),
        ("pre_bboxes", lambda o: o["pre_outputs"]["pred_boxes"][..., 4]),
    ]:
        t_prop = tensor_of(out_prop)
        t_shift = tensor_of(out_shift)
        assert torch.allclose(t_prop, t_shift, atol=1e-4), (
            f"shifted 与 proportional 的 {key} θ 必须一致（同为 π/4），"
            f"got prop={t_prop.mean():.4f} shift={t_shift.mean():.4f}"
        )
```

在 `test/test_deimv2_obb_rep2_eval.py` 追加（公开 eval 输出回归）：

```python
def test_angle_rep2_eval_shifted_config_returns_public_physical_obb() -> None:
    """rep2 + decoder_angle_encoding=shifted 在 eval 下仍输出公开 5D 物理 OBB。

    与 proportional 配置输出逐位一致（rep2 强制 proportional）。
    """
    torch.manual_seed(0)
    feats = [torch.randn(1, 32, 4, 4), torch.randn(1, 32, 2, 2)]

    def run(encoding):
        torch.manual_seed(0)
        model = DEIMTransformer(
            num_classes=5,
            hidden_dim=32,
            num_queries=4,
            feat_channels=[32, 32],
            feat_strides=[4, 8],
            num_levels=2,
            num_points=2,
            nhead=4,
            num_layers=3,
            dim_feedforward=64,
            dropout=0.0,
            activation="relu",
            num_denoising=0,
            learn_query_content=False,
            eval_spatial_size=(16, 16),
            eval_idx=-1,
            eps=1e-2,
            aux_loss=False,
            cross_attn_method="default",
            query_select_method="default",
            reg_max=4,
            reg_scale=4.0,
            layer_scale=1,
            mlp_act="relu",
            use_gateway=True,
            share_bbox_head=False,
            share_score_head=False,
            box_mode="obb",
            angle_rep=2,
            offset_scale_source="pre",
            use_angle_first=False,
            decoder_angle_encoding=encoding,
        )
        model.eval()
        with torch.no_grad():
            return model(feats)

    out_prop = run("proportional")
    out_shift = run("shifted")
    for key in ["pred_boxes", "pred_logits"]:
        assert torch.equal(out_prop[key], out_shift[key]), (
            f"rep2 eval {key} 逐位一致"
        )
    pred_boxes = out_shift["pred_boxes"]
    assert pred_boxes.shape[-1] == 5
    assert torch.isfinite(pred_boxes).all()
    assert (pred_boxes[..., 4] >= 0).all()
    assert (pred_boxes[..., 4] < math.pi).all()
```

- [ ] **Step 2: 运行验证预期失败**

```bash
python -m pytest test/test_deimv2_obb_smoke.py -q -k "forward_matrix or rep2_forward_bitwise or shifted_public_theta"
python -m pytest test/test_deimv2_obb_rep2_eval.py -q
```

Expected: 失败 — 至少 `test_shifted_public_theta_matches_proportional_same_anchor` 的 `pred_boxes`/`ref_points`/`pre_bboxes` θ 与 proportional 相差 ~π/4（站点 7 未还原 shifted，输出的是 θ_shift 直通的错误物理值）。前向矩阵的域测试（θ ∈ [0,π)）在未实现时可能「假绿」（θ_shift*π 恰在 [0,π)），故以 known-answer 判别测试为准。rep2 golden `torch.equal` 应在 Unit 3 起即通过（rep2 强制 proportional），此处作为回归而非 RED。

### User Implementation Gate

- [ ] **Step 3: 用户实现**

`engine/deim/deim_decoder.py`：

1. `TransformerDecoder.forward` geometry decode（第 482-497 行）改为：

```python
            elif self.box_mode == "obb":
                if self._resolved_angle_encoding() == "shifted":
                    # 站点 3: θ_shift → 物理角，供 distance2bbox_obb 几何解码
                    ref_phys = torch.cat(
                        [
                            ref_points_initial[..., :4],
                            shifted_norm_to_physical_rad(ref_points_initial[..., 4:5]),
                        ],
                        dim=-1,
                    )
                else:
                    theta_scale = torch.ones_like(ref_points_initial)
                    # theta:[0,1]→[0,pi]
                    theta_scale[..., 4] *= torch.pi
                    ref_phys = ref_points_initial * theta_scale
                distance = integral(pred_corners, project)
                inter_ref_bbox = distance2bbox_obb(
                    ref_phys,
                    distance,
                    reg_scale,
                    offset_scale_source=self.offset_scale_source,
                )
                if self._resolved_angle_encoding() == "shifted":
                    # 站点 4: 物理角 → θ_shift，回到 decoder 私有 [0,1) 空间
                    inter_ref_bbox = torch.cat(
                        [
                            inter_ref_bbox[..., :4],
                            physical_rad_to_shifted_norm(inter_ref_bbox[..., 4:]),
                        ],
                        dim=-1,
                    )
                else:
                    inter_ref_bbox = torch.cat(
                        [inter_ref_bbox[..., :4], inter_ref_bbox[..., 4:] / torch.pi],
                        dim=-1,
                    )
```

2. `DEIMTransformer.forward` 公开输出（第 1275-1296 行）改为（用局部选择，避免三处重复分支）：

```python
        # criterion/matcher/postprocessor 中 theta 量纲为 [0, pi)
        # decoder 内部 [0,1) → 外部 [0, pi)
        if self.box_mode == "obb":
            if self._resolved_angle_encoding() == "shifted":
                # 站点 7: 公开输出 θ_shift → 物理角 [0, π)
                theta_decode = shifted_norm_to_physical_rad
            else:
                theta_decode = norm_to_physical_rad
            out_bboxes = torch.cat(
                [out_bboxes[..., :4], theta_decode(out_bboxes[..., 4:])], dim=-1
            )
            out_refs = torch.cat(
                [out_refs[..., :4], theta_decode(out_refs[..., 4:])], dim=-1
            )
            pre_bboxes = torch.cat(
                [pre_bboxes[..., :4], theta_decode(pre_bboxes[..., 4:])], dim=-1
            )
```

### AI Green/Review Gate

- [ ] **Step 4: 运行定向 GREEN**

```bash
python -m pytest test/test_deimv2_obb_smoke.py -q -k "forward_matrix or rep2_forward_bitwise"
python -m pytest test/test_deimv2_obb_rep2_eval.py -q
```

Expected: 全绿（8 参数化前向 + 1 rep2 golden；rep2_eval 2）。

- [ ] **Step 5: 运行相关 GREEN（完整 smoke + rep2）**

```bash
python -m pytest test/test_deimv2_obb_smoke.py test/test_deimv2_obb_rep2_eval.py -q
```

Expected: 全绿。

- [ ] **Step 6: 代码 review checklist**

- 站点 3/4 成对且同守卫：`shifted_norm_to_physical_rad`（前）↔ `physical_rad_to_shifted_norm`（后）；若只有一半守卫会破坏内部 θ ∈ [0,1) 不变量；
- 站点 7 使用局部 `theta_decode` 选择，三处输出统一；`out_refs`/`pre_bboxes` 与 `out_bboxes` 同步转换；
- rep2 分支（`angle_rep == 2` 的 `ref_points_initial` 5D θ_norm 路径）走 proportional：`ref_points_initial[..., 4:5]` 在 proportional 分支下被 `theta_scale[...,4]*π` 缩放，与现状逐位一致；
- `ref_points_initial` 为 5D 时 `[..., 4:5]` 与 `theta_scale[..., 4]` 语义对齐（rep0/1/3）；rep2/3 的 `inter_ref_bbox` 均为 5D θ_norm 直通；
- 站点 3/4/7 未触碰 `deim_criterion.py:378/385` 的公开物理输出再归一（隔离成立）。

- [ ] **Step 7: 提交（用户决定）**

```bash
git add engine/deim/deim_decoder.py test/test_deimv2_obb_smoke.py test/test_deimv2_obb_rep2_eval.py
git commit -m "feat: decoder geometry 与公开输出支持 shifted 编码（站点 3/4/7）"
```

---

## Review Unit 7: 配置入口 + 全量回归 + 收尾

**Files:**
- Modify: `configs/custom_obb/synthetic_configs/synthetic_exp_020_anrep0_offset_per.yml`（DEIMTransformer 块追加显式默认键，可选）
- 验证: 全量测试套件、`py_compile`、`git diff --check`

**Interfaces:**
- Consumes: 全部 Unit 1-6 产出
- Produces: `decoder_angle_encoding` yaml 键文档化（默认 `proportional`）；全量回归证据。

### User Implementation Gate

- [ ] **Step 1: 配置入口（可选但推荐）**

在 `configs/custom_obb/synthetic_configs/synthetic_exp_020_anrep0_offset_per.yml` 的 `DEIMTransformer:` 块（`box_mode: obb` / `angle_rep: 0` 附近，第 258-259 行后）追加：

```yaml
  decoder_angle_encoding: proportional
```

说明：默认值即 `proportional`，此键为显式文档化；后续 shifted 实验在复制出的新配置中改该键为 `shifted` 即可（经 `engine/core/workspace.py create()` 的 kwargs 管道直达构造器，已由 Unit 3 测试验证）。

### AI Green/Review Gate

- [ ] **Step 2: 全量 OBB 定向回归**

```bash
python -m pytest test/test_obb_angle_contract.py test/test_obb_domain_audit.py test/test_obb_roundtrip.py test/test_obb_adr_geometry.py test/test_obb_adr_loss.py test/test_matcher_obb_angle.py test/test_deim_criterion_obb_loss.py test/test_deim_postprocessor.py test/test_deimv2_obb_smoke.py test/test_deimv2_obb_rep2_eval.py test/test_obb_transforms.py test/test_obb_eval.py -q
```

Expected: 全绿（原 277 + 新增约 20）。

- [ ] **Step 3: 全量套件**

```bash
python -m pytest test/ -q
```

Expected: 全绿（spec §12.4：既有 406 测试套件 + 新增全部通过）。若个别文件因环境（如 DOTA 权重/数据）失败，记录为预存在项并在报告中注明。

- [ ] **Step 4: 静态检查**

```bash
python -m py_compile engine/deim/obb_angle_contract.py engine/deim/dfine_decoder.py engine/deim/deim_decoder.py engine/deim/denoising.py
git diff --check
```

Expected: 无输出（exit 0）。

- [ ] **Step 5: 术语残留扫描（spec 一致性）**

```bash
grep -rn "shifted_direct\|scope [AB]" engine/ test/ configs/ | grep -v "\.pyc" | head
```

Expected: 无输出（`shifted_direct`、scope A/B 已从全代码库清除）。

- [ ] **Step 6: 终局 review checklist（对照 spec §7/§8）**

- 站点 1/1b/2/3/4/5/6/7 全部存在且同守卫；无遗漏转换点；
- `query_pos_head`/`query_angle_head`、rep0 6D refinement、rep1/3 Δθ 残差、rep2 全部路径、`distance2bbox_obb`/`bbox2distance_obb`、criterion、matcher、postprocessor、eval、export 均未改动（spec §8 清单逐项核对）；
- `proportional` 默认值 + checkpoint 兼容：既有配置不传新键 → 逐位不变（Unit 6 矩阵 + 全量回归证明）；
- `physical_rad_to_loss_rad` 未接入（spec §9 二阶段，仅记录）。

- [ ] **Step 7: 提交（用户决定）**

```bash
git add configs/custom_obb/synthetic_configs/synthetic_exp_020_anrep0_offset_per.yml
git commit -m "docs: 基线配置显式声明 decoder_angle_encoding=proportional"
```

---

## Self-Review（计划编写阶段已完成）

1. **Spec 覆盖**：§5 纯函数 → Unit 1；§6 配置面 → Unit 3；§7 站点 1/1b → Unit 4、2 → Unit 5、3/4/7 → Unit 6、5 → Unit 2、6 → Unit 4；§12 测试 → 各 Unit RED；§13 文件清单全部覆盖。
2. **占位符扫描**：所有 Step 含具体代码/命令/Expected，无 "TBD/TODO/implement later"。
3. **类型一致性**：`physical_rad_to_shifted_norm` / `shifted_norm_to_physical_rad` / `decoder_angle_encoding` / `angle_encoding` / `_resolved_angle_encoding()` 命名与签名在全部 Unit 中一致；`_make_obb_model` helper 参数与 `DEIMTransformer` 构造参数对齐。

## Execution Handoff

**计划已完成并保存至 `docs/superpowers/plans/2026-08-07-obb-decoder-shifted-angle.md`。两种执行方式：**

1. **Subagent-Driven（推荐）** — 每个 Unit 派发独立 subagent，任务间人工 review，快速迭代
2. **Inline Execution** — 本会话内用 executing-plans 批量执行，带 checkpoint review

**注意**：本项目流程为 **AI 写测试（RED）→ 用户写生产代码 → AI 验证（GREEN）+ review**。无论选择哪种执行方式，每个 Review Unit 的「User Implementation Gate」都需要用户亲手修改 production 文件（AI 不直接改 production）。



