# H4: LQE Angle-Distribution Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a configurable `lqe_num_dist` parameter to LQE so OBB mode can use only the first 4 spatial distributions (α,β,γ,δ) for quality estimation, excluding the 2 angle distributions (ε,η), then run a short training + diagnostic to test whether angle-distribution pollution is the root cause of score↔IoU decoupling.

**Architecture:** LQE currently uses all `num_reg_dist` (=6 for OBB) distributions to compute `quality_score`. The change adds an `lqe_num_dist` parameter (default = `num_reg_dist`, preserving existing behavior) that slices `pred_corners` to only the first `lqe_num_dist` distributions before computing top-k stats. When `lqe_num_dist=4` in OBB mode, LQE only sees (α,β,γ,δ) — the spatial distances that correlate with IoU — and ignores (ε,η) — the vertex offsets that may introduce noise.

**Tech Stack:** PyTorch, DEIMv2 engine, YAML config

## Global Constraints

- Must not change HBB behavior: when `lqe_num_dist` is unset, it defaults to `num_reg_dist` (=4 for HBB, =6 for OBB), preserving existing behavior exactly
- Must not break checkpoint loading: the `reg_conf` MLP weight shape changes when `lqe_num_dist` differs from `num_reg_dist`, so OBB checkpoints from previous runs will not load into the new architecture — this is expected and acceptable for the H4 experiment
- `lqe_num_dist` must never exceed `num_reg_dist` (would slice beyond available data)
- Python environment: `/home/cx/apps/miniconda3/envs/deimv2/`
- Training config: `configs/custom_obb/synthetic_configs/synthetic_exp_020.yml`
- HBB compatibility: `box_mode="hbb"` must produce identical output to before (no `lqe_num_dist` in config → default behavior)

---

## File Structure

```
engine/deim/dfine_decoder.py  # MODIFY: LQE class — add lqe_num_dist param, slice pred_corners
engine/deim/deim_decoder.py   # MODIFY: TransformerDecoder + DEIMTransformer — pass lqe_num_dist through
configs/custom_obb/synthetic_configs/synthetic_exp_020.yml  # MODIFY: add lqe_num_dist: 4 to DEIMTransformer
test/test_lqe_ablation.py     # CREATE: unit test verifying LQE uses only first N distributions
```

---

### Task 1: Unit test for LQE distribution slicing

**Files:**
- Create: `test/test_lqe_ablation.py`

**Interfaces:**
- Produces: test verifying `LQE(lqe_num_dist=4)` only uses first 4 distributions of a 6-distribution input

- [ ] **Step 1: Write the failing test**

```python
"""Unit test for LQE lqe_num_dist parameter — verifies that setting
lqe_num_dist < num_reg_dist causes LQE to only use the first
lqe_num_dist distributions, ignoring the rest."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as F
from engine.deim.dfine_decoder import LQE


def test_lqe_uses_only_first_n_distributions():
    """When lqe_num_dist=4 and num_reg_dist=6, LQE should slice
    pred_corners to only the first 4*(reg_max+1) elements before
    computing quality_score. The reg_conf MLP input dimension should
    be 4*(k+1), not 6*(k+1)."""
    reg_max = 32
    k = 4
    num_reg_dist = 6
    lqe_num_dist = 4
    B, L = 2, 10

    lqe = LQE(k=k, hidden_dim=64, num_layers=2, reg_max=reg_max,
              num_reg_dist=num_reg_dist, lqe_num_dist=lqe_num_dist)

    # reg_conf input dim should be lqe_num_dist * (k+1) = 4*5 = 20
    assert lqe.reg_conf.layers[0].in_features == lqe_num_dist * (k + 1), \
        f"Expected {lqe_num_dist * (k + 1)}, got {lqe.reg_conf.layers[0].in_features}"

    # pred_corners: full 6-distribution input
    pred_corners = torch.randn(B, L, num_reg_dist * (reg_max + 1))
    scores = torch.zeros(B, L, 1)

    out = lqe(scores, pred_corners)

    assert out.shape == scores.shape, f"Output shape {out.shape} != expected {scores.shape}"

    # Verify: changing the last 2 distributions (ε,η) should NOT affect output
    pred_corners_2 = pred_corners.clone()
    pred_corners_2[:, :, lqe_num_dist * (reg_max + 1):] = torch.randn(
        B, L, (num_reg_dist - lqe_num_dist) * (reg_max + 1)
    )
    out_2 = lqe(scores, pred_corners_2)
    assert torch.allclose(out, out_2, atol=1e-6), \
        "LQE output changed when only the excluded distributions were modified — slicing is not working"


def test_lqe_default_preserves_existing_behavior():
    """When lqe_num_dist is not set (defaults to num_reg_dist),
    LQE should behave identically to the original implementation."""
    reg_max = 32
    k = 4
    num_reg_dist = 6
    B, L = 2, 10

    lqe_default = LQE(k=k, hidden_dim=64, num_layers=2, reg_max=reg_max,
                      num_reg_dist=num_reg_dist)
    lqe_explicit = LQE(k=k, hidden_dim=64, num_layers=2, reg_max=reg_max,
                       num_reg_dist=num_reg_dist, lqe_num_dist=num_reg_dist)

    # Same architecture
    assert lqe_default.reg_conf.layers[0].in_features == lqe_explicit.reg_conf.layers[0].in_features

    # Copy weights to ensure identical computation
    lqe_explicit.load_state_dict(lqe_default.state_dict())

    pred_corners = torch.randn(B, L, num_reg_dist * (reg_max + 1))
    scores = torch.zeros(B, L, 1)

    out_default = lqe_default(scores, pred_corners)
    out_explicit = lqe_explicit(scores, pred_corners)

    assert torch.allclose(out_default, out_explicit, atol=1e-6), \
        "Default lqe_num_dist should produce identical output to original"


def test_lqe_hbb_mode_unaffected():
    """HBB mode: num_reg_dist=4, lqe_num_dist defaults to 4.
    No slicing occurs, behavior unchanged."""
    reg_max = 32
    k = 4
    num_reg_dist = 4
    B, L = 2, 10

    lqe = LQE(k=k, hidden_dim=64, num_layers=2, reg_max=reg_max,
              num_reg_dist=num_reg_dist)

    # Should NOT accept lqe_num_dist > num_reg_dist
    try:
        lqe_bad = LQE(k=k, hidden_dim=64, num_layers=2, reg_max=reg_max,
                      num_reg_dist=num_reg_dist, lqe_num_dist=6)
        assert False, "Should have raised ValueError for lqe_num_dist > num_reg_dist"
    except ValueError:
        pass

    pred_corners = torch.randn(B, L, num_reg_dist * (reg_max + 1))
    scores = torch.zeros(B, L, 1)
    out = lqe(scores, pred_corners)
    assert out.shape == scores.shape


if __name__ == "__main__":
    test_lqe_uses_only_first_n_distributions()
    print("test_lqe_uses_only_first_n_distributions PASS")
    test_lqe_default_preserves_existing_behavior()
    print("test_lqe_default_preserves_existing_behavior PASS")
    test_lqe_hbb_mode_unaffected()
    print("test_lqe_hbb_mode_unaffected PASS")
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/cx/win_dir/thired/DEIMv2_DAOD && python test/test_lqe_ablation.py
```

Expected: FAIL with `TypeError: __init__() got an unexpected keyword argument 'lqe_num_dist'`

---

### Task 2: Modify LQE to accept `lqe_num_dist` parameter

**Files:**
- Modify: `engine/deim/dfine_decoder.py:334-359` (LQE class)

**Interfaces:**
- Consumes: nothing (leaf change)
- Produces: `LQE.__init__` now accepts `lqe_num_dist` parameter; `LQE.forward` slices `pred_corners` to first `lqe_num_dist` distributions

- [ ] **Step 1: Modify LQE.__init__ to accept lqe_num_dist**

In `engine/deim/dfine_decoder.py`, replace the LQE class (lines 334-359) with:

```python
class LQE(nn.Module):
    def __init__(self, k, hidden_dim, num_layers, reg_max, act="relu", num_reg_dist=4, lqe_num_dist=None):
        """
        参考文献： Generalized Focal Loss V2 https://arxiv.org/abs/2011.12885
        根据文献 k 取值 4 效果较好

        Args:
            lqe_num_dist: Number of distributions to use for quality estimation.
                          If None, defaults to num_reg_dist (use all distributions).
                          If set < num_reg_dist, only the first lqe_num_dist
                          distributions are used (e.g. 4 for OBB to exclude angle ε,η).
        """

        super(LQE, self).__init__()
        self.k = k
        self.reg_max = reg_max
        self.num_reg_dist = num_reg_dist
        self.lqe_num_dist = lqe_num_dist if lqe_num_dist is not None else num_reg_dist
        if self.lqe_num_dist > self.num_reg_dist:
            raise ValueError(
                f"lqe_num_dist ({self.lqe_num_dist}) cannot exceed num_reg_dist ({self.num_reg_dist})"
            )
        self.reg_conf = MLP(
            self.lqe_num_dist * (k + 1), hidden_dim, 1, num_layers, act=act
        )
        init.constant_(self.reg_conf.layers[-1].bias, 0)
        init.constant_(self.reg_conf.layers[-1].weight, 0)

    def forward(self, scores, pred_corners):
        B, L, _ = pred_corners.size()
        # Slice to only the first lqe_num_dist distributions
        slice_len = self.lqe_num_dist * (self.reg_max + 1)
        pred_corners_sliced = pred_corners[..., :slice_len]
        prob = F.softmax(
            pred_corners_sliced.reshape(B, L, self.lqe_num_dist, self.reg_max + 1), dim=-1
        )
        prob_topk, _ = prob.topk(self.k, dim=-1)
        stat = torch.cat([prob_topk, prob_topk.mean(dim=-1, keepdim=True)], dim=-1)
        quality_score = self.reg_conf(stat.reshape(B, L, -1))
        return scores + quality_score
```

- [ ] **Step 2: Run unit test to verify it passes**

```bash
cd /home/cx/win_dir/thired/DEIMv2_DAOD && python test/test_lqe_ablation.py
```

Expected: All 3 tests PASS

- [ ] **Step 3: Commit**

```bash
cd /home/cx/win_dir/thired/DEIMv2_DAOD
git add engine/deim/dfine_decoder.py test/test_lqe_ablation.py
git commit -m "feat(lqe): add lqe_num_dist param to slice distributions for quality estimation

H4 experiment: allow LQE to use only first N distributions (e.g. 4 for OBB
to exclude angle ε,η). Default behavior unchanged (uses all num_reg_dist)."
```

---

### Task 3: Pass `lqe_num_dist` through TransformerDecoder

**Files:**
- Modify: `engine/deim/deim_decoder.py:132-170` (TransformerDecoder.__init__)

**Interfaces:**
- Consumes: `lqe_num_dist` parameter from DEIMTransformer
- Produces: `TransformerDecoder` creates LQE instances with `lqe_num_dist` passed through

- [ ] **Step 1: Add lqe_num_dist to TransformerDecoder.__init__ signature and LQE creation**

In `engine/deim/deim_decoder.py`, modify `TransformerDecoder.__init__` (line 132-170).

Add `lqe_num_dist=None` to the parameter list (after `box_mode="hbb"`):

```python
    def __init__(
        self,
        hidden_dim,
        decoder_layer,
        decoder_layer_wide,
        num_layers,
        num_head,
        reg_max,
        reg_scale,
        up,
        eval_idx=-1,
        layer_scale=2,
        act="relu",
        num_reg_dist=4,
        box_mode="hbb",
        lqe_num_dist=None,
    ):
```

Then modify the LQE creation (line 163-170) to pass `lqe_num_dist`:

```python
        self.lqe_layers = nn.ModuleList(
            [
                copy.deepcopy(
                    LQE(4, 64, 2, reg_max, act=act, num_reg_dist=num_reg_dist,
                        lqe_num_dist=lqe_num_dist)
                )
                for _ in range(num_layers)
            ]
        )
```

- [ ] **Step 2: Verify no syntax errors**

```bash
cd /home/cx/win_dir/thired/DEIMv2_DAOD && python3 -c "
import ast
ast.parse(open('engine/deim/deim_decoder.py').read())
print('Syntax OK')
"
```

Expected: `Syntax OK`

- [ ] **Step 3: Commit**

```bash
cd /home/cx/win_dir/thired/DEIMv2_DAOD
git add engine/deim/deim_decoder.py
git commit -m "feat(decoder): pass lqe_num_dist through TransformerDecoder to LQE"
```

---

### Task 4: Pass `lqe_num_dist` through DEIMTransformer

**Files:**
- Modify: `engine/deim/deim_decoder.py:315-428` (DEIMTransformer.__init__ + decoder creation)

**Interfaces:**
- Consumes: `lqe_num_dist` from YAML config
- Produces: DEIMTransformer creates TransformerDecoder with `lqe_num_dist` passed through

- [ ] **Step 1: Add lqe_num_dist to DEIMTransformer.__init__ signature**

In `engine/deim/deim_decoder.py`, add `lqe_num_dist=None` to `DEIMTransformer.__init__` (after `box_mode="hbb"`, line 346):

```python
    def __init__(
        self,
        num_classes=80,
        hidden_dim=256,
        num_queries=300,
        feat_channels=[512, 1024, 2048],
        feat_strides=[8, 16, 32],
        num_levels=3,
        num_points=4,
        nhead=8,
        num_layers=6,
        dim_feedforward=1024,
        dropout=0.0,
        activation="relu",
        num_denoising=100,
        label_noise_ratio=0.5,
        box_noise_scale=1.0,
        learn_query_content=False,
        eval_spatial_size=None,
        eval_idx=-1,
        eps=1e-2,
        aux_loss=True,
        cross_attn_method="default",
        query_select_method="default",
        reg_max=32,
        reg_scale=4.0,
        layer_scale=1,
        mlp_act="relu",
        use_gateway=True,
        share_bbox_head=False,
        share_score_head=False,
        box_mode="hbb",
        lqe_num_dist=None,
    ):
```

- [ ] **Step 2: Pass lqe_num_dist to TransformerDecoder**

In the same file, find the `TransformerDecoder` creation (line 414-428) and add `lqe_num_dist=lqe_num_dist`:

```python
        self.decoder = TransformerDecoder(
            hidden_dim,
            decoder_layer,
            decoder_layer_wide,
            num_layers,
            nhead,
            reg_max,
            self.reg_scale,
            self.up,
            eval_idx,
            layer_scale,
            act=activation,
            num_reg_dist=self.num_reg_dist,
            box_mode=self.box_mode,
            lqe_num_dist=lqe_num_dist,
        )
```

- [ ] **Step 3: Verify syntax and import**

```bash
cd /home/cx/win_dir/thired/DEIMv2_DAOD && python3 -c "
import ast
ast.parse(open('engine/deim/deim_decoder.py').read())
print('Syntax OK')
import sys; sys.path.insert(0, '.')
from engine.deim.deim_decoder import DEIMTransformer
print('Import OK')
"
```

Expected: `Syntax OK` and `Import OK`

- [ ] **Step 4: Run existing tests to verify no regression**

```bash
cd /home/cx/win_dir/thired/DEIMv2_DAOD && python test/test_lqe_ablation.py
```

Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
cd /home/cx/win_dir/thired/DEIMv2_DAOD
git add engine/deim/deim_decoder.py
git commit -m "feat(transformer): expose lqe_num_dist in DEIMTransformer config

Allows YAML config to set lqe_num_dist=4 for OBB mode to exclude angle
distributions from LQE quality estimation."
```

---

### Task 5: Add `lqe_num_dist: 4` to synthetic_exp_020 config

**Files:**
- Modify: `configs/custom_obb/synthetic_configs/synthetic_exp_020.yml`

**Interfaces:**
- Produces: config that triggers LQE to use only 4 distributions in OBB mode

- [ ] **Step 1: Add lqe_num_dist to the DEIMTransformer config section**

In `configs/custom_obb/synthetic_configs/synthetic_exp_020.yml`, find the `DEIMTransformer` section (around line 280+) and add `lqe_num_dist: 4` alongside `box_mode: obb`:

```yaml
DEIMTransformer:
  box_mode: "obb"
  lqe_num_dist: 4
  # ... rest of existing config unchanged
```

Find the exact location by searching for `box_mode` in the file:

```bash
cd /home/cx/win_dir/thired/DEIMv2_DAOD && grep -n "box_mode" configs/custom_obb/synthetic_configs/synthetic_exp_020.yml
```

Add `lqe_num_dist: 4` on the line after `box_mode: obb`.

- [ ] **Step 2: Verify config loads and LQE gets lqe_num_dist=4**

```bash
cd /home/cx/win_dir/thired/DEIMv2_DAOD && python3 -c "
import sys; sys.path.insert(0, '.')
from engine.core import YAMLConfig
cfg = YAMLConfig('configs/custom_obb/synthetic_configs/synthetic_exp_020.yml')
model = cfg.model
# Check LQE layers have lqe_num_dist=4
decoder = model.decoder if hasattr(model, 'decoder') else model.model.decoder
lqe = decoder.lqe_layers[0]
print(f'LQE lqe_num_dist = {lqe.lqe_num_dist}')
print(f'LQE reg_conf input dim = {lqe.reg_conf.layers[0].in_features}')
assert lqe.lqe_num_dist == 4, f'Expected 4, got {lqe.lqe_num_dist}'
assert lqe.reg_conf.layers[0].in_features == 4 * (4 + 1), f'Expected 20, got {lqe.reg_conf.layers[0].in_features}'
print('Config verification PASS')
"
```

Expected: `LQE lqe_num_dist = 4` and `LQE reg_conf input dim = 20` and `Config verification PASS`

- [ ] **Step 3: Commit**

```bash
cd /home/cx/win_dir/thired/DEIMv2_DAOD
git add configs/custom_obb/synthetic_configs/synthetic_exp_020.yml
git commit -m "config: set lqe_num_dist=4 for H4 OBB ablation experiment"
```

---

### Task 6: Smoke test — forward pass with new config

**Files:**
- Test: no new files

**Interfaces:**
- Produces: confirmation that forward + backward pass works with `lqe_num_dist=4`

- [ ] **Step 1: Run a single-batch forward+backward smoke test**

```bash
cd /home/cx/win_dir/thired/DEIMv2_DAOD && python3 -c "
import sys, torch; sys.path.insert(0, '.')
from engine.core import YAMLConfig
from engine.solver import TASKS

cfg = YAMLConfig('configs/custom_obb/synthetic_configs/synthetic_exp_020.yml')
solver = TASKS[cfg.yaml_cfg['task']](cfg)
solver.train()

# Get one batch
samples, targets = next(iter(solver.train_dataloader))
device = torch.device('cuda')
samples = samples.to(device)
targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

# Forward
outputs = solver.model(samples, targets=targets)
print(f'Forward OK: pred_logits {outputs[\"pred_logits\"].shape}, pred_boxes {outputs[\"pred_boxes\"].shape}')

# Backward
loss_dict = solver.criterion(outputs, targets, epoch=0)
loss = sum(loss_dict.values())
loss.backward()
print(f'Backward OK: total_loss = {loss.item():.4f}')
print('Smoke test PASS')
" 2>&1 | tail -20
```

Expected: `Forward OK` + `Backward OK` + `Smoke test PASS` with no errors

- [ ] **Step 2: If smoke test fails, debug before proceeding**

Common failure modes:
- `RuntimeError: shape mismatch` in LQE forward → check slice_len calculation
- `KeyError: 'lqe_num_dist'` in config → check YAML indentation
- `size mismatch for reg_conf` in checkpoint loading → expected (new architecture, can't load old checkpoint); use `--resume` with no checkpoint or train from scratch

---

### Task 7: Run short training + diagnostic (the H4 experiment itself)

**Files:**
- Run: `train.py` with `synthetic_exp_020.yml`
- Run: `test/diagnose_hungarian_matching.py`

**Interfaces:**
- Produces: training log + matching diagnostic report for H4 experiment

- [ ] **Step 1: Run 30-epoch training**

```bash
cd /home/cx/win_dir/thired/DEIMv2_DAOD
python train.py -c configs/custom_obb/synthetic_configs/synthetic_exp_020.yml 2>&1 | tail -5
```

Note: this overwrites `outputs/synthetic_exp_020/last.pth`. The training should take ~30 minutes on RTX 4060 Ti.

Expected: training completes without crash. The config already has `matcher_change_epoch: 22` from the H2 experiment.

- [ ] **Step 2: Run the Hungarian matching diagnostic**

```bash
cd /home/cx/win_dir/thired/DEIMv2_DAOD
python test/diagnose_hungarian_matching.py
```

Expected: produces `test/outputs/matching_diag/matching_report.txt` with updated Q1/Q2/Q3.

- [ ] **Step 3: Run inference diagnostic**

```bash
cd /home/cx/win_dir/thired/DEIMv2_DAOD
python test/test_infer_diag.py --num 20
```

Expected: produces `test/outputs/infer_diag/score_dist.txt` with updated score distribution.

- [ ] **Step 4: Record results and compare to H2 baseline**

Create a comparison table:

| Metric | H2 baseline (matcher_change_epoch:22) | H4 (lqe_num_dist:4) | Change |
|--------|--------------------------------------|---------------------|--------|
| Q3 Pearson r | 0.0843 | ? | ? |
| Q3 Spearman ρ | -0.0149 | ? | ? |
| Q2 class norm_sep | 5.15 | ? | ? |
| Q2 bbox norm_sep | 0.99 | ? | ? |
| Q2 chamfer norm_sep | 0.57 | ? | ? |
| Q2 probiou norm_sep | 1.82 | ? | ? |
| test AP50 (ep29) | 0.876 | ? | ? |
| test AP75 (ep29) | 0.831 | ? | ? |
| test Precision | 0.786 | ? | ? |

**Decision gate:**
- If Q3 Pearson r ≥ 0.2 → **H4 confirmed**: LQE angle pollution was a root cause. Consider making `lqe_num_dist=4` the default for OBB.
- If Q3 Pearson r ∈ [0.12, 0.2) → **partial**: LQE pollution contributed but is not the sole cause. Combine H4 + H3 (fix mal_iou_type) next.
- If Q3 Pearson r < 0.12 → **H4 rejected**: LQE pollution is not the root cause. Move to H3 (mal_iou_type) or reconsider decoder decoupling.

- [ ] **Step 5: Commit results**

```bash
cd /home/cx/win_dir/thired/DEIMv2_DAOD
git add -A
git commit -m "experiment: H4 LQE ablation results — lqe_num_dist=4 on density_020"
```

---

## Self-Review

**1. Spec coverage:**
- ✅ LQE slicing: Task 2 modifies LQE.__init__ and forward
- ✅ Parameter passthrough: Task 3 (TransformerDecoder), Task 4 (DEIMTransformer)
- ✅ Config: Task 5 adds lqe_num_dist: 4 to YAML
- ✅ Unit test: Task 1 tests slicing, default behavior, HBB compat
- ✅ Smoke test: Task 6 verifies forward+backward
- ✅ Experiment: Task 7 runs training + diagnostic

**2. Placeholder scan:** No TBD/TODO. All code blocks are complete. All commands have expected outputs.

**3. Type consistency:**
- `lqe_num_dist` used consistently across LQE.__init__, TransformerDecoder.__init__, DEIMTransformer.__init__
- Default `None` → falls back to `num_reg_dist` everywhere
- Unit test checks `reg_conf.layers[0].in_features == lqe_num_dist * (k + 1)` — consistent with LQE.__init__ using `self.lqe_num_dist * (k + 1)`
