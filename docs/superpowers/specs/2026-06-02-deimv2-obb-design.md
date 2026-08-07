# Design: DEIMv2-OBB

## Architecture Overview

```
Image → Backbone → Neck → Encoder → Decoder → Head/Criterion → Loss
                                          │
                                          ├── HBB path (box_mode='hbb'): 4-dof DDF, GIoU loss
                                          └── OBB path (box_mode='obb'): 6-dist DDF, KLD+ProbIoU
```

## `box_mode` Propagation

```
config.yml: box_mode: 'obb'
    │
    ├─→ DEIMTransformer.__init__(box_mode='obb')
    │      → self.box_mode, self._num_box_dof=5, self.num_reg_dist=6
    │
    ├─→ DEIMCriterion.__init__(box_mode='obb')
    │      → self.box_mode, self.num_reg_dist=6
    │
    ├─→ PostProcessor.__init__(box_mode='obb')
    │
    └─→ Matcher.__init__(box_mode='obb')
```

## Component Changes

### New Files (zero risk to HBB)

| File | Purpose |
|------|---------|
| `obb_geometry.py` | Bi-directional mapping: external rect ↔ oriented box |
| `obb_ops.py` | ProbIoU, KLD loss for OBB |
| `chamfer_cost.py` | Vertex-set Chamfer distance for matching |
| `obb_dataset.py` | DOTA/DIOR-R dataset with OBB annotations |
| `configs/deimv2_obb_*.yml` | OBB training configs |

### Modified Files (HBB-gated)

| File | Change | Gate |
|------|--------|------|
| `deim_decoder.py` | 6-dist DDF, angle_factor, 5-dof output | `box_mode='hbb'` / `box_mode='obb'` |
| `deim_criterion.py` | OBB MAL + KLD losses, FGL target via bbox2distance_obb | `box_mode='hbb'` / `box_mode='obb'` |
| `matcher.py` | OBB cost types: cost_chamfer, cost_kld | `box_mode` field |
| `denoising.py` | OCD: angle noise, geometric noise, probability noise | `noise_mode` param |
| `postprocessor.py` | 5-dof output | `box_mode` field |
| `dfine_decoder.py` | Rotated cross-attention sampling (5-dof branch) | `ref.shape[-1] == 5` |
| `dfine_utils.py` | `bbox2distance_obb()` for OBB FGL targets | new function |

### Unmodified Files

All other files in `engine/deim/` remain unchanged.

## 6-Distribution ADR

Each oriented box is represented by 6 probability distributions:

```
External Rectangle (4 distributions):
  α — distance from center to left edge
  β — distance from center to top edge
  γ — distance from center to right edge
  δ — distance from center to bottom edge

Vertex Offsets (2 distributions):
  ε — distance from OBB top vertex to external rect top-right corner
  η — distance from OBB rightmost vertex to external rect bottom-right corner
```

The oriented box is reconstructed as:
1. External rect: `(cx-α, cy-β, cx+γ, cy+δ)` → `(x1, y1, x2, y2)` → `(c_cx, c_cy, Wr, Hr)`
2. Vertex offsets: `(ε, η)` relative to external rect corners → OBB vertices
3. OBB `(cx,cy,w,h,θ)` via `obb_geometry.decode()`

## Multi-Task Loss Weighting (Kendall Uncertainty)

### Design

Total loss is weighted by learnable task-specific uncertainty parameters σ_i:

```
L_total = Σ_i [ p_i · 0.5·exp(-2s_i) · L_i + p_i · s_i ]

s_i = log σ_i   (learnable task uncertainty)
p_i = weight_dict_i / mean(weight_dict)   (fixed prior multiplier)
```

- `p_i` encodes the user's fixed task preference (e.g., bbox > mal)
- `s_i` adapts to loss magnitude automatically
- Both are multiplicative → they compose without interference
- Equilibrium: `exp(-2s_i) = 1/L_i` regardless of `p_i` → prior doesn't affect self-adaptation

### Why not GradNorm

GradNorm (Chen et al., ICML 2018) was attempted but **fundamentally incompatible** with DEIMv2-OBB:

| Issue | Detail |
|-------|--------|
| No shared parameter bottleneck | `loss_mal` → decoder self-attn/ffn; `loss_bbox/kld/fgl` → `dec_bbox_head` MLPs. No single param receives gradient from all 4 losses. Verified via parameter-level grid search (260 params, 0 with full coverage). |
| `create_graph=True` breaks on deformable attention | `grid_sampler_2d_backward` second derivative not implemented in PyTorch → crashes in `torch.autograd.grad(L_grad, w_i)` |
| Probe-based init has bootstrap bias | Focal loss at cold-start has inflated gradient norm → `1/‖∇L_mal‖` → weight → 0 + collapse |

### Implementation

| File | Role |
|------|------|
| `engine/solver/kendall.py` | `KendallWeighting(nn.Module)`: learnable `log_sigma`, fixed `prior` buffer, `weighted_loss()`, `_aggregate_loss()` for aux/dn/enc/pre |
| `engine/solver/det_solver.py` | Reads `criterion.weight_dict`, computes `p_i = w_i/mean(w)`, passes as `prior` to `KendallWeighting`; creates separate `Adam(log_sigma, lr=sigma_lr)` |
| `engine/solver/det_engine.py` | Calls `kendall.weighted_loss(loss_dict)` in training loop; `kendall_optimizer.zero_grad()` + `.step()` alongside main optimizer |
| `configs/custom_obb/deimv2_obb_sp.yml` | `DEIMCriterion.weight_dict` provides `p_i` source; `KendallWeighting: {enabled: true, sigma_lr: 0.001}` |

Key invariant: `log_sigma` has its own optimizer (not added to main optimizer's param groups) — this avoids `FlatCosineLRScheduler`'s `base_lrs` indexing assuming a fixed number of groups.

## Key Invariants

1. **HBB path untouched**: When `box_mode='hbb'`, all code paths use the original logic
2. **Shape consistency**: HBB outputs `(bs,nq,4)`, OBB outputs `(bs,nq,5)`
3. **Weight compatibility**: OBB model can load HBB pretrained weights (new params are random init)
4. **Loss structure**: Same loss keys (`loss_bbox`, `loss_giou` → `loss_kld`, etc.), OBB adds `loss_obb`

## Risk Mitigation

- All OBB code isolated in `elif box_mode == 'obb':` branches
- Internal dispatch uses tensor shape (`ref.shape[-1] == 5`) not `box_mode`
- Every phase tested with HBB config first to verify no regression
