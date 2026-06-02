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

## Key Invariants

1. **HBB path untouched**: When `box_mode='hbb'`, all code paths use the original logic
2. **Shape consistency**: HBB outputs `(bs,nq,4)`, OBB outputs `(bs,nq,5)`
3. **Weight compatibility**: OBB model can load HBB pretrained weights (new params are random init)
4. **Loss structure**: Same loss keys (`loss_bbox`, `loss_giou` → `loss_kld`, etc.), OBB adds `loss_obb`

## Risk Mitigation

- All OBB code isolated in `elif box_mode == 'obb':` branches
- Internal dispatch uses tensor shape (`ref.shape[-1] == 5`) not `box_mode`
- Every phase tested with HBB config first to verify no regression
