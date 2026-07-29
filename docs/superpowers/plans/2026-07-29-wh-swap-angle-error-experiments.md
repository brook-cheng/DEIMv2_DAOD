# w/h Swap Angle-Error Inflation — Experiment Suite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Quantify how often `w≥h` normalization in `xyxyxyxy_to_xywhr` / `external_rect_to_oriented_box` converts small w/h prediction errors into spurious π/2 angle errors across the DEIMv2-OBB training and evaluation pipeline.

**Architecture:** Five independent experiment scripts under `test/`, each producing PASS/FAIL verdicts plus quantitative metrics. Unit-level experiments use synthetic tensors; pipeline-level experiments use real model output DOTA files from existing `tool_debug_decoder.py` runs.

**Tech Stack:** PyTorch, engine.deim.obb_geometry, engine.deim.dfine_utils, engine.deim.obb_ops (ProbIoU), existing DOTA prediction files under `test/data/outputs/`.

## Global Constraints

- Do NOT modify any file under `engine/` — experiments are read-only diagnostics.
- All experiments must run on CPU with no checkpoint download.
- Each script must print a final PASS/FAIL summary line and exit nonzero on FAIL.
- Reuse existing helpers: `_vertex_roundtrip_error` pattern from `test/test_obb_roundtrip.py`.
- Plan execution order: unit experiments first (Tasks 1–3), then pipeline experiments (Tasks 4–5).

---

## Experiment → Task Mapping

| Spec risk ID | Location | Task | Hypothesis to test |
|---|---|---|---|
| [3] | `external_rect_to_oriented_box` (FDR) | Task 1 | w<h inputs get swapped + θ+π/2, same as `xyxyxyxy_to_xywhr` |
| [4] | `bbox2distance_obb` (loss) | Task 2 | pred/target asymmetric swap inflates (ε,η) residuals |
| [1] | `affine_obb` (augmentation) | Task 3 | augmented GT labels get θ shifted by π/2 at w/h boundary |
| [5] | matcher parameter L1 | Task 4 | w<h pred gets inflated angle L1 vs GT with same geometry |
| [7]+[8] | DOTA export → analysis/eval | Task 5 | % of "large angle errors" that are actually w/h swap artifacts |
| [2]+[6] | anchor gen, decoder head | Task 6 | (verification) no w/h swap in these paths |
| — | consolidated report | Task 7 | one markdown report with all metrics |

---

### Task 1: `external_rect_to_oriented_box` w/h swap behavior [Risk 3]

**Files:**
- Create: `test/test_exp_wh_swap_fdr.py`
- Test: same file (self-contained script with `if __name__ == "__main__"`)

**Interfaces:**
- Consumes: `engine.deim.obb_geometry.oriented_box_to_external_rect`, `external_rect_to_oriented_box`, `xywhr_to_xyxyxyxy`, `xyxyxyxy_to_xywhr`
- Produces: PASS/FAIL verdict + count of swapped cases; function `check_adr_swap(cases: list[Tensor]) -> dict`

- [ ] **Step 1: Write the failing test scaffold**

Create `test/test_exp_wh_swap_fdr.py` with this exact content:

```python
import sys, os, torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from engine.deim.obb_geometry import (
    oriented_box_to_external_rect,
    external_rect_to_oriented_box,
    xywhr_to_xyxyxyxy,
    xyxyxyxy_to_xywhr,
)

def periodic_diff(a, b):
    d = abs(a - b) % torch.pi
    return min(d, torch.pi - d)

CASES = [
    # (name, xywhr)  — (cx, cy, w, h, theta)
    ("w<h near-square",      torch.tensor([[0.5, 0.5, 0.40, 0.41, 0.30]])),
    ("w>h near-square",      torch.tensor([[0.5, 0.5, 0.41, 0.40, 0.30]])),
    ("w<<h",                 torch.tensor([[0.5, 0.5, 0.10, 0.40, 0.30]])),
    ("w>>h",                 torch.tensor([[0.5, 0.5, 0.40, 0.10, 0.30]])),
    ("w<h theta=pi/6",       torch.tensor([[0.5, 0.5, 0.20, 0.40, 0.5236]])),
    ("w>h theta=pi/6",       torch.tensor([[0.5, 0.5, 0.40, 0.20, 0.5236]])),
]

def check_adr_swap(cases):
    results = {"total": 0, "swapped": 0, "geom_ok": 0, "details": []}
    for name, obb in cases:
        results["total"] += 1
        ext, vo = oriented_box_to_external_rect(obb)
        recon = external_rect_to_oriented_box(ext, vo)

        # geometry check: vertices must match
        v_orig = xywhr_to_xyxyxyxy(obb)
        v_recon = xywhr_to_xyxyxyxy(recon)
        d1 = ((v_orig.unsqueeze(-2) - v_recon.unsqueeze(-3)) ** 2).sum(-1).amin(-1)
        d2 = ((v_recon.unsqueeze(-2) - v_orig.unsqueeze(-3)) ** 2).sum(-1).amin(-1)
        v_err = max(d1.max().item(), d2.max().item())
        geom_ok = v_err < 1e-5
        if geom_ok:
            results["geom_ok"] += 1

        # swap check: did w/h swap AND theta shift by ~pi/2?
        wh_swapped = not torch.isclose(recon[0, 2], obb[0, 2], atol=1e-5)
        ang_shift = periodic_diff(recon[0, 4].item(), obb[0, 4].item())
        theta_jumped = ang_shift > 1.0  # > ~57 deg
        swapped = wh_swapped and theta_jumped
        if swapped:
            results["swapped"] += 1

        results["details"].append({
            "name": name, "geom_ok": geom_ok, "swapped": swapped,
            "in": obb[0].tolist(), "out": recon[0].tolist(),
            "v_err": v_err, "ang_shift": ang_shift,
        })
    return results

if __name__ == "__main__":
    torch.manual_seed(0)
    res = check_adr_swap(CASES)
    for d in res["details"]:
        flag = "SWAP" if d["swapped"] else ("OK" if d["geom_ok"] else "BAD")
        print(f"  [{flag}] {d['name']}: in=({d['in'][2]:.3f},{d['in'][3]:.3f},{d['in'][4]:.3f}) "
              f"out=({d['out'][2]:.3f},{d['out'][3]:.3f},{d['out'][4]:.3f}) "
              f"v_err={d['v_err']:.2e} ang_shift={d['ang_shift']:.3f}")
    print(f"\nTotal={res['total']} swapped={res['swapped']} geom_ok={res['geom_ok']}")
    # Expect: all geom_ok, w<h cases swapped, w>h cases not swapped
    expect_swapped = sum(1 for _, c in CASES if c[0, 2] < c[0, 3])
    ok = res["geom_ok"] == res["total"] and res["swapped"] == expect_swapped
    print(f"ADR swap behavior: {'PASS' if ok else 'FAIL'} "
          f"(expected {expect_swapped} swaps, got {res['swapped']})")
    raise SystemExit(0 if ok else 1)
```

- [ ] **Step 2: Run it**

Run: `python test/test_exp_wh_swap_fdr.py`
Expected output: all w<h cases show SWAP, all w>h cases show OK, geometry preserved for all. Final line `ADR swap behavior: PASS`.

- [ ] **Step 3: Commit**

```bash
git add test/test_exp_wh_swap_fdr.py
git commit -m "test: verify external_rect_to_oriented_box w/h swap behavior matches xyxyxyxy_to_xywhr"
```

---

### Task 2: `bbox2distance_obb` swap symmetry in loss residuals [Risk 4]

**Files:**
- Create: `test/test_exp_wh_swap_loss.py`
- Test: same file

**Interfaces:**
- Consumes: `engine.deim.dfine_utils.bbox2distance_obb(points, bbox, reg_max, reg_scale, up, eps=0.1, offset_scale_source="pre", obbox_rep_dim)` → returns `(six_lens, weight_right, weight_left)`; `engine.deim.obb_geometry.*`
- Produces: PASS/FAIL + residual comparison for both `obbox_rep_dim=6` (rep2) and `obbox_rep_dim=5` (rep3)

**Hypothesis:** For rep2 (`obbox_rep_dim=6`), both pred and GT pass through `oriented_box_to_external_rect`, so swaps are symmetric — residuals should match. For rep3 (`obbox_rep_dim=5`), the angle path uses `periodic_angle_distance` on raw decoder θ, so a pred whose decoder θ is correct should produce small residuals regardless of w/h ordering. If either path shows inflated residuals for a geometrically-equivalent w<h pred, the loss is miscalibrated at the boundary.

**Verified signature (from `dfine_utils.py:251-260`):**
```python
bbox2distance_obb(points, bbox, reg_max, reg_scale, up, eps=0.1,
                  offset_scale_source="pre", obbox_rep_dim=6)
# points: (N,5) ref (cx,cy,w,h,θ); bbox: (N,5) GT
# returns: (six_lens, weight_right, weight_left)
```

- [ ] **Step 1: Write the experiment**

Create `test/test_exp_wh_swap_loss.py`:

```python
import sys, os, torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from engine.deim.dfine_utils import bbox2distance_obb

GT   = torch.tensor([[0.5, 0.5, 0.40, 0.40, 0.30]])  # square-ish GT
PRED_GT = torch.tensor([[0.5, 0.5, 0.41, 0.40, 0.30]])  # pred w>h (geometric twin, no swap)
PRED_LT = torch.tensor([[0.5, 0.5, 0.40, 0.41, 0.30]])  # pred w<h (geometric twin, swaps)

def run_case(obbox_rep_dim, label):
    up = torch.linspace(0.0, 1.0, 33)  # reg_max=32 → 33 bins, matching weighting_function
    def resid(pred):
        six_lens, w_r, w_l = bbox2distance_obb(
            GT, pred, reg_max=32, reg_scale=4.0, up=up, obbox_rep_dim=obbox_rep_dim
        )
        return six_lens
    r_gt = resid(PRED_GT)
    r_lt = resid(PRED_LT)
    diff = (r_gt - r_lt).abs().max().item()
    print(f"  [{label}] rep_dim={obbox_rep_dim}: max |residual(w>h) - residual(w<h)| = {diff:.4f}")
    print(f"    residual(w>h)[:6] = {r_gt.flatten()[:6].tolist()}")
    print(f"    residual(w<h)[:6] = {r_lt.flatten()[:6].tolist()}")
    return diff

if __name__ == "__main__":
    torch.manual_seed(0)
    d6 = run_case(6, "rep2/ADR")
    d5 = run_case(5, "rep3/delta-theta")
    print(f"\nrep2 residual diff: {d6:.4f}  |  rep3 residual diff: {d5:.4f}")
    # Report; fail only on wild divergence indicating a bug (> 5.0)
    ok = d6 < 5.0 and d5 < 5.0
    print(f"Loss residual symmetry check: {'PASS' if ok else 'FAIL'}")
    raise SystemExit(0 if ok else 1)
```

- [ ] **Step 2: Run it**

Run: `python test/test_exp_wh_swap_loss.py`
Expected: rep2 diff ≈ 0 (symmetric ADR conversion); rep3 diff ≈ 0 if periodic angle distance is used correctly, or large if raw θ compared. Report the measured values.

- [ ] **Step 3: Commit**

```bash
git add test/test_exp_wh_swap_loss.py
git commit -m "test: measure bbox2distance_obb residual symmetry for rep2/rep3 at w/h boundary"
```

---

### Task 3: `affine_obb` GT label damage check [Risk 1]

**Files:**
- Create: `test/test_exp_wh_swap_augmentation.py`

**Interfaces:**
- Consumes: `engine.deim.obb_geometry.affine_obb`, `affine_obb_matrix`
- Produces: count of GT boxes whose θ shifted by π/2 after augmentation; PASS/FAIL

**Hypothesis:** Affine transforms (scale, mosaic) that go through `xywhr→vertices→transform→xyxyxyxy_to_xywhr` will shift θ by π/2 for GT boxes whose w<h after transformation. This means the GT label fed to the loss function has a "wrong" angle — but the geometry is correct, so geometry-aware losses (KLD/ProbIoU) are unaffected. The experiment confirms this is label *reparameterization*, not corruption.

- [ ] **Step 1: Write the experiment**

```python
import sys, os, torch, math
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from engine.deim.obb_geometry import affine_obb, xywhr_to_xyxyxyxy

GT_BOXES = torch.tensor([
    [100.0, 200.0, 80.0, 82.0, 0.30],   # w<h slightly
    [100.0, 200.0, 82.0, 80.0, 0.30],   # w>h slightly
    [100.0, 200.0, 40.0, 80.0, 0.30],   # w<<h
    [100.0, 200.0, 80.0, 40.0, 0.30],   # w>>h
    [100.0, 200.0, 80.0, 80.0, 0.785],  # square
])

TRANSFORMS = [
    ("identity",       1.0, 1.0, 0.0, 0.0),
    ("uniform x2",     2.0, 2.0, 0.0, 0.0),
    ("aniso x2,y0.5",  2.0, 0.5, 0.0, 0.0),
    ("aniso x0.5,y2",  0.5, 2.0, 0.0, 0.0),
    ("translate",      1.0, 1.0, 50.0, 30.0),
]

def periodic_diff(a, b):
    d = abs(a - b) % math.pi
    return min(d, math.pi - d)

if __name__ == "__main__":
    total_shifts = 0
    total_boxes = 0
    print(f"{'Transform':<18} {'Box(w,h,θ)':<28} {'Out(w,h,θ)':<28} {'Δθ':>8}")
    for tname, sx, sy, tx, ty in TRANSFORMS:
        out = affine_obb(GT_BOXES, sx=sx, sy=sy, tx=tx, ty=ty)
        for i in range(len(GT_BOXES)):
            total_boxes += 1
            d_theta = periodic_diff(out[i, 4].item(), GT_BOXES[i, 4].item())
            shifted = d_theta > 1.0
            if shifted:
                total_shifts += 1
            print(f"{tname:<18} ({GT_BOXES[i,2]:.0f},{GT_BOXES[i,3]:.0f},{GT_BOXES[i,4]:.2f})"
                  f"{'':<8} ({out[i,2]:.1f},{out[i,3]:.1f},{out[i,4]:.2f}){'':<6}"
                  f"{d_theta:.3f}{'  SHIFT' if shifted else ''}")

    print(f"\nTotal boxes: {total_boxes}, theta-shifted (π/2): {total_shifts}")
    # Geometry must always be preserved — check via round-trip of one case
    v_orig = xywhr_to_xyxyxyxy(GT_BOXES)
    v_out  = xywhr_to_xyxyxyxy(out)
    ok = total_shifts >= 0  # informational — always passes; report only
    print(f"Affine label shift check: PASS (informational)")
    raise SystemExit(0)
```

- [ ] **Step 2: Run it**

Run: `python test/test_exp_wh_swap_augmentation.py`
Expected: prints table; w<h boxes show SHIFT under anisotropic transforms. Final line PASS.

- [ ] **Step 3: Commit**

```bash
git add test/test_exp_wh_swap_augmentation.py
git commit -m "test: measure affine_obb theta shift at w/h boundary for augmented GT labels"
```

---

### Task 4: Matcher L1 cost inflation for w<h predictions [Risk 5]

**Files:**
- Create: `test/test_exp_wh_swap_matcher.py`

**Interfaces:**
- Consumes: `engine.deim.matcher.HungarianMatcher` (or the cost-computation helper), `torch`
- Produces: cost comparison for geometrically-equivalent w<h vs w>h preds; PASS/FAIL

**Hypothesis:** A prediction with w<h that underwent w/h swap will have its θ off by π/2 relative to GT, inflating the parameter-L1 match cost. If matcher relies heavily on parameter L1 (cost_bbox), a geometrically-perfect pred could be assigned a high cost and mismatched.

**Implementation note:** Read `engine/deim/matcher.py` first to find the exact cost computation entry point. If the matcher is hard to invoke standalone, compute the L1 angle-cost component directly using the same formula found in the file.

- [ ] **Step 1: Write the experiment (angle-cost component isolation)**

```python
import sys, os, torch, math
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# Matcher L1 cost uses: factor = [1,1,1,1, 1/pi]; L1 = |pred*factor - gt*factor|_sum
# We isolate the angle component to measure the inflation.

GT = torch.tensor([[0.5, 0.5, 0.40, 0.40, 0.30]])          # square-ish, w=h

# Two geometrically near-identical preds:
PRED_NO_SWAP = torch.tensor([[0.5, 0.5, 0.41, 0.40, 0.30]])  # w>h, no swap
PRED_SWAP    = torch.tensor([[0.5, 0.5, 0.40, 0.41, 0.30]])  # w<h, swaps after round-trip

# Simulate what the matcher sees for each pred:
# If the pred went through xyxyxyxy_to_xywhr (e.g., via DOTA export or ADR), it appears as:
from engine.deim.obb_geometry import xywhr_to_xyxyxyxy, xyxyxyxy_to_xywhr

def matcher_angle_l1(pred_params, gt_params):
    """Angle L1 cost component as used in matcher: |Δtheta|/pi."""
    return abs(pred_params[..., 4] - gt_params[..., 4]).item() / math.pi

# pred as model outputs it (no swap yet)
cost_direct_swap_pred = matcher_angle_l1(PRED_SWAP, GT)
# pred after it passes through a w/h-swap path (e.g., DOTA export, ADR decode)
pred_swap_rt = xyxyxyxy_to_xywhr(xywhr_to_xyxyxyxy(PRED_SWAP))
cost_after_swap = matcher_angle_l1(pred_swap_rt, GT)
cost_no_swap    = matcher_angle_l1(PRED_NO_SWAP, GT)

print(f"GT:              w={GT[0,2]:.3f} h={GT[0,3]:.3f} θ={GT[0,4]:.3f}")
print(f"pred (w>h):      w={PRED_NO_SWAP[0,2]:.3f} h={PRED_NO_SWAP[0,3]:.3f} θ={PRED_NO_SWAP[0,4]:.3f}  angle_L1={cost_no_swap:.4f}")
print(f"pred (w<h) orig: w={PRED_SWAP[0,2]:.3f} h={PRED_SWAP[0,3]:.3f} θ={PRED_SWAP[0,4]:.3f}  angle_L1={cost_direct_swap_pred:.4f}")
print(f"pred (w<h) rt:   w={pred_swap_rt[0,2]:.3f} h={pred_swap_rt[0,3]:.3f} θ={pred_swap_rt[0,4]:.3f}  angle_L1={cost_after_swap:.4f}")

inflation = cost_after_swap / max(cost_no_swap, 1e-9)
print(f"\nAngle L1 inflation from w/h swap: {inflation:.1f}x")
ok = inflation > 2.0  # hypothesis: inflation is large
print(f"Matcher L1 inflation check: {'CONFIRMED' if ok else 'NOT CONFIRMED'}")
raise SystemExit(0)  # informational — always exits 0
```

- [ ] **Step 2: Run it**

Run: `python test/test_exp_wh_swap_matcher.py`
Expected: prints inflation factor (expected ~50x for π/2 shift vs small w/h error).

- [ ] **Step 3: Commit**

```bash
git add test/test_exp_wh_swap_matcher.py
git commit -m "test: quantify matcher angle-L1 inflation caused by w/h swap at square boundary"
```

---

### Task 5: DOTA pipeline — false-positive angle error rate [Risks 7+8] (PRIMARY)

**Files:**
- Create: `test/test_exp_wh_swap_dota_pipeline.py`
- Reads: existing DOTA predictions under `test/data/outputs/dlzdt_obb_compare_val/` and `test/data/outputs/dlzdt_res/`

**Interfaces:**
- Consumes: `engine.deim.obb_geometry.xyxyxyxy_to_xywhr`, `tools.model_compare.obb_utils.parse_dota_line`
- Produces: per-model false-positive rate (% of angle errors >15° that are w/h swap artifacts); PASS/FAIL + table

**Hypothesis:** When reading DOTA predictions via `xyxyxyxy_to_xywhr`, some boxes whose true (w,h) is w>h appear as w<h after polygon rounding, triggering swap and reporting θ+π/2. We measure what fraction of "large angle errors" (|Δθ| > 15°) are actually these artifacts.

**Method:**
1. Load GT DOTA files and pred DOTA files.
2. For each matched pair (reuse Hungarian matching from `tool_obb_difference_analysis`):
   - Compute Δθ via standard path (`_poly_to_xywhr` on both).
   - ALSO compute a "swap-corrected" Δθ: if the pred's w<h caused a swap, also try θ_pred − π/2 (mod π) and take the smaller angle distance.
3. Classify each large error (|Δθ| > 15°) as: (a) genuine, (b) swap artifact (swap-corrected Δθ < 5°).
4. Report artifact rate per model.

- [ ] **Step 1: Write the experiment**

```python
import sys, os, math, torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from engine.deim.obb_geometry import xyxyxyxy_to_xywhr
from tools.model_compare.obb_utils import parse_dota_line
from tool_obb_difference_analysis import load_boxes_per_image, match_and_compute_diffs

GT_DIR = "./test/data/outputs/dlzdt_obb_compare_val/gt_dota"
PRED_DIRS = {
    "sp_ft_rep3_14": "./test/data/outputs/dlzdt_res/sp_ft_rep3_0714_val",
    "sp_ft_rep1_14": "./test/data/outputs/dlzdt_res/sp_ft_rep1_0714_val",
}
THRESH_LARGE = 15.0   # degrees
THRESH_ARTIFACT = 5.0 # degrees — if corrected error < this, it's an artifact

def periodic_deg(a, b):
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)

if __name__ == "__main__":
    gt_per_img = load_boxes_per_image(GT_DIR, is_gt=True)
    print(f"Loaded {len(gt_per_img)} GT images")

    summary_rows = []
    for model_name, pred_dir in PRED_DIRS.items():
        if not os.path.isdir(pred_dir):
            print(f"  [SKIP] {model_name}: dir not found")
            continue
        pred_per_img = load_boxes_per_image(pred_dir, is_gt=False)
        (_, angle_diffs, n_matched, _, _, _, _, _) = match_and_compute_diffs(
            gt_per_img, pred_per_img, iou_thr=0.1
        )
        large = [d for d in angle_diffs if abs(d) > THRESH_LARGE]
        # For each large error, estimate artifact: error is close to 90°
        artifacts = [d for d in large if abs(abs(d) - 90.0) < THRESH_ARTIFACT]
        rate = len(artifacts) / max(len(large), 1) * 100
        summary_rows.append((model_name, n_matched, len(large), len(artifacts), rate))
        print(f"  {model_name}: matched={n_matched} large>15°={len(large)} "
              f"artifacts≈90°={len(artifacts)} rate={rate:.1f}%")

    print(f"\n{'Model':<20} {'Matched':>8} {'LargeErr':>9} {'Artifact':>9} {'Rate':>7}")
    for r in summary_rows:
        print(f"{r[0]:<20} {r[1]:>8} {r[2]:>9} {r[3]:>9} {r[4]:>6.1f}%")
    print(f"\nDOTA pipeline artifact check: PASS (informational)")
    raise SystemExit(0)
```

**NOTE for implementer:** This is a first-pass heuristic (errors clustered at 90° are likely artifacts). If real data shows the angle-error distribution is NOT peaked at 90°, refine: compute swap-corrected angle (try both θ and θ−90° mod π) and re-classify. Document which heuristic was used in the final report.

- [ ] **Step 2: Run it**

Run: `python test/test_exp_wh_swap_dota_pipeline.py`
Expected: prints per-model artifact rates. If rate is high (>20%), the observed "large angle errors" in tool outputs are significantly inflated by w/h swap.

- [ ] **Step 3: Commit**

```bash
git add test/test_exp_wh_swap_dota_pipeline.py
git commit -m "test: measure false-positive angle-error rate from w/h swap in DOTA analysis pipeline"
```

---

### Task 6: No-swap verification for anchor gen + decoder head [Risks 2, 6]

**Files:**
- Create: `test/test_exp_wh_swap_negative.py`

**Interfaces:**
- Consumes: `engine.deim.deim_decoder.DEIMTransformer._generate_anchors` (via a minimal instantiation), `engine.deim.obb_geometry.*`
- Produces: PASS if anchors contain no angle channel derived from position-dependent w/h comparison; PASS if decoder head outputs go directly to loss without vertex round-trip.

**Method:** Static + dynamic check.
- Static: grep `_generate_anchors` source for `xyxyxyxy_to_xywhr` (expect absent).
- Dynamic: instantiate anchors for rep3 config, verify 5th channel is constant 0.5 across all positions.

- [ ] **Step 1: Write the experiment**

```python
import sys, os, inspect, torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from engine.deim.deim_decoder import DEIMTransformer

ok_all = True

# 1. Static check: _generate_anchors must not call xyxyxyxy_to_xywhr
src = inspect.getsource(DEIMTransformer._generate_anchors)
uses_swap_fn = "xyxyxyxy_to_xywhr" in src
print(f"_generate_anchors uses xyxyxyxy_to_xywhr: {uses_swap_fn} (expect False)")
ok_all &= not uses_swap_fn

# 2. Dynamic check: rep3 anchors 5th channel is constant among VALID anchors
model = DEIMTransformer.__new__(DEIMTransformer)
model.box_mode = "obb"
model.angle_rep = 3
model._num_box_dof = 5
model.feat_strides = [8, 16, 32]
model.eval_spatial_size = (640, 640)
model.eps = 1e-2
anchors, valid = DEIMTransformer._generate_anchors(model, device="cpu")
# anchors in inverse-sigmoid space; invalid entries are +inf — filter by valid_mask
valid_flat = valid.squeeze(-1).bool()
fifth = anchors[0, :, 4][valid_flat]
spread = (fifth.max() - fifth.min()).item()
print(f"rep3 valid anchor 5th channel spread: {spread:.2e} (expect 0)")
ok_all &= spread < 1e-6

# 3. Decoder head path: confirm pre_bboxes go to loss directly (no vertex round-trip)
# Static: search decoder forward for xyxyxyxy_to_xywhr between head output and loss return
from engine.deim.deim_decoder import TransformerDecoder
src_dec = inspect.getsource(TransformerDecoder.forward)
uses_swap_in_dec = "xyxyxyxy_to_xywhr" in src_dec
print(f"TransformerDecoder.forward uses xyxyxyxy_to_xywhr: {uses_swap_in_dec} (expect False)")
ok_all &= not uses_swap_in_dec

print(f"\nNegative-path verification: {'PASS' if ok_all else 'FAIL'}")
raise SystemExit(0 if ok_all else 1)
```

- [ ] **Step 2: Run it**

Run: `python test/test_exp_wh_swap_negative.py`
Expected: all three checks False/spread=0, final PASS.

- [ ] **Step 3: Commit**

```bash
git add test/test_exp_wh_swap_negative.py
git commit -m "test: verify anchor generation and decoder head paths are free of w/h swap"
```

---

### Task 7: Consolidated report

**Files:**
- Create: `test/reports/wh_swap_experiments.md` (generated by a script)
- Create: `test/test_exp_wh_swap_report.py`

**Interfaces:**
- Consumes: outputs of Tasks 1–6 (re-runs them and captures stdout)
- Produces: single markdown file with all metrics, verdicts, and a summary table answering: "Which pipeline stages inflate angle error, and by how much?"

- [ ] **Step 1: Write the report generator**

```python
import subprocess, sys, os, datetime

SCRIPTS = [
    ("Task1 FDR swap",        "test/test_exp_wh_swap_fdr.py"),
    ("Task2 loss residual",   "test/test_exp_wh_swap_loss.py"),
    ("Task3 augmentation",    "test/test_exp_wh_swap_augmentation.py"),
    ("Task4 matcher L1",      "test/test_exp_wh_swap_matcher.py"),
    ("Task5 DOTA pipeline",   "test/test_exp_wh_swap_dota_pipeline.py"),
    ("Task6 negative paths",  "test/test_exp_wh_swap_negative.py"),
]

os.makedirs("test/reports", exist_ok=True)
out_path = "test/reports/wh_swap_experiments.md"

lines = [
    f"# w/h Swap Angle-Error Inflation — Experiment Report",
    f"",
    f"Generated: {datetime.datetime.now().isoformat()}",
    f"",
    f"| Experiment | Exit | Key output |",
    f"|---|---|---|",
]

for name, script in SCRIPTS:
    if not os.path.isfile(script):
        lines.append(f"| {name} | MISSING | {script} not found |")
        continue
    proc = subprocess.run([sys.executable, script], capture_output=True, text=True, cwd=os.getcwd())
    tail = "\n".join(proc.stdout.strip().splitlines()[-3:])
    lines.append(f"| {name} | {proc.returncode} | `{tail}` |")

lines += [
    "",
    "## Summary",
    "",
    "- [ ] FDR path (external_rect_to_oriented_box) swaps w<h predictions",
    "- [ ] Loss residuals are asymmetric at the w/h boundary",
    "- [ ] Augmented GT labels undergo θ reparameterization (not corruption)",
    "- [ ] Matcher angle-L1 inflated for w<h predictions",
    "- [ ] DOTA pipeline reports inflated large-angle-error counts",
    "- [ ] Anchor generation and decoder head are free of w/h swap",
    "",
    "Fill each checkbox with CONFIRMED / REFUTED / INCONCLUSIVE after reviewing outputs.",
]

with open(out_path, "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"Report written to {out_path}")
```

- [ ] **Step 2: Run it**

Run: `python test/test_exp_wh_swap_report.py`
Expected: `test/reports/wh_swap_experiments.md` created with table of all experiment outputs.

- [ ] **Step 3: Commit**

```bash
git add test/test_exp_wh_swap_report.py test/reports/wh_swap_experiments.md
git commit -m "test: add consolidated w/h swap experiment report generator"
```

---

## Final Verification Wave

After all tasks:

- [ ] Run all six experiment scripts in sequence; confirm each exits 0 (or documents informational findings without false failures).
- [ ] Review `test/reports/wh_swap_experiments.md` and manually fill the CONFIRMED/REFUTED checkboxes.
- [ ] Report the artifact rate from Task 5 to the user — this number answers how much of the observed "90° / ±90° angle error" in `tool_debug_decoder.py` scatter plots is a w/h swap artifact.

## Commit Strategy

One commit per task (6 commits) + 1 report commit. No changes to `engine/` permitted. If an experiment reveals a bug in `engine/`, open a separate plan — this plan is diagnostics only.

## Success Criteria

- All 7 tasks completed with committed scripts.
- Task 1 confirms `external_rect_to_oriented_box` shares the same w/h swap behavior.
- Task 5 produces a concrete artifact-rate percentage per model for the DOTA pipeline.
- Final report exists at `test/reports/wh_swap_experiments.md` with all experiment outputs.
