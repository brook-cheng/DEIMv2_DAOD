# Test: xywhr_to_xyxyxyxy and xyxyxyxy_to_xywhr mutual inverse properties
# Validates geometric round-trip with edge cases.
# Geometric equivalence = same vertex set, not necessarily same parameterization.

import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import torch
from engine.deim.obb_geometry import xywhr_to_xyxyxyxy, xyxyxyxy_to_xywhr

torch.manual_seed(42)
TOL = 1e-5
passed, failed = 0, 0


def _vertex_roundtrip_error(orig_v, recon_v):
    """Max bidirectional nearest-neighbour distance between vertex sets."""
    d1 = ((orig_v.unsqueeze(-2) - recon_v.unsqueeze(-3)) ** 2).sum(dim=-1).amin(dim=-1)
    d2 = ((recon_v.unsqueeze(-2) - orig_v.unsqueeze(-3)) ** 2).sum(dim=-1).amin(dim=-1)
    return torch.max(d1.max(dim=-1).values, d2.max(dim=-1).values).max()


def _check(name, xywhr, expect_vertex_match=True, expect_param_match=True, tol=TOL):
    global passed, failed
    v = xywhr_to_xyxyxyxy(xywhr)
    recon = xyxyxyxy_to_xywhr(v)
    v_recon = xywhr_to_xyxyxyxy(recon)

    v_err = _vertex_roundtrip_error(v, v_recon)
    p_err = (recon - xywhr).abs().max().item()

    vertex_ok = v_err < tol
    param_ok = p_err < tol

    if expect_vertex_match and not vertex_ok:
        failed += 1
        print(f"  [FAIL] {name}: vertex_err={v_err:.2e} param_err={p_err:.2e}")
        print(f"         orig: {xywhr[0].tolist()}")
        print(f"         recon:{recon[0].tolist()}")
    elif expect_param_match and not param_ok:
        failed += 1
        print(f"  [FAIL] {name}: param_err={p_err:.2e} (vertex_err={v_err:.2e})")
        print(f"         orig: {xywhr[0].tolist()}")
        print(f"         recon:{recon[0].tolist()}")
    else:
        passed += 1
        status = "vertex+param" if (vertex_ok and param_ok) else "vertex-only"
        print(f"  [PASS] {name}: {status}  v_err={v_err:.2e} p_err={p_err:.2e}")


print("=" * 60)
print("Round-trip: xywhr -> vertices -> xywhr")
print("=" * 60)

# ── Canonical cases (w > h): parameters must match exactly ──
print("\n-- w > h (canonical, exact round-trip) --")
_check("w=0.4 h=0.2 θ=0",      torch.tensor([[0.5, 0.5, 0.4, 0.2, 0.0]]))
_check("w=0.4 h=0.2 θ=π/6",    torch.tensor([[0.5, 0.5, 0.4, 0.2, 0.523599]]))
_check("w=0.4 h=0.2 θ=π/4",    torch.tensor([[0.5, 0.5, 0.4, 0.2, 0.785398]]))
_check("w=0.4 h=0.2 θ=π/2",    torch.tensor([[0.5, 0.5, 0.4, 0.2, 1.570796]]))
_check("w=0.4 h=0.2 θ=2π/3",   torch.tensor([[0.5, 0.5, 0.4, 0.2, 2.094395]]))
_check("w=0.4 h=0.2 θ=π-0.01", torch.tensor([[0.5, 0.5, 0.4, 0.2, 3.131593]]),
       expect_param_match=False)

# ── Swapped cases (w < h): geometric match, parameter swap ──
print("\n-- w < h (vertex match expected, w/h swap + θ+π/2) --")
_check("w=0.2 h=0.4 θ=0",      torch.tensor([[0.5, 0.5, 0.2, 0.4, 0.0]]),
       expect_param_match=False)
_check("w=0.2 h=0.4 θ=π/6",    torch.tensor([[0.5, 0.5, 0.2, 0.4, 0.523599]]),
       expect_param_match=False)

# ── Square boxes: θ may differ by π/2 ──
print("\n-- square (vertex match, θ may differ by π/2) --")
_check("square w=h=0.3 θ=0",     torch.tensor([[0.5, 0.5, 0.3, 0.3, 0.0]]),
       expect_param_match=False)
_check("square w=h=0.3 θ=π/4",   torch.tensor([[0.5, 0.5, 0.3, 0.3, 0.785398]]),
       expect_param_match=False)

# ── Edge cases ──
print("\n-- extreme aspect ratios --")
_check("thin w=0.8 h=0.02 θ=0",  torch.tensor([[0.5, 0.5, 0.8, 0.02, 0.0]]))
_check("wide w=0.02 h=0.8 θ=0",  torch.tensor([[0.5, 0.5, 0.02, 0.8, 0.0]]),
       expect_param_match=False)
_check("needle w=0.9 h=0.001",   torch.tensor([[0.5, 0.5, 0.9, 0.001, 0.0]]))

print("\n-- angle boundary values --")
_check("θ=1e-6",                 torch.tensor([[0.5, 0.5, 0.3, 0.1, 1e-6]]))
_check("θ=π-1e-6",               torch.tensor([[0.5, 0.5, 0.3, 0.1, torch.pi - 1e-6]]),
       expect_param_match=False)
_check("θ=π/2-1e-6",             torch.tensor([[0.5, 0.5, 0.3, 0.1, torch.pi / 2 - 1e-6]]))
_check("θ=π/2+1e-6",             torch.tensor([[0.5, 0.5, 0.3, 0.1, torch.pi / 2 + 1e-6]]))

print("\n-- degenerate coordinates --")
_check("center at origin",       torch.tensor([[0.0, 0.0, 0.2, 0.1, 0.785398]]))
_check("center at (1,1)",        torch.tensor([[1.0, 1.0, 0.2, 0.1, 0.785398]]))
_check("tiny box w=h=1e-4",      torch.tensor([[0.5, 0.5, 1e-4, 1e-4, 0.785398]]),
       expect_param_match=False)

# ── Batch round-trip ──
print("\n-- batch (2000 random boxes) --")
N = 2000
obbs = torch.cat([
    torch.rand(N, 1),
    torch.rand(N, 1),
    torch.rand(N, 1) * 0.5,
    torch.rand(N, 1) * 0.5,
    torch.rand(N, 1) * torch.pi,
], dim=-1)
v_batch = xywhr_to_xyxyxyxy(obbs)
recon_batch = xyxyxyxy_to_xywhr(v_batch)
v_recon_batch = xywhr_to_xyxyxyxy(recon_batch)

# All vertices must match geometrically
v_err_batch = _vertex_roundtrip_error(v_batch, v_recon_batch)
if v_err_batch < TOL:
    passed += 1
    print(f"  [PASS] batch 2000: max_vertex_err={v_err_batch:.2e}")
else:
    failed += 1
    print(f"  [FAIL] batch 2000: max_vertex_err={v_err_batch:.2e}")

# Canonical (w>h) subset: spatial params exact, angle periodic
mask_wh = obbs[:, 2] >= obbs[:, 3]
n_wh = mask_wh.sum().item()
if n_wh > 0:
    spatial_err = (recon_batch[mask_wh, :4] - obbs[mask_wh, :4]).abs().max().item()
    ang_diff = (recon_batch[mask_wh, 4] - obbs[mask_wh, 4]).abs()
    ang_diff = torch.minimum(ang_diff % torch.pi, torch.pi - (ang_diff % torch.pi))
    ang_err = ang_diff.max().item()
    p_err = max(spatial_err, ang_err)
    if p_err < 1e-3:
        passed += 1
        print(f"  [PASS] canonical (w>=h) subset {n_wh}: max_err={p_err:.2e}")
    else:
        failed += 1
        print(f"  [FAIL] canonical (w>=h) subset {n_wh}: spatial={spatial_err:.2e} angle={ang_err:.2e}")

print(f"\n{'='*60}")
print("DEIM vs Ultralytics comparison")
print("=" * 60)

# Import Ultralytics implementations (check availability)
try:
    import sys as _sys
    _ult_path = "/home/cx/win_dir/thired/ultralytics_update"
    if _ult_path not in _sys.path:
        _sys.path.insert(0, _ult_path)
    from ultralytics.utils.ops import xywhr2xyxyxyxy as _ult_to_vertices
    from ultralytics.utils.ops import xyxyxyxy2xywhr as _ult_from_vertices

    _HAS_ULT = True
except ImportError:
    _HAS_ULT = False

if _HAS_ULT:

    def _ult_roundtrip(xywhr):
        v = _ult_to_vertices(xywhr)  # (N,4,2)
        # ultralytics xyxyxyxy2xywhr requires (N,8)
        return _ult_from_vertices(v.reshape(-1, 8))

    def _periodic_ang_diff(a, b):
        d = abs(a - b) % torch.pi
        return torch.minimum(d, torch.pi - d)

    comparison_cases = [
        ("w>h θ=0",     torch.tensor([[0.5, 0.5, 0.4, 0.2, 0.0]])),
        ("w>h θ=π/4",   torch.tensor([[0.5, 0.5, 0.4, 0.2, 0.785398]])),
        ("w<h θ=0",     torch.tensor([[0.5, 0.5, 0.2, 0.4, 0.0]])),
        ("w<h θ=π/6",   torch.tensor([[0.5, 0.5, 0.2, 0.4, 0.523599]])),
        ("square θ=0",  torch.tensor([[0.5, 0.5, 0.3, 0.3, 0.0]])),
        ("square θ=π/4",torch.tensor([[0.5, 0.5, 0.3, 0.3, 0.785398]])),
        ("w>>h θ=π/6",  torch.tensor([[0.5, 0.5, 0.4, 0.004, 0.523599]])),
        ("w<<h θ=π/6",  torch.tensor([[0.5, 0.5, 0.004, 0.4, 0.523599]])),
    ]

    print("\n-- Single-box: w/h + θ comparison --")
    for name, obb in comparison_cases:
        deim_v = xywhr_to_xyxyxyxy(obb)
        ult_v = _ult_to_vertices(obb)

        deim_out = xyxyxyxy_to_xywhr(deim_v)
        ult_out = _ult_roundtrip(obb)

        wh_match = (deim_out[0, 2:4] - ult_out[0, 2:4]).abs().max().item() < 1e-4
        ang_match = _periodic_ang_diff(deim_out[0, 4], ult_out[0, 4]) < 1e-4

        if wh_match and ang_match:
            passed += 1
            print(f"  [PASS] {name}: DEIM=({deim_out[0,2]:.3f},{deim_out[0,3]:.3f},{deim_out[0,4]:.3f}rad) "
                  f"ULT=({ult_out[0,2]:.3f},{ult_out[0,3]:.3f},{ult_out[0,4]:.3f}rad)")
        else:
            failed += 1
            print(f"  [FAIL] {name}: wh_match={wh_match}, ang_match={ang_match}")
            print(f"         DEIM: {deim_out[0].tolist()}")
            print(f"         ULT:  {ult_out[0].tolist()}")

    # ── Batch: both should produce identical w/h + θ for all boxes ──
    print("\n-- Batch: 2000 random boxes, DEIM vs Ultralytics consistency --")
    obbs_batch = torch.cat([
        torch.rand(2000, 1), torch.rand(2000, 1),
        torch.rand(2000, 1) * 0.5, torch.rand(2000, 1) * 0.5,
        torch.rand(2000, 1) * torch.pi,
    ], dim=-1)

    deim_v = xywhr_to_xyxyxyxy(obbs_batch)
    ult_v = _ult_to_vertices(obbs_batch)
    deim_out = xyxyxyxy_to_xywhr(deim_v)
    ult_out = _ult_roundtrip(obbs_batch)

    wh_err = (deim_out[:, 2:4] - ult_out[:, 2:4]).abs().max().item()
    ang_raw = abs(deim_out[:, 4] - ult_out[:, 4])
    ang_err = torch.minimum(ang_raw % torch.pi, torch.pi - (ang_raw % torch.pi)).max().item()

    if wh_err < 1e-4 and ang_err < 1e-4:
        passed += 1
        print(f"  [PASS] batch 2000: max_wh_err={wh_err:.2e}  max_angle_err={ang_err:.2e} rad")
    else:
        failed += 1
        print(f"  [FAIL] batch 2000: max_wh_err={wh_err:.2e}  max_angle_err={ang_err:.2e} rad")

    # ── Verify both show the same w<h → w/h+θ+π/2 shift ──
    print("\n-- w<h shift consistency: both produce w/h swap + θ±π/2 --")
    w_lt_h = obbs_batch[:, 2] < obbs_batch[:, 3]
    n_swapped = w_lt_h.sum().item()
    if n_swapped > 0:
        orig_wh = obbs_batch[w_lt_h][:, 2:]
        deim_wh = deim_out[w_lt_h][:, 2:]
        ult_wh = ult_out[w_lt_h][:, 2:]

        # Both should have w >= h after swap
        deim_ok = (deim_wh[:, 0] >= deim_wh[:, 1]).all().item()
        ult_ok = (ult_wh[:, 0] >= ult_wh[:, 1]).all().item()

        # Reconstructed vertices should match (geometric equivalence)
        deim_v_round = xywhr_to_xyxyxyxy(deim_out)
        ult_v_round = _ult_to_vertices(ult_out)
        v_err = _vertex_roundtrip_error(deim_v, deim_v_round)
        v_err_u = _vertex_roundtrip_error(ult_v, ult_v_round)

        if deim_ok and ult_ok and v_err < 1e-4 and v_err_u < 1e-4:
            passed += 1
            print(f"  [PASS] w<h subset {n_swapped}: both w>=h after swap, "
                  f"DEIM v_err={v_err:.2e}, ULT v_err={v_err_u:.2e}")
        else:
            failed += 1
            print(f"  [FAIL] w<h subset: deim_w>=h={deim_ok}, ult_w>=h={ult_ok}, "
                  f"v_err={v_err:.2e}, v_err_u={v_err_u:.2e}")
    else:
        print(f"  [SKIP] no w<h boxes in batch")

else:
    print("  [SKIP] Ultralytics not available (import failed)")

# ── W/H swap robustness: near-boundary cases ──
print(f"\n{'='*60}")
print("W/H swap — parameter vs geometry preservation")
print("=" * 60)
try:
    from engine.deim.obb_ops import batch_probiou as _probiou
    _PROBIOU_OK = True
except ImportError:
    _PROBIOU_OK = False

swap_pairs = [
    ("near-square w< → h (h大0.001)",   torch.tensor([[0.5, 0.5, 0.4, 0.401, 0.3]])),
    ("near-square w< → h (h大0.01)",    torch.tensor([[0.5, 0.5, 0.4, 0.410, 0.3]])),
    ("near-square w< → h (h大0.02)",    torch.tensor([[0.5, 0.5, 0.4, 0.420, 0.3]])),
    ("near-square w> → h (w大0.001)",   torch.tensor([[0.5, 0.5, 0.401, 0.4, 0.3]])),
    ("near-square w> → h (w大0.01)",    torch.tensor([[0.5, 0.5, 0.410, 0.4, 0.3]])),
    ("near-square w> → h (w大0.02)",    torch.tensor([[0.5, 0.5, 0.420, 0.4, 0.3]])),
    ("w<<h (w=0.05 h=0.4)",             torch.tensor([[0.5, 0.5, 0.05, 0.40, 0.3]])),
    ("w<<h (w=0.05 h=0.4) θ=π/6",       torch.tensor([[0.5, 0.5, 0.05, 0.40, 0.523599]])),
    ("w<<h (w=0.2 h=0.4)",              torch.tensor([[0.5, 0.5, 0.20, 0.40, 0.3]])),
    ("w>>h (w=0.4 h=0.05)",             torch.tensor([[0.5, 0.5, 0.40, 0.05, 0.3]])),
    ("w>>h (w=0.4 h=0.05) θ=π/6",       torch.tensor([[0.5, 0.5, 0.40, 0.05, 0.523599]])),
    ("w>>h (w=0.4 h=0.2)",              torch.tensor([[0.5, 0.5, 0.40, 0.20, 0.3]])),
]

for name, xywhr in swap_pairs:
    v = xywhr_to_xyxyxyxy(xywhr)
    out = xyxyxyxy_to_xywhr(v)
    v2 = xywhr_to_xyxyxyxy(out)

    v_err = _vertex_roundtrip_error(v, v2)
    p_err = (out - xywhr).abs().max().item()
    ang_diff_orig = abs(out[0, 4].item() - xywhr[0, 4].item())
    w_lt_h = xywhr[0, 2] < xywhr[0, 3]
    swapped = out[0, 2].item() > out[0, 3].item() + 1e-6 if w_lt_h else False

    # Build status
    parts = [f"v_err={v_err:.2e}  p_err={p_err:.2e}"]
    if w_lt_h:
        parts.append(f"w<h→swapped={'YES' if swapped else 'NO'}")
    if ang_diff_orig > 1e-3 and w_lt_h:
        parts.append(f"Δθ={ang_diff_orig:.4f}")

    status_ok = v_err < TOL
    if w_lt_h:
        status_ok = status_ok and swapped

    if status_ok:
        passed += 1
        print(f"  [PASS] {name}: {', '.join(parts)}")
    else:
        failed += 1
        print(f"  [FAIL] {name}: {', '.join(parts)}")
        print(f"         orig={xywhr[0].tolist()}  →  out={out[0].tolist()}")

# ── ProbIoU preservation across w/h swap ──
if _PROBIOU_OK:
    print("\n-- ProbIoU invariance across w/h swap (loss consistency) --")
    ref = torch.tensor([[0.5, 0.5, 0.4, 0.4, 0.3]])  # GT reference
    invar_cases = [
        ("w<h (0.4, 0.41)",  torch.tensor([[0.5, 0.5, 0.4, 0.41, 0.3]])),
        ("w>h (0.41, 0.4)",  torch.tensor([[0.5, 0.5, 0.41, 0.4, 0.3]])),
        ("w<<h (0.1, 0.8)",  torch.tensor([[0.5, 0.5, 0.1, 0.80, 0.3]])),
        ("w>>h (0.8, 0.1)",  torch.tensor([[0.5, 0.5, 0.80, 0.1, 0.3]])),
    ]
    all_preserved = True
    for label, pred in invar_cases:
        iou_before = _probiou(ref, pred)[0, 0].item()
        v_pred = xywhr_to_xyxyxyxy(pred)
        pred_rt = xyxyxyxy_to_xywhr(v_pred)
        iou_after = _probiou(ref, pred_rt)[0, 0].item()
        delta_iou = abs(iou_before - iou_after)
        ok = delta_iou < 1e-5
        if not ok:
            all_preserved = False
        print(f"  {'OK' if ok else 'FAIL'} {label}: "
              f"ProbIoU={iou_before:.4f}→{iou_after:.4f}  Δ={delta_iou:.2e}  "
              f"orig=({pred[0,2]:.2f},{pred[0,3]:.2f},{pred[0,4]:.2f}) "
              f"rt=({pred_rt[0,2]:.2f},{pred_rt[0,3]:.2f},{pred_rt[0,4]:.2f})")

    if all_preserved:
        passed += 1
        print(f"  [PASS] ProbIoU invariant across all w/h swap cases")
    else:
        failed += 1
        print(f"  [FAIL] ProbIoU NOT invariant")

# ── DEIM vs Ultralytics: w/h swap side-by-side ──
if _HAS_ULT:
    print("\n-- DEIM vs Ultralytics: w/h swap output comparison --")
    print(f"  {'Case':<32} {'DEIM':>28} {'ULT':>28} {'一致?'}")
    all_match = True
    for label, xywhr in swap_pairs:
        v = xywhr_to_xyxyxyxy(xywhr)
        d_out = xyxyxyxy_to_xywhr(v)
        u_out = _ult_roundtrip(xywhr)

        d_str = f"({d_out[0,2]:.4f},{d_out[0,3]:.4f},{d_out[0,4]:.3f})"
        u_str = f"({u_out[0,2]:.4f},{u_out[0,3]:.4f},{u_out[0,4]:.3f})"

        wh_ok = (d_out[0, 2:4] - u_out[0, 2:4]).abs().max().item() < 1e-4
        ang_ok = _periodic_ang_diff(d_out[0, 4], u_out[0, 4]) < 1e-4
        match = wh_ok and ang_ok
        if not match:
            all_match = False
        print(f"  {label:<32} {d_str:>28} {u_str:>28} {'✓' if match else '✗'}")

    if all_match:
        passed += 1
        print(f"  [PASS] DEIM and ULT produce identical w/h swap output")
    else:
        failed += 1
        print(f"  [FAIL] DEIM and ULT diverge on w/h swap")

print(f"\n{'='*60}")
print(f"Passed: {passed},  Failed: {failed}")
if failed:
    raise SystemExit(1)
