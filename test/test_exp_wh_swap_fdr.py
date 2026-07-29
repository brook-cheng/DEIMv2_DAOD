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
        v_orig = xywhr_to_xyxyxyxy(obb)
        v_recon = xywhr_to_xyxyxyxy(recon)
        d1 = ((v_orig.unsqueeze(-2) - v_recon.unsqueeze(-3)) ** 2).sum(-1).amin(-1)
        d2 = ((v_recon.unsqueeze(-2) - v_orig.unsqueeze(-3)) ** 2).sum(-1).amin(-1)
        v_err = max(d1.max().item(), d2.max().item())
        geom_ok = v_err < 1e-5
        if geom_ok:
            results["geom_ok"] += 1
        wh_swapped = not torch.isclose(recon[0, 2], obb[0, 2], atol=1e-5)
        ang_shift = periodic_diff(recon[0, 4].item(), obb[0, 4].item())
        theta_jumped = ang_shift > 1.0
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
    expect_swapped = sum(1 for _, c in CASES if c[0, 2] < c[0, 3])
    ok = res["geom_ok"] == res["total"] and res["swapped"] == expect_swapped
    print(f"ADR swap behavior: {'PASS' if ok else 'FAIL'} "
          f"(expected {expect_swapped} swaps, got {res['swapped']})")
    raise SystemExit(0 if ok else 1)
