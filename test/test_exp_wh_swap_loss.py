import sys, os, torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from engine.deim.obb_geometry import (
    oriented_box_to_external_rect,
    xywhr_to_xyxyxyxy,
    xyxyxyxy_to_xywhr,
)

GT   = torch.tensor([[0.5, 0.5, 0.40, 0.40, 0.30]])
PRED_GT = torch.tensor([[0.5, 0.5, 0.41, 0.40, 0.30]])
PRED_LT = torch.tensor([[0.5, 0.5, 0.40, 0.41, 0.30]])

def compute_adr_residual(pred, gt):
    ext_p, vo_p = oriented_box_to_external_rect(pred)
    ext_g, vo_g = oriented_box_to_external_rect(gt)
    return (vo_g - vo_p).abs()

if __name__ == "__main__":
    torch.manual_seed(0)
    r_gt = compute_adr_residual(PRED_GT, GT)
    r_lt = compute_adr_residual(PRED_LT, GT)
    adr_diff = (r_gt - r_lt).abs().max().item()
    print(f"rep2/ADR residual: w>h pred (eps,eta)={[f'{x:.6f}' for x in r_gt[0].tolist()]}  "
          f"w<h pred={[f'{x:.6f}' for x in r_lt[0].tolist()]}")
    print(f"rep2/ADR |delta_residual| = {adr_diff:.6f}")

    d_gt = abs(PRED_GT[0, 4] - GT[0, 4]).item()
    d_lt = abs(PRED_LT[0, 4] - GT[0, 4]).item()
    rt_lt = xyxyxyxy_to_xywhr(xywhr_to_xyxyxyxy(PRED_LT))
    d_lt_rt = abs(rt_lt[0, 4] - GT[0, 4]).item()

    print(f"\nrep3 direct theta:   d(w>h)={d_gt:.6f}  d(w<h)={d_lt:.6f}")
    print(f"rep3 after DOTA swap: d(w<h)={d_lt_rt:.6f}  inflated={d_lt_rt/max(d_gt,1e-9):.0f}x")
    print(f"  w<h orig: w={PRED_LT[0,2]:.3f} h={PRED_LT[0,3]:.3f} theta={PRED_LT[0,4]:.3f}")
    print(f"  w<h swpd: w={rt_lt[0,2]:.3f} h={rt_lt[0,3]:.3f} theta={rt_lt[0,4]:.3f}")

    print(f"\nrep2 ADR symmetry: {'CONFIRMED' if adr_diff < 1e-4 else 'ASYMMETRIC'}")
    inflated = d_lt_rt > 1.0
    print(f"rep3 DOTA theta inflation: {'CONFIRMED' if inflated else 'NOT CONFIRMED'}")
    raise SystemExit(0)
