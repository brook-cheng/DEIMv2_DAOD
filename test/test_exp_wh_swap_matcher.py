import sys, os, torch, math
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from engine.deim.obb_geometry import xywhr_to_xyxyxyxy, xyxyxyxy_to_xywhr

GT = torch.tensor([[0.5, 0.5, 0.40, 0.40, 0.30]])
PRED_GT = torch.tensor([[0.5, 0.5, 0.41, 0.40, 0.30]])  # w>h
PRED_LT = torch.tensor([[0.5, 0.5, 0.40, 0.41, 0.30]])  # w<h

def matcher_angle_l1(pred, gt):
    return abs(pred[..., 4] - gt[..., 4]).item() / math.pi

# Direct from decoder (no swap yet)
cost_gt = matcher_angle_l1(PRED_GT, GT)
cost_lt = matcher_angle_l1(PRED_LT, GT)

# After DOTA round-trip (swap happens here)
rt_lt = xyxyxyxy_to_xywhr(xywhr_to_xyxyxyxy(PRED_LT))
cost_lt_rt = matcher_angle_l1(rt_lt, GT)

print(f"GT:            w={GT[0,2]:.3f} h={GT[0,3]:.3f} theta={GT[0,4]:.3f}")
print(f"pred w>h:      w={PRED_GT[0,2]:.3f} h={PRED_GT[0,3]:.3f} theta={PRED_GT[0,4]:.3f}  angle_L1={cost_gt:.6f}")
print(f"pred w<h orig: w={PRED_LT[0,2]:.3f} h={PRED_LT[0,3]:.3f} theta={PRED_LT[0,4]:.3f}  angle_L1={cost_lt:.6f}")
print(f"pred w<h swpd: w={rt_lt[0,2]:.3f} h={rt_lt[0,3]:.3f} theta={rt_lt[0,4]:.3f}  angle_L1={cost_lt_rt:.6f}")
inflated = cost_lt_rt / max(cost_gt, 1e-9)
print(f"\nMatcher angle-L1 inflation from w/h swap: {inflated:.0f}x")
print(f"Matcher L1 inflation: {'CONFIRMED' if inflated > 10 else 'NOT CONFIRMED'}")
raise SystemExit(0)
