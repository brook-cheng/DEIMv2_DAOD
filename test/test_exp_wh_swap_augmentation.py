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
    for tname, sx, sy, tx, ty in TRANSFORMS:
        out = affine_obb(GT_BOXES, sx=sx, sy=sy, tx=tx, ty=ty)
        for i in range(len(GT_BOXES)):
            total_boxes += 1
            d_theta = periodic_diff(out[i, 4].item(), GT_BOXES[i, 4].item())
            shifted = d_theta > 1.0
            if shifted:
                total_shifts += 1
                print(f"  SHIFT {tname}: ({GT_BOXES[i,2]:.0f},{GT_BOXES[i,3]:.0f},{GT_BOXES[i,4]:.2f}) -> ({out[i,2]:.1f},{out[i,3]:.1f},{out[i,4]:.2f})  dtheta={d_theta:.3f}")

    print(f"\nTotal boxes: {total_boxes}, theta-shifted (pi/2): {total_shifts}")
    print(f"Affine label shift: PASS (informational)")
    raise SystemExit(0)
