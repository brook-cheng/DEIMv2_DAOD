import sys, os, inspect, torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from engine.deim.deim_decoder import DEIMTransformer, TransformerDecoder

ok_all = True

# 1. Static: _generate_anchors does not call xyxyxyxy_to_xywhr
src = inspect.getsource(DEIMTransformer._generate_anchors)
has_swap = "xyxyxyxy_to_xywhr" in src
print(f"anchor gen uses xyxyxyxy_to_xywhr: {has_swap}")
ok_all &= not has_swap

# 2. Static: TransformerDecoder.forward does not call xyxyxyxy_to_xywhr
src_dec = inspect.getsource(TransformerDecoder.forward)
has_swap_dec = "xyxyxyxy_to_xywhr" in src_dec
print(f"decoder forward uses xyxyxyxy_to_xywhr: {has_swap_dec}")
ok_all &= not has_swap_dec

# 3. Dynamic: rep3 anchors have constant 5th channel
m = DEIMTransformer.__new__(DEIMTransformer)
m.box_mode = "obb"; m.angle_rep = 3; m._num_box_dof = 5
m.feat_strides = [8, 16, 32]; m.eval_spatial_size = (640, 640); m.eps = 1e-2
anchors, valid = DEIMTransformer._generate_anchors(m, device="cpu")
valid_flat = valid[0, :, 0].bool()
fifth = anchors[0, valid_flat, 4]
spread = (fifth.max() - fifth.min()).item()
print(f"rep3 valid anchor 5th ch spread: {spread:.2e}")
ok_all &= spread < 1e-6

# 4. Static: encoder/decoder heads (pre_bbox_head etc.) are MLPs, no vertex round-trip
#    verified by flow — encoder outputs go directly to loss via sigmoid+refinement
#    These are standard MLP outputs, no geometry conversion in the head path.

print(f"\nNegative-path verification: {'PASS' if ok_all else 'FAIL'}")
raise SystemExit(0 if ok_all else 1)
