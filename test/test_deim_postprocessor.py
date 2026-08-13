import math
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from engine.deim.postprocessor import PostProcessor


def test_non_focal_obb_postprocessor_uses_class_dim_and_scaled_boxes():
    # Given: a non-focal PostProcessor in OBB mode with two normalized OBBs
    # and pixel-space image sizes.
    processor = PostProcessor(
        num_classes=2,
        use_focal_loss=False,
        num_top_queries=1,
        remap_mscoco_category=False,
        box_mode="obb",
    )
    logits = torch.tensor(
        [[[4.0, 0.0, -2.0], [0.0, 2.0, -2.0]]], dtype=torch.float32
    )
    pred_boxes = torch.tensor(
        [[[0.5, 0.5, 0.2, 0.2, math.pi / 4],
          [0.25, 0.25, 0.1, 0.1, math.pi / 6]]],
        dtype=torch.float32,
    )
    orig_sizes = torch.tensor([[200.0, 100.0]])

    # When: the postprocessor runs the non-focal top-k path.
    result = processor(
        {"pred_logits": logits, "pred_boxes": pred_boxes}, orig_sizes
    )[0]

    # Then: scores come from the per-query class softmax over the last
    # dimension, and the selected box is pixel-scaled from bbox_pred.
    expected_score = torch.softmax(logits, dim=-1)[0, 0, 0]
    expected_box = torch.tensor([100.0, 50.0, 40.0, 20.0, math.pi / 4])
    assert result["labels"].tolist() == [0]
    assert torch.allclose(result["scores"], expected_score.reshape(1))
    assert torch.allclose(result["boxes"][0], expected_box, atol=1e-6)


def test_obb_postprocessor_scales_with_width_height_factor_order():
    # Given: a non-focal PostProcessor in OBB mode with a single normalized
    # OBB (cx, cy, w, h, theta) and a non-square pixel-space image size.
    processor = PostProcessor(
        num_classes=2,
        use_focal_loss=False,
        num_top_queries=1,
        remap_mscoco_category=False,
        box_mode="obb",
    )
    logits = torch.tensor([[[4.0, 0.0]]], dtype=torch.float32)
    pred_boxes = torch.tensor(
        [[[0.5, 0.5, 0.2, 0.1, math.pi / 4]]], dtype=torch.float32
    )
    orig_sizes = torch.tensor([[1024.0, 576.0]])

    # When: the postprocessor scales the OBB into pixel space.
    result = processor(
        {"pred_logits": logits, "pred_boxes": pred_boxes}, orig_sizes
    )[0]

    # Then: cx is scaled by width and cy by height, w is scaled by width
    # and h by height (not swapped), and theta is left unchanged.
    expected_box = torch.tensor([512.0, 288.0, 204.8, 57.6, math.pi / 4])
    assert result["labels"].tolist() == [0]
    assert torch.allclose(result["boxes"][0], expected_box, atol=1e-5)
