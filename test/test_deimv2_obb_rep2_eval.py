import math
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from engine.deim.deim_decoder import DEIMTransformer


def test_angle_rep2_eval_forward_returns_public_physical_obb() -> None:
    # Given: the supported rep2 decoder configuration and matching eval features.
    torch.manual_seed(0)
    model = DEIMTransformer(
        num_classes=5,
        hidden_dim=32,
        num_queries=4,
        feat_channels=[32, 32],
        feat_strides=[4, 8],
        num_levels=2,
        num_points=2,
        nhead=4,
        num_layers=3,
        dim_feedforward=64,
        dropout=0.0,
        activation="relu",
        num_denoising=0,
        learn_query_content=False,
        eval_spatial_size=(16, 16),
        eval_idx=-1,
        eps=1e-2,
        aux_loss=False,
        cross_attn_method="default",
        query_select_method="default",
        reg_max=4,
        reg_scale=4.0,
        layer_scale=1,
        mlp_act="relu",
        use_gateway=True,
        share_bbox_head=False,
        share_score_head=False,
        box_mode="obb",
        angle_rep=2,
        offset_scale_source="pre",
        use_angle_first=False,
    )
    model.eval()
    features = [torch.randn(1, 32, 4, 4), torch.randn(1, 32, 2, 2)]

    # When: eval forward runs through the supported rep2 path.
    with torch.no_grad():
        outputs = model(features)

    # Then: the public boundary exposes finite 5D OBBs in physical radians.
    pred_boxes = outputs["pred_boxes"]
    assert pred_boxes.shape[-1] == 5
    assert torch.isfinite(pred_boxes).all()
    assert (pred_boxes[..., 4] >= 0).all()
    assert (pred_boxes[..., 4] < math.pi).all()


def test_angle_rep2_eval_shifted_config_returns_public_physical_obb() -> None:
    """rep2 + decoder_angle_encoding=shifted 在 eval 下仍输出公开 5D 物理 OBB。

    与 proportional 配置输出逐位一致（rep2 强制 proportional）。
    """
    torch.manual_seed(0)
    feats = [torch.randn(1, 32, 4, 4), torch.randn(1, 32, 2, 2)]

    def run(encoding):
        torch.manual_seed(0)
        model = DEIMTransformer(
            num_classes=5,
            hidden_dim=32,
            num_queries=4,
            feat_channels=[32, 32],
            feat_strides=[4, 8],
            num_levels=2,
            num_points=2,
            nhead=4,
            num_layers=3,
            dim_feedforward=64,
            dropout=0.0,
            activation="relu",
            num_denoising=0,
            learn_query_content=False,
            eval_spatial_size=(16, 16),
            eval_idx=-1,
            eps=1e-2,
            aux_loss=False,
            cross_attn_method="default",
            query_select_method="default",
            reg_max=4,
            reg_scale=4.0,
            layer_scale=1,
            mlp_act="relu",
            use_gateway=True,
            share_bbox_head=False,
            share_score_head=False,
            box_mode="obb",
            angle_rep=2,
            offset_scale_source="pre",
            use_angle_first=False,
            decoder_angle_encoding=encoding,
        )
        model.eval()
        with torch.no_grad():
            return model(feats)

    out_prop = run("proportional")
    out_shift = run("shifted")
    for key in ["pred_boxes", "pred_logits"]:
        assert torch.equal(out_prop[key], out_shift[key]), (
            f"rep2 eval {key} 逐位一致"
        )
    pred_boxes = out_shift["pred_boxes"]
    assert pred_boxes.shape[-1] == 5
    assert torch.isfinite(pred_boxes).all()
    assert (pred_boxes[..., 4] >= 0).all()
    assert (pred_boxes[..., 4] < math.pi).all()
