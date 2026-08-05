"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
Modules to compute the matching cost and solve the corresponding LSAP.

Copyright (c) 2024 The D-FINE Authors All Rights Reserved.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.optimize import linear_sum_assignment
from typing import Dict
import math

from .box_ops import box_cxcywh_to_xyxy, generalized_box_iou, box_iou, ciou
from .obb_ops import batch_probiou
from .chamfer_cost import chamfer_cost_obb
from .yolo_obb_loss import compute_angle_cost_matrix

from ..core import register
import numpy as np


@register()
class HungarianMatcher(nn.Module):
    __share__ = [
        "use_focal_loss",
    ]

    def __init__(
        self,
        weight_dict,
        use_focal_loss=False,
        alpha=0.25,
        gamma=2.0,
        change_matcher=False,
        iou_order_alpha=1.0,
        matcher_change_epoch=10000,
        box_mode="hbb",
        angle_factor=math.pi,
        lambda_angle=1.0,
        angle_order_alpha=1.0,
    ):
        super().__init__()
        self.cost_class = weight_dict.get("cost_class", 0)
        self.cost_bbox = weight_dict.get("cost_bbox", 0)
        self.cost_giou = weight_dict.get("cost_giou", 0)  # hbb
        self.cost_chamfer = weight_dict.get("cost_chamfer", 0)  # obb
        self.cost_probiou = weight_dict.get("cost_probiou", 0)  # obb
        self.cost_angle = weight_dict.get("cost_angle", 0)
        self.late_cost_bbox = weight_dict.get("late_cost_bbox", 0.0)

        self.change_matcher = change_matcher
        self.iou_order_alpha = iou_order_alpha
        self.angle_order_alpha = angle_order_alpha
        self.matcher_change_epoch = matcher_change_epoch

        if self.change_matcher:
            print(
                f"Using the new matching cost with iou_order_alpha = {iou_order_alpha} at epoch {matcher_change_epoch}"
            )

        self.use_focal_loss = use_focal_loss
        self.alpha = alpha
        self.gamma = gamma
        self.box_mode = box_mode
        # Planned: use this factor to scale the matcher angle cost.
        self.angle_factor = angle_factor
        self.lambda_angle = lambda_angle

        if self.box_mode == "hbb":
            assert self.cost_class != 0 or self.cost_bbox != 0 or self.cost_giou != 0
        else:
            assert (
                self.cost_class != 0
                or self.cost_bbox != 0
                or self.cost_probiou != 0
                or self.cost_angle != 0
                or self.cost_chamfer != 0
            )

    @torch.no_grad()
    def forward(
        self, outputs: Dict[str, torch.Tensor], targets, return_topk=False, epoch=0
    ):
        """Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        if self.use_focal_loss:
            out_prob = F.sigmoid(outputs["pred_logits"].flatten(0, 1))
        else:
            out_prob = (
                outputs["pred_logits"].flatten(0, 1).softmax(-1)
            )  # [batch_size * num_queries, num_classes]

        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        if tgt_bbox.numel() == 0:
            empty_indices = [
                (
                    torch.empty(0, dtype=torch.int64),
                    torch.empty(0, dtype=torch.int64),
                )
                for _ in targets
            ]
            if return_topk:
                return {"indices_o2m": empty_indices}
            return {"indices": empty_indices}

        if self.change_matcher and epoch >= self.matcher_change_epoch:
            # Compute the class_score
            class_score = out_prob[
                :, tgt_ids
            ]  # shape = [batch_size * num_queries, gt num within a batch]

            # # Compute iou
            if self.box_mode == "obb":
                bbox_iou = batch_probiou(
                    out_bbox,
                    tgt_bbox,
                    eps=1e-8,
                )
            elif self.box_mode == "hbb":
                bbox_iou, _ = box_iou(
                    box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox)
                )

            if self.box_mode == "obb":
                pred_center = out_bbox[..., :2]
                tgt_center = tgt_bbox[..., :2]
                pred_sides = torch.sort(out_bbox[..., 2:4], dim=-1).values
                tgt_sides = torch.sort(tgt_bbox[..., 2:4], dim=-1).values
                cost_center = torch.cdist(pred_center, tgt_center, p=1)
                cost_sides = torch.cdist(pred_sides, tgt_sides, p=1)
                cost_bbox = cost_center + cost_sides

                if self.cost_angle != 0:
                    angle_cost = compute_angle_cost_matrix(
                        out_bbox,
                        tgt_bbox,
                        lambda_val=3.0,
                    )
                    angle_quality = (1.0 - angle_cost).clamp(0.0, 1.0)
                else:
                    angle_quality = torch.ones_like(bbox_iou)

                geometry_quality = (
                    class_score
                    * torch.pow(bbox_iou, self.iou_order_alpha)
                    * torch.pow(angle_quality, self.angle_order_alpha)
                )
                C = -geometry_quality + self.late_cost_bbox * cost_bbox
            else:
                # Final cost matrix
                C = (-1) * (class_score * torch.pow(bbox_iou, self.iou_order_alpha))
        else:
            # Compute the classification cost. Contrary to the loss, we don't use the NLL,
            # but approximate it in 1 - proba[target class].
            # The 1 is a constant that doesn't change the matching, it can be ommitted.
            if self.use_focal_loss:
                out_prob = out_prob[:, tgt_ids]
                neg_cost_class = (
                    (1 - self.alpha)
                    * (out_prob**self.gamma)
                    * (-(1 - out_prob + 1e-8).log())
                )
                pos_cost_class = (
                    self.alpha
                    * ((1 - out_prob) ** self.gamma)
                    * (-(out_prob + 1e-8).log())
                )
                cost_class = pos_cost_class - neg_cost_class
            else:
                cost_class = -out_prob[:, tgt_ids]
            if self.box_mode == "obb":
                pred_center = out_bbox[..., :2]
                tgt_center = tgt_bbox[..., :2]
                pred_sides = torch.sort(out_bbox[..., 2:4], dim=-1).values
                tgt_sides = torch.sort(tgt_bbox[..., 2:4], dim=-1).values
                cost_center = torch.cdist(pred_center, tgt_center, p=1)
                cost_sides = torch.cdist(pred_sides, tgt_sides, p=1)
                cost_bbox = cost_center + cost_sides
                cost_probiou = -batch_probiou(out_bbox, tgt_bbox, eps=1e-8)
                C = (
                    self.cost_bbox * cost_bbox
                    + self.cost_class * cost_class
                    + self.cost_probiou * cost_probiou
                )
                if self.cost_angle != 0:
                    C += self.cost_angle * compute_angle_cost_matrix(
                        out_bbox,
                        tgt_bbox,
                        lambda_val=3.0,
                    )
                if self.cost_chamfer > 0:
                    C += self.cost_chamfer * chamfer_cost_obb(out_bbox, tgt_bbox)
            elif self.box_mode == "hbb":
                # Compute the L1 cost between boxes
                cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

                # Compute the giou cost betwen boxes
                cost_giou = -generalized_box_iou(
                    box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox)
                )

                # Final cost matrix 3 * self.cost_bbox + 2 * self.cost_class + self.cost_giou
                C = (
                    self.cost_bbox * cost_bbox
                    + self.cost_class * cost_class
                    + self.cost_giou * cost_giou
                )

        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        C = torch.nan_to_num(C, nan=1.0)
        indices_pre = [
            linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))
        ]
        indices = [
            (
                torch.as_tensor(i, dtype=torch.int64),
                torch.as_tensor(j, dtype=torch.int64),
            )
            for i, j in indices_pre
        ]

        # Compute topk indices
        if return_topk:
            return {
                "indices_o2m": self.get_top_k_matches(
                    C, sizes=sizes, k=return_topk, initial_indices=indices_pre
                )
            }

        return {"indices": indices}  # , 'indices_o2m': C.min(-1)[1]}

    def get_top_k_matches(self, C, sizes, k=1, initial_indices=None):
        indices_list = []
        # C_original = C.clone()
        for i in range(k):
            indices_k = (
                [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
                if i > 0
                else initial_indices
            )
            indices_list.append(
                [
                    (
                        torch.as_tensor(i, dtype=torch.int64),
                        torch.as_tensor(j, dtype=torch.int64),
                    )
                    for i, j in indices_k
                ]
            )
            for c, idx_k in zip(C.split(sizes, -1), indices_k):
                idx_k = np.stack(idx_k)
                c[:, idx_k] = 1e6
        indices_list = [
            (
                torch.cat([indices_list[i][j][0] for i in range(k)], dim=0),
                torch.cat([indices_list[i][j][1] for i in range(k)], dim=0),
            )
            for j in range(len(sizes))
        ]
        # C.copy_(C_original)
        return indices_list
