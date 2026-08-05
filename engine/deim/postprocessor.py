"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision

from ..core import register

__all__ = ["PostProcessor"]


def mod(a, b):
    out = a - a // b * b
    return out


@register()
class PostProcessor(nn.Module):
    __share__ = [
        "num_classes",
        "use_focal_loss",
        "num_top_queries",
        "remap_mscoco_category",
    ]

    def __init__(
        self,
        num_classes=80,
        use_focal_loss=True,
        num_top_queries=300,
        remap_mscoco_category=False,
        box_mode="hbb",
    ) -> None:
        super().__init__()
        self.use_focal_loss = use_focal_loss
        self.num_top_queries = num_top_queries
        self.num_classes = int(num_classes)
        self.remap_mscoco_category = remap_mscoco_category
        self.deploy_mode = False
        self.box_mode = box_mode

    def extra_repr(self) -> str:
        return f"use_focal_loss={self.use_focal_loss}, num_classes={self.num_classes}, num_top_queries={self.num_top_queries}"

    # def forward(self, outputs, orig_target_sizes):
    def forward(self, outputs, orig_target_sizes: torch.Tensor):
        logits, boxes = outputs["pred_logits"], outputs["pred_boxes"]
        # orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)

        if self.box_mode == "hbb":
            bbox_pred = torchvision.ops.box_convert(
                boxes, in_fmt="cxcywh", out_fmt="xyxy"
            )
            bbox_pred *= orig_target_sizes.repeat(1, 2).unsqueeze(1)
        elif self.box_mode == "obb":
            # OBB: 保留 cxcywhθ，逐维缩放到像素
            # PostProcessor only sees 5D OBBs (cx, cy, w, h, theta); the
            # (epsilon, eta) vertex offsets are consumed inside the decoder
            # / dfine_utils decode boundary (external_rect_to_oriented_box)
            # and are not available here. Offset-validity guarding is
            # therefore applied at the geometry decode boundary via
            # external_rect_to_oriented_box(clamp_offsets=True), not in the
            # postprocessor (plan Todo 6, MUST DO #5).
            img_w = orig_target_sizes[:, 0:1]
            img_h = orig_target_sizes[:, 1:2]
            factor = torch.cat(
                [img_w, img_h, img_w, img_h, torch.ones_like(img_w)], dim=-1
            ).unsqueeze(1)
            bbox_pred = boxes * factor  # cx×W, cy×H, w×W, h×H, θ 不变(归一化到[0,π))

        if self.use_focal_loss:
            scores = F.sigmoid(logits)
            scores, index = torch.topk(scores.flatten(1), self.num_top_queries, dim=-1)
            # labels = index % self.num_classes
            labels = mod(index, self.num_classes)
            index = index // self.num_classes
            boxes = bbox_pred.gather(
                dim=1, index=index.unsqueeze(-1).repeat(1, 1, bbox_pred.shape[-1])
            )

        else:
            scores = F.softmax(logits)[:, :, :-1]
            scores, labels = scores.max(dim=-1)
            if scores.shape[1] > self.num_top_queries:
                scores, index = torch.topk(scores, self.num_top_queries, dim=-1)
                labels = torch.gather(labels, dim=1, index=index)
                boxes = torch.gather(
                    boxes, dim=1, index=index.unsqueeze(-1).tile(1, 1, boxes.shape[-1])
                )

        if self.deploy_mode:
            return labels, boxes, scores

        if self.remap_mscoco_category:
            from ..data.dataset import mscoco_label2category

            labels = (
                torch.tensor(
                    [mscoco_label2category[int(x.item())] for x in labels.flatten()]
                )
                .to(boxes.device)
                .reshape(labels.shape)
            )

        results = []
        for lab, box, sco in zip(labels, boxes, scores):
            result = dict(labels=lab, boxes=box, scores=sco)
            results.append(result)

        return results

    def deploy(
        self,
    ):
        self.eval()
        self.deploy_mode = True
        return self
