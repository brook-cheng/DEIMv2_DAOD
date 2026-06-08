import torch
from .obb_geometry import xywhr_to_xyxyxyxy


def chamfer_cost_obb(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Chamfer distance between two sets of OBBs.

    Args:
        boxes1: (N, 5)  — xywhr in normalized coords.
        boxes2: (M, 5)  — xywhr in normalized coords.

    Returns:
        (N, M) — Chamfer distances.
    """
    verts1 = xywhr_to_xyxyxyxy(boxes1)  # (N, 4, 2)
    verts2 = xywhr_to_xyxyxyxy(boxes2)  # (M, 4, 2)

    diff = verts1.unsqueeze(1).unsqueeze(3) - verts2.unsqueeze(0).unsqueeze(2)
    # verts1: (N, 1, 4, 1, 2)
    # verts2: (1, M, 1, 4, 2)
    # diff:   (N, M, 4, 4, 2)

    dist = (diff**2).sum(dim=-1)  # (N, M, 4, 4)
    forward = dist.min(dim=-1).values.mean(dim=-1)  # (N, M)
    backward = dist.min(dim=-2).values.mean(dim=-1)  # (N, M)
    return forward + backward
