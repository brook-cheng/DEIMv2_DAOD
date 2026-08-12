import torch

from engine.deim.obb_geometry import affine_obb


def rescale_obb_to_original(
    boxes_xywhr: torch.Tensor,
    original_size: tuple[int, int],
    inference_size: tuple[int, int],
) -> torch.Tensor:
    original_h, original_w = original_size
    inference_h, inference_w = inference_size
    return affine_obb(
        boxes_xywhr,
        sx=original_w / inference_w,
        sy=original_h / inference_h,
    )
