import torch
import math


def xy_wh_r_2_xy_sigma(xywhr):
    """Convert oriented bounding box to 2-D Gaussian distribution.

    Args:
        xywhr (torch.Tensor): rbboxes with shape (N, 5).

    Returns:
        xy (torch.Tensor): center point of 2-D Gaussian distribution
            with shape (N, 2).
        sigma (torch.Tensor): covariance matrix of 2-D Gaussian distribution
            with shape (N, 2, 2).
    """
    _shape = xywhr.shape
    assert _shape[-1] == 5
    xy = xywhr[..., :2]
    wh = xywhr[..., 2:4].clamp(min=1e-7, max=1e7).reshape(-1, 2)
    r = xywhr[..., 4]
    cos_r = torch.cos(r)
    sin_r = torch.sin(r)
    R = torch.stack((cos_r, -sin_r, sin_r, cos_r), dim=-1).reshape(-1, 2, 2)
    S = 0.5 * torch.diag_embed(wh)

    sigma = R.bmm(S.square()).bmm(R.permute(0, 2, 1)).reshape(_shape[:-1] + (2, 2))

    return xy, sigma


def _get_covariance_matrix(
    boxes: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate covariance matrix from oriented bounding boxes.

    Args:
        boxes (torch.Tensor): A tensor of shape (N, 5) representing rotated bounding boxes, with xywhr format.

    Returns:
        (torch.Tensor): Covariance matrices corresponding to original rotated bounding boxes.
    """
    # Gaussian bounding boxes, ignore the center points (the first two columns) because they are not needed here.
    gbbs = torch.cat((boxes[:, 2:4].pow(2) / 12, boxes[:, 4:]), dim=-1)
    a, b, c = gbbs.split(1, dim=-1)
    cos = c.cos()
    sin = c.sin()
    cos2 = cos.pow(2)
    sin2 = sin.pow(2)
    return a * cos2 + b * sin2, a * sin2 + b * cos2, (a - b) * cos * sin


def probiou(
    obb1: torch.Tensor, obb2: torch.Tensor, CIoU: bool = False, eps: float = 1e-7
) -> torch.Tensor:
    """
    Calculate probabilistic IoU between oriented bounding boxes.

    Args:
        obb1 (torch.Tensor): Ground truth OBBs, shape (N, 5), format xywhr.
            x, y, w, and h are all real-valued coordinates, not normalized
            to the range [0, 1]. For example, for an image of size 1024×1024,
            the values of x, y, w, and h all lie within the range [0, 1024].
            The angle r is given in radians and is also a real value,
            not normalized to [0, 1]. For instance, its range is [0, π].
        obb2 (torch.Tensor): Predicted OBBs, shape (N, 5), format xywhr.
            x, y, w, and h are all real-valued coordinates, not normalized
            to the range [0, 1]. For example, for an image of size 1024×1024,
            the values of x, y, w, and h all lie within the range [0, 1024].
            The angle r is given in radians and is also a real value,
            not normalized to [0, 1]. For instance, its range is [0, π].
        CIoU (bool, optional): If True, calculate CIoU.
        eps (float, optional): Small value to avoid division by zero.

    Returns:
        (torch.Tensor): OBB similarities, shape (N, 1).

    Notes:
        OBB format: [center_x, center_y, width, height, rotation_angle].

    References:
        https://arxiv.org/pdf/2106.06072v1.pdf
    """
    x1, y1 = obb1[..., :2].split(1, dim=-1)
    x2, y2 = obb2[..., :2].split(1, dim=-1)
    a1, b1, c1 = _get_covariance_matrix(obb1)
    a2, b2, c2 = _get_covariance_matrix(obb2)

    t1 = (
        ((a1 + a2) * (y1 - y2).pow(2) + (b1 + b2) * (x1 - x2).pow(2))
        / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)
    ) * 0.25
    t2 = (
        ((c1 + c2) * (x2 - x1) * (y1 - y2))
        / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)
    ) * 0.5
    t3 = (
        ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2))
        / (
            4
            * ((a1 * b1 - c1.pow(2)).clamp(min=0) * (a2 * b2 - c2.pow(2)).clamp(min=0)).sqrt()
            + eps
        )
        + eps
    ).log() * 0.5
    bd = (t1 + t2 + t3).clamp(eps, 100.0)
    hd = (1.0 - (-bd).exp() + eps).sqrt()
    iou = 1 - hd
    if CIoU:  # only include the wh aspect ratio part
        w1, h1 = obb1[..., 2:4].split(1, dim=-1)
        w2, h2 = obb2[..., 2:4].split(1, dim=-1)
        v = (4 / math.pi**2) * ((w2 / h2).atan() - (w1 / h1).atan()).pow(2)
        with torch.no_grad():
            alpha = v / (v - iou + (1 + eps))
        return iou - v * alpha  # CIoU
    return iou


def batch_probiou(
    obb1: torch.Tensor, obb2: torch.Tensor, eps: float = 1e-7
) -> torch.Tensor:
    """Pairwise ProbIoU between two sets of OBBs (N×M).

    Args:
        obb1: (N, 5) — xywhr in pixel coords.
        obb2: (M, 5) — xywhr in pixel coords.

    Returns:
        (N, M) — similarity matrix.
    """
    x1, y1 = obb1[..., :2].split(1, dim=-1)                  # (N,1) each
    x2, y2 = (x.squeeze(-1)[None] for x in obb2[..., :2].split(1, dim=-1))  # (1,M) each
    a1, b1, c1 = _get_covariance_matrix(obb1)                  # (N,1) each
    a2, b2, c2 = (x.squeeze(-1)[None] for x in _get_covariance_matrix(obb2))  # (1,M) each

    denom = (a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps
    t1 = ((a1 + a2) * (y1 - y2).pow(2) + (b1 + b2) * (x1 - x2).pow(2)) / denom * 0.25
    t2 = ((c1 + c2) * (x2 - x1) * (y1 - y2)) / denom * 0.5
    t3 = (
        ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2))
        / (
            4
            * ((a1 * b1 - c1.pow(2)).clamp(min=0) * (a2 * b2 - c2.pow(2)).clamp(min=0)).sqrt()
            + eps
        )
        + eps
    ).log() * 0.5
    bd = (t1 + t2 + t3).clamp(eps, 100.0)
    hd = (1.0 - (-bd).exp() + eps).sqrt()
    return 1 - hd


def kld_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    fun: str = 'log1p',
    tau: float = 1.0,
    reduction: str = 'mean',
    eps: float = 1e-7,
) -> torch.Tensor:
    """Kullback-Leibler Divergence loss between OBBs as 2D Gaussians.

    Converts each OBB to a 2D Gaussian via xy_wh_r_2_xy_sigma (cov = w²/4, h²/4),
    then computes::

        KLD = 0.5 * (tr(Σₜ⁻¹ Σₚ) + (μₜ-μₚ)ᵀ Σₜ⁻¹ (μₜ-μₚ) - log(det(Σₚ)/det(Σₜ)) - 2)
        loss = 1 - 1 / (tau + f(KLD))

    Args:
        pred:   (..., 5) — predicted OBBs, pixel coords.
        target: (..., 5) — ground-truth OBBs, pixel coords.
        fun:    post-processing function: 'log1p', 'sqrt', or 'none'.
        tau:    smoothness parameter (>=1).
        reduction: 'mean', 'sum', or 'none'.

    Returns:
        Scalar loss (or per-element if reduction='none').
    """
    mu_p, sigma_p = xy_wh_r_2_xy_sigma(pred)          # (..., 2), (..., 2, 2)
    mu_t, sigma_t = xy_wh_r_2_xy_sigma(target)        # (..., 2), (..., 2, 2)

    # Woodbury / analytical inverse for 2×2
    det_t = sigma_t[..., 0, 0] * sigma_t[..., 1, 1] - sigma_t[..., 0, 1].pow(2).clamp(min=eps)
    inv_t = torch.stack([
        torch.stack([ sigma_t[..., 1, 1], -sigma_t[..., 0, 1]], dim=-1),
        torch.stack([-sigma_t[..., 0, 1],  sigma_t[..., 0, 0]], dim=-1),
    ], dim=-2) / det_t.unsqueeze(-1).unsqueeze(-1)

    # Mahalanobis term: (mu_t - mu_p)^T inv_t (mu_t - mu_p)
    dmu = (mu_t - mu_p).unsqueeze(-1)                  # (..., 2, 1)
    maha = dmu.transpose(-2, -1).matmul(inv_t).matmul(dmu).squeeze(-1).squeeze(-1)  # (...)

    # Trace term: tr(inv_t @ sigma_p)
    trace_term = (inv_t * sigma_p.transpose(-2, -1)).sum(dim=(-2, -1))  # (...)

    # Log-det term
    det_p = sigma_p[..., 0, 0] * sigma_p[..., 1, 1] - sigma_p[..., 0, 1].pow(2).clamp(min=eps)
    logdet_term = (det_p.clamp(min=eps) / det_t.clamp(min=eps)).log()   # (...)

    kld = 0.5 * (trace_term + maha - logdet_term - 2.0).clamp(min=0)  # (...)

    # Post-process
    if fun == 'log1p':
        dist = torch.log1p(kld)
    elif fun == 'sqrt':
        dist = torch.sqrt(kld.clamp(min=eps))
    elif fun == 'none':
        dist = kld
    else:
        raise ValueError(f"Unknown fun: {fun}")

    loss = 1.0 - 1.0 / (tau + dist)

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    return loss


def rbbox_overlaps_obb(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor,
    mode: str = 'probiou',
    eps: float = 1e-7,
) -> torch.Tensor:
    """OBB overlap matrix.

    Args:
        boxes1: (N, 5)  — xywhr in pixel coords.
        boxes2: (M, 5)  — xywhr in pixel coords.
        mode:  'probiou' (default).

    Returns:
        (N, M) — overlap scores in [0, 1].
    """
    if mode == 'probiou':
        return batch_probiou(boxes1, boxes2, eps=eps)
    raise ValueError(f"Unknown mode: {mode}")
