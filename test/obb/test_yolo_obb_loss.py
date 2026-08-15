"""Tests for engine.deim.yolo_obb_loss pure OBB loss helpers (Task 1-2).

Run: python -m pytest test/test_yolo_obb_loss.py -q
"""
import math
import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from engine.deim.yolo_obb_loss import (
    canonical_side_l1_loss,
    compute_angle_cost_matrix,
    yolo_angle_loss,
    yolo_probiou_loss,
)


def _obb(vals, dtype=torch.float32, grad=False):
    """Build a single-row OBB tensor from a list of 5 floats."""
    return torch.tensor([vals], dtype=dtype, requires_grad=grad)


def _valid_obb(n, seed=42):
    """Generate *n* valid OBBs with positive w/h and theta in [0, pi)."""
    torch.manual_seed(seed)
    t = torch.rand(n, 5, dtype=torch.float32)
    t[:, 2:4] = t[:, 2:4] * 0.5 + 0.1
    t[:, 4] = t[:, 4] * math.pi
    return t


# (function, has_normalizer) for parametrized tests
LOSS_FNS = [
    (canonical_side_l1_loss, True),
    (yolo_probiou_loss, True),
    (yolo_angle_loss, True),
    (compute_angle_cost_matrix, False),
]


# --- Identical pairs: L1/angle zero, ProbIoU < 1e-3 ---

@pytest.mark.parametrize("fn, thresh, exact", [
    (canonical_side_l1_loss, 0.0, True),
    (yolo_angle_loss, 1e-6, False),
    (yolo_probiou_loss, 1e-3, False),
], ids=["l1", "angle", "probiou"])
def test_identical_pair(fn, thresh, exact):
    """Given identical OBBs, loss must be zero (exact or near)."""
    pred = _obb([0.5, 0.5, 0.3, 0.2, 0.5])
    loss = fn(pred, pred.clone(), 1.0)
    if exact:
        assert loss.item() == thresh
    else:
        assert torch.isfinite(loss)
        assert loss.item() < thresh


# --- Canonical L1 invariance and gradient ---

def test_canonical_l1_invariance():
    """Given w/h exchange or equivalent (h,w,theta+pi/2), L1 is unchanged."""
    pred = _obb([0.6, 0.55, 0.35, 0.15, 0.4])
    tgt = _obb([0.5, 0.5, 0.3, 0.2, 0.5])
    pred_sw = _obb([0.6, 0.55, 0.15, 0.35, 0.4])
    tgt_eq = _obb([0.5, 0.5, 0.2, 0.3, math.pi / 2])
    base = canonical_side_l1_loss(pred, tgt, 1.0)
    assert torch.isclose(base, canonical_side_l1_loss(pred_sw, tgt, 1.0), atol=1e-6)
    assert torch.isclose(base, canonical_side_l1_loss(pred, tgt_eq, 1.0), atol=1e-6)


def test_canonical_l1_gradient_descent():
    """Given oversized/undersized sides, one step must reduce loss."""
    tgt = _obb([0.5, 0.5, 0.4, 0.4, 0.0])
    pred = _obb([0.5, 0.5, 0.3, 0.5, 0.0], grad=True)
    loss = canonical_side_l1_loss(pred, tgt, 1.0)
    loss.backward()
    with torch.no_grad():
        pred_new = pred - 0.1 * pred.grad
    assert canonical_side_l1_loss(pred_new, tgt, 1.0).item() < loss.item()


def test_canonical_l1_gradient_direction():
    """Given pred right/above target, dL/dcx>0, dL/dcy>0; undersized
    short side dL/dw<0; oversized long side dL/dh>0."""
    tgt = _obb([0.5, 0.5, 0.4, 0.4, 0.0])
    pred = _obb([0.6, 0.55, 0.3, 0.5, 0.0], grad=True)
    canonical_side_l1_loss(pred, tgt, 1.0).backward()
    g = pred.grad[0]
    assert g[0] > 0 and g[1] > 0
    assert g[2] < 0 and g[3] > 0


# --- Angle: delta=pi/2 near zero, delta=pi/4 square near one ---

def test_angle_delta_pi_over_2_near_zero():
    """Given delta=pi/2, angle loss and cost must be near zero."""
    pred = _obb([0.5, 0.5, 0.3, 0.2, math.pi / 2])
    tgt = _obb([0.5, 0.5, 0.3, 0.2, 0.0])
    assert yolo_angle_loss(pred, tgt, 1.0).item() < 1e-6
    cost = compute_angle_cost_matrix(pred, tgt)
    assert cost.shape == (1, 1) and cost.item() < 1e-6


def test_angle_delta_pi_over_4_square_near_one():
    """Given delta=pi/4 and square GT, angle loss must be near one."""
    pred = _obb([0.5, 0.5, 0.3, 0.3, math.pi / 4])
    tgt = _obb([0.5, 0.5, 0.3, 0.3, 0.0])
    assert abs(yolo_angle_loss(pred, tgt, 1.0).item() - 1.0) < 1e-5


def test_angle_extreme_ratio_lower_than_square():
    """Given equal angle errors, extreme-ratio penalty must be lower."""
    d = math.pi / 4
    pe_sq = _obb([0.5, 0.5, 0.3, 0.3, d])
    pe_ex = _obb([0.5, 0.5, 1.0, 0.01, d])
    sq = yolo_angle_loss(pe_sq, _obb([0.5, 0.5, 0.3, 0.3, 0]), 1.0)
    ex = yolo_angle_loss(pe_ex, _obb([0.5, 0.5, 1.0, 0.01, 0]), 1.0)
    assert ex.item() < sq.item()


# --- Angle cost matrix: shape, empty, NaN, dtype/device (f32+f64) ---

def test_angle_cost_matrix_properties():
    """Given valid/empty inputs, cost has correct shape, no NaN, preserves dtype."""
    pred, tgt = _valid_obb(10), _valid_obb(3)
    cost = compute_angle_cost_matrix(pred, tgt)
    assert cost.shape == (10, 3) and torch.isfinite(cost).all()
    assert cost.dtype == torch.float32 and cost.device == pred.device
    p64, t64 = _valid_obb(5).double(), _valid_obb(3).double()
    assert compute_angle_cost_matrix(p64, t64).dtype == torch.float64
    assert compute_angle_cost_matrix(pred, torch.zeros(0, 5)).shape == (10, 0)
    assert compute_angle_cost_matrix(torch.zeros(0, 5), tgt).shape == (0, 3)


# --- Invalid shape: ValueError naming the invalid shape (all helpers) ---

@pytest.mark.parametrize("fn, has_norm", LOSS_FNS, ids=["l1", "probiou", "angle", "cost"])
@pytest.mark.parametrize("which", ["pred", "target"])
@pytest.mark.parametrize("last_dim", [4, 6, 3, 7])
def test_invalid_shape_raises_valueerror(fn, has_norm, which, last_dim):
    """Given final dim != 5, helper must raise ValueError naming the shape."""
    bad = last_dim if which == "pred" else 5
    good = 5 if which == "pred" else last_dim
    pred = torch.zeros(2, bad, dtype=torch.float32)
    target = torch.zeros(2, good, dtype=torch.float32)
    with pytest.raises(ValueError, match=str(last_dim)):
        fn(pred, target, 1.0) if has_norm else fn(pred, target)


# --- Empty matched pairs: scalar zero, same dtype ---

@pytest.mark.parametrize("fn", [canonical_side_l1_loss, yolo_probiou_loss, yolo_angle_loss],
                         ids=["l1", "probiou", "angle"])
def test_empty_returns_scalar_zero(fn):
    """Given empty matched pairs, loss must return scalar zero."""
    pred = torch.zeros(0, 5, dtype=torch.float32)
    loss = fn(pred, pred, 1.0)
    assert loss.shape == () and loss.item() == 0.0
    assert loss.dtype == torch.float32


# --- Gradient preservation: finite, nonzero ---

@pytest.mark.parametrize("fn", [canonical_side_l1_loss, yolo_probiou_loss, yolo_angle_loss],
                         ids=["l1", "probiou", "angle"])
def test_preserves_gradients(fn):
    """Given pred with requires_grad, loss must produce finite nonzero gradients."""
    pred = _obb([0.6, 0.55, 0.35, 0.15, 0.4], grad=True)
    tgt = _obb([0.5, 0.5, 0.3, 0.2, 0.5])
    fn(pred, tgt, 1.0).backward()
    g = pred.grad
    assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0


# --- Normalizer (float/tensor/scaling) and lambda ---

def test_normalizer_and_lambda():
    """Given tensor normalizer, L1 matches float; normalizer=2 halves;
    smaller lambda down-weights extreme ratios more."""
    pred, tgt = _obb([0.6, 0.5, 0.3, 0.2, 0.0]), _obb([0.5, 0.5, 0.3, 0.2, 0.0])
    l2f = canonical_side_l1_loss(pred, tgt, 2.0)
    assert torch.isclose(l2f, canonical_side_l1_loss(pred, tgt, torch.tensor(2.0)), atol=1e-6)
    assert torch.isclose(l2f * 2, canonical_side_l1_loss(pred, tgt, 1.0), atol=1e-6)
    pe = _obb([0.5, 0.5, 1.0, 0.01, math.pi / 4])
    te = _obb([0.5, 0.5, 1.0, 0.01, 0.0])
    la1 = yolo_angle_loss(pe, te, 1.0, 1.0).item()
    la3 = yolo_angle_loss(pe, te, 1.0, 3.0).item()
    assert la1 < la3


# --- Non-mutation (all four helpers) ---

@pytest.mark.parametrize("fn, has_norm", LOSS_FNS, ids=["l1", "probiou", "angle", "cost"])
def test_does_not_mutate_inputs(fn, has_norm):
    """Given valid inputs, helper must not modify them in place."""
    pred = _obb([0.6, 0.55, 0.35, 0.15, 0.4])
    tgt = _obb([0.5, 0.5, 0.3, 0.2, 0.5])
    pb, tb = pred.clone(), tgt.clone()
    _ = fn(pred, tgt, 1.0) if has_norm else fn(pred, tgt)
    assert torch.equal(pred, pb) and torch.equal(tgt, tb)


# --- Stale-state guard ---

def test_imports_on_disk_repo_module():
    """stale_state guard: imported module must come from on-disk repo file."""
    import engine.deim.yolo_obb_loss as mod
    expected = os.path.realpath(os.path.join(ROOT, "engine", "deim", "yolo_obb_loss.py"))
    assert os.path.realpath(mod.__file__) == expected


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
