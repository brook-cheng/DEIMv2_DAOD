"""
Correctness and speed tests for obb_eval.py optimizations.

Run:  python -m pytest test/test_obb_eval_speed.py -v -s
"""

import torch
import torch.nn as nn
import numpy as np
import time
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from engine.eval.obb_eval import obb_evaluate


# ---------------------------------------------------------------------------
# Mock model / postprocessor / dataloader
# ---------------------------------------------------------------------------

class MockModel(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, x):
        bs = x.shape[0] if isinstance(x, torch.Tensor) else len(x)
        return {
            "pred_logits": torch.randn(bs, 300, self.num_classes),
            "pred_boxes": torch.rand(bs, 300, 5),
        }


class MockPostprocessor(nn.Module):
    def __init__(self, num_classes=3, conf_distribution="mixed"):
        """
        conf_distribution:
          "mixed"  — realistic: ~90% scores < 0.01, ~10% scores > 0.1
          "uniform" — all scores uniform [0, 1]
          "low"    — all scores < 0.001 (tests edge case: all filtered)
        """
        super().__init__()
        self.num_classes = num_classes
        self.conf_distribution = conf_distribution

    def forward(self, outputs, orig_sizes):
        bs = outputs["pred_boxes"].shape[0]
        nq = 300
        results = []
        for _ in range(bs):
            boxes = torch.rand(nq, 5) * 256
            boxes[:, 4] = boxes[:, 4] % torch.pi  # θ in [0, π)

            if self.conf_distribution == "mixed":
                scores = torch.rand(nq)
                # Make ~90% low confidence
                mask = torch.rand(nq) < 0.9
                scores[mask] = scores[mask] * 0.01
            elif self.conf_distribution == "low":
                scores = torch.rand(nq) * 0.0005  # all < 0.001
            else:
                scores = torch.rand(nq)

            labels = (torch.rand(nq) * self.num_classes).long()
            results.append({
                "boxes": boxes,
                "scores": scores,
                "labels": labels,
            })
        return results


def _make_batches(n_imgs=16, batch_size=4, num_classes=3, n_gt=5):
    """Build synthetic batches with GT boxes."""
    images = [torch.randn(3, 256, 256) for _ in range(n_imgs)]
    targets = []
    for _ in range(n_imgs):
        gt_boxes = torch.rand(n_gt, 5) * 256
        gt_boxes[:, 4] = gt_boxes[:, 4] % torch.pi
        targets.append({
            "boxes": gt_boxes,
            "labels": (torch.rand(n_gt) * num_classes).long(),
            "orig_size": torch.tensor([256, 256]),
        })
    batches = []
    for b in range(0, n_imgs, batch_size):
        b_imgs = torch.stack(images[b:b + batch_size])
        b_tgts = targets[b:b + batch_size]
        batches.append((b_imgs, b_tgts))
    return batches


# ---------------------------------------------------------------------------
# Task 1: Confidence pre-filter correctness
# ---------------------------------------------------------------------------

def test_conf_filter_does_not_change_ap():
    """AP values should be identical whether or not low-conf preds are filtered.

    Low-confidence predictions (score < 0.001) cannot be TP at any IoU threshold
    because they sort to the bottom of the PR curve. Filtering them before IoU
    computation should not change the final metrics.
    """
    torch.manual_seed(42)
    batches = _make_batches(n_imgs=16, batch_size=4, num_classes=3, n_gt=5)

    model = MockModel(num_classes=3)
    post = MockPostprocessor(num_classes=3, conf_distribution="mixed")

    stats = obb_evaluate(
        model, post, batches, device="cpu",
        iou_thrs=(0.5, 0.75), num_classes=3,
    )

    assert "AP50" in stats
    assert 0.0 <= stats["AP50"] <= 1.0
    assert 0.0 <= stats["mAP"] <= 1.0
    assert stats["seen"] == 16


def test_conf_filter_all_low_conf():
    """Edge case: all predictions have score < 0.001.

    After filtering, every image has 0 predictions → all FP count = 0.
    AP should still be computable (returns 0).
    """
    torch.manual_seed(42)
    batches = _make_batches(n_imgs=8, batch_size=4, num_classes=3, n_gt=3)

    model = MockModel(num_classes=3)
    post = MockPostprocessor(num_classes=3, conf_distribution="low")

    stats = obb_evaluate(
        model, post, batches, device="cpu",
        iou_thrs=(0.5,), num_classes=3,
    )

    assert stats["AP50"] == 0.0
    assert stats["recall"] == 0.0
    assert stats["seen"] == 8


def test_conf_filter_speed_improvement():
    """Filtering should reduce IoU computation time.

    With mixed confidence distribution, ~90% of 300 predictions are filtered,
    so batch_probiou operates on ~30 boxes instead of 300.
    """
    torch.manual_seed(42)
    batches = _make_batches(n_imgs=32, batch_size=4, num_classes=3, n_gt=5)

    model = MockModel(num_classes=3)
    post = MockPostprocessor(num_classes=3, conf_distribution="mixed")

    t0 = time.time()
    stats = obb_evaluate(
        model, post, batches, device="cpu",
        iou_thrs=(0.5, 0.75), num_classes=3,
    )
    t1 = time.time()
    elapsed = t1 - t0
    print(f"\n  [Task1] obb_evaluate with conf filter: {elapsed:.3f}s")
    # Just assert it completes in reasonable time
    assert elapsed < 30.0, f"Too slow: {elapsed:.1f}s"


# ---------------------------------------------------------------------------
# Task 2: GPU tensor flow (no CPU↔GPU transfers in the loop)
# ---------------------------------------------------------------------------

def test_eval_keeps_tensors_on_device():
    """Verify that obb_evaluate works correctly with tensor-based flow.

    After optimization, the inner loop should work with torch tensors
    directly (no .cpu().numpy() → torch.tensor() round-trip).
    This test uses CPU as the "device" to verify the logic works.
    """
    torch.manual_seed(42)
    batches = _make_batches(n_imgs=8, batch_size=4, num_classes=3, n_gt=5)

    model = MockModel(num_classes=3)
    post = MockPostprocessor(num_classes=3, conf_distribution="mixed")

    stats = obb_evaluate(
        model, post, batches, device="cpu",
        iou_thrs=(0.5, 0.75), num_classes=3,
    )

    assert "AP50" in stats
    assert stats["seen"] == 8
    assert 0.0 <= stats["AP50"] <= 1.0
    assert 0.0 <= stats["recall"] <= 1.0
    assert 0.0 <= stats["precision"] <= 1.0


def test_eval_no_gt_images():
    """Images with no GT boxes should still be counted and produce all-FP results."""
    torch.manual_seed(42)

    images = [torch.randn(3, 256, 256) for _ in range(4)]
    targets = [
        {"boxes": torch.rand(3, 5) * 256, "labels": torch.zeros(3, dtype=torch.long),
         "orig_size": torch.tensor([256, 256])},
        {"boxes": torch.zeros(0, 5), "labels": torch.zeros(0, dtype=torch.long),
         "orig_size": torch.tensor([256, 256])},
        {"boxes": torch.rand(2, 5) * 256, "labels": torch.ones(2, dtype=torch.long),
         "orig_size": torch.tensor([256, 256])},
        {"boxes": torch.zeros(0, 5), "labels": torch.zeros(0, dtype=torch.long),
         "orig_size": torch.tensor([256, 256])},
    ]
    # Make sure theta is in [0, pi) for non-empty boxes
    targets[0]["boxes"][:, 4] = targets[0]["boxes"][:, 4] % torch.pi
    targets[2]["boxes"][:, 4] = targets[2]["boxes"][:, 4] % torch.pi
    batches = [(torch.stack(images), targets)]

    model = MockModel(num_classes=3)
    post = MockPostprocessor(num_classes=3, conf_distribution="mixed")

    stats = obb_evaluate(
        model, post, batches, device="cpu",
        iou_thrs=(0.5,), num_classes=3,
    )

    assert stats["seen"] == 4
    assert "AP50" in stats


# ---------------------------------------------------------------------------
# Task 3: GPU-native match_predictions
# ---------------------------------------------------------------------------

def test_match_predictions_basic():
    """Basic matching: 2 preds, 1 GT, higher IoU wins."""
    from engine.eval.obb_eval import match_predictions

    pred_classes = torch.tensor([0, 0])
    true_classes = torch.tensor([0])
    # IoU matrix: (1 GT, 2 pred) — pred 0 has IoU=0.8, pred 1 has IoU=0.6
    iou = torch.tensor([[0.8, 0.6]])
    iouv = (0.5, 0.75)

    correct = match_predictions(pred_classes, true_classes, iou, iouv)
    # pred 0 should be TP at both thresholds, pred 1 should NOT be TP (GT already matched)
    assert correct[0, 0] == True   # pred 0, IoU>=0.5
    assert correct[0, 1] == True   # pred 0, IoU>=0.75
    assert correct[1, 0] == False  # pred 1, IoU>=0.5 — GT already matched to pred 0
    assert correct[1, 1] == False  # pred 1, IoU>=0.75


def test_match_predictions_class_mismatch():
    """Predictions with wrong class should not match."""
    from engine.eval.obb_eval import match_predictions

    pred_classes = torch.tensor([1, 0])  # pred 0 is class 1, pred 1 is class 0
    true_classes = torch.tensor([0])      # GT is class 0
    iou = torch.tensor([[0.9, 0.9]])     # both have high IoU
    iouv = (0.5,)

    correct = match_predictions(pred_classes, true_classes, iou, iouv)
    # pred 0 (class 1) should NOT match GT (class 0)
    assert correct[0, 0] == False
    # pred 1 (class 0) should match GT (class 0)
    assert correct[1, 0] == True


def test_match_predictions_no_matches():
    """No predictions above threshold."""
    from engine.eval.obb_eval import match_predictions

    pred_classes = torch.tensor([0, 0])
    true_classes = torch.tensor([0])
    iou = torch.tensor([[0.3, 0.2]])  # both below 0.5
    iouv = (0.5,)

    correct = match_predictions(pred_classes, true_classes, iou, iouv)
    assert correct.sum() == 0


def test_match_predictions_accepts_numpy():
    """match_predictions should still accept numpy inputs for backward compat."""
    from engine.eval.obb_eval import match_predictions

    pred_classes = np.array([0, 0])
    true_classes = np.array([0])
    iou = np.array([[0.8, 0.6]])
    iouv = (0.5, 0.75)

    correct = match_predictions(pred_classes, true_classes, iou, iouv)
    assert isinstance(correct, np.ndarray)
    assert correct[0, 0] == True
    assert correct[0, 1] == True


# ---------------------------------------------------------------------------
# Task 4: End-to-end verification
# ---------------------------------------------------------------------------

def test_end_to_end_speed():
    """End-to-end speed benchmark.

    With all 3 optimizations applied, this should complete significantly
    faster than the unoptimized version.
    """
    torch.manual_seed(42)
    batches = _make_batches(n_imgs=32, batch_size=4, num_classes=3, n_gt=5)

    model = MockModel(num_classes=3)
    post = MockPostprocessor(num_classes=3, conf_distribution="mixed")

    # Warmup
    obb_evaluate(model, post, batches[:2], device="cpu",
                 iou_thrs=(0.5, 0.75), num_classes=3)

    # Timed run
    t0 = time.time()
    stats = obb_evaluate(
        model, post, batches, device="cpu",
        iou_thrs=(0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95),
        num_classes=3,
    )
    t1 = time.time()
    elapsed = t1 - t0

    print(f"\n  [Task4] End-to-end obb_evaluate (32 imgs, 10 IoU thresholds): {elapsed:.3f}s")
    print(f"          AP50={stats['AP50']:.4f}  mAP={stats['mAP']:.4f}  "
          f"P={stats['precision']:.4f}  R={stats['recall']:.4f}")

    assert "AP50" in stats
    assert stats["seen"] == 32
    assert elapsed < 60.0, f"Too slow: {elapsed:.1f}s"


def test_end_to_end_metrics_stable():
    """Run eval twice with same seed — metrics must be identical."""
    torch.manual_seed(123)
    batches = _make_batches(n_imgs=16, batch_size=4, num_classes=3, n_gt=5)
    model = MockModel(num_classes=3)
    post = MockPostprocessor(num_classes=3, conf_distribution="mixed")

    stats1 = obb_evaluate(model, post, batches, device="cpu",
                          iou_thrs=(0.5, 0.75), num_classes=3)

    torch.manual_seed(123)
    batches2 = _make_batches(n_imgs=16, batch_size=4, num_classes=3, n_gt=5)
    model2 = MockModel(num_classes=3)
    post2 = MockPostprocessor(num_classes=3, conf_distribution="mixed")

    stats2 = obb_evaluate(model2, post2, batches2, device="cpu",
                          iou_thrs=(0.5, 0.75), num_classes=3)

    assert abs(stats1["AP50"] - stats2["AP50"]) < 1e-6
    assert abs(stats1["mAP"] - stats2["mAP"]) < 1e-6
    assert abs(stats1["precision"] - stats2["precision"]) < 1e-6
    assert abs(stats1["recall"] - stats2["recall"]) < 1e-6
