"""
Profile obb_evaluate bottlenecks.

Run:  python -m pytest test/test_profile_obb_eval.py -v -s
"""

import time
import numpy as np
import torch
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))


class TestProfileOBBEval:
    """Profiling tests — not assertions, just timing benchmarks."""

    N_CLASSES = 15
    N_IMGS = 32
    DETS_PER_CLS = 10
    GTS_PER_CLS = 3

    # ------------------------------------------------------------------
    # Fixtures
    # ------------------------------------------------------------------
    @staticmethod
    def _make_synthetic_data():
        all_dets = [[] for _ in range(TestProfileOBBEval.N_CLASSES)]
        all_gts  = [[] for _ in range(TestProfileOBBEval.N_CLASSES)]
        for _ in range(TestProfileOBBEval.N_IMGS):
            b = np.random.rand(TestProfileOBBEval.DETS_PER_CLS * TestProfileOBBEval.N_CLASSES, 5) * 640
            s = np.random.rand(TestProfileOBBEval.DETS_PER_CLS * TestProfileOBBEval.N_CLASSES)
            l = (np.random.rand(TestProfileOBBEval.DETS_PER_CLS * TestProfileOBBEval.N_CLASSES) * TestProfileOBBEval.N_CLASSES).astype(int)
            gb = np.random.rand(TestProfileOBBEval.GTS_PER_CLS * TestProfileOBBEval.N_CLASSES, 5) * 640
            gl = (np.random.rand(TestProfileOBBEval.GTS_PER_CLS * TestProfileOBBEval.N_CLASSES) * TestProfileOBBEval.N_CLASSES).astype(int)
            for c in range(TestProfileOBBEval.N_CLASSES):
                mp = l == c
                all_dets[c].append(np.c_[b[mp], s[mp]] if mp.any() else np.zeros((0, 6), dtype=np.float32))
                mg = gl == c
                all_gts[c].append(gb[mg] if mg.any() else np.zeros((0, 5), dtype=np.float32))
        return all_dets, all_gts

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------
    def test_profile_data_collection(self):
        """Stage 1: per-image per-class numpy collection."""
        all_dets, all_gts = self._make_synthetic_data()
        t0 = time.time()
        for _ in range(50):
            self._make_synthetic_data()
        t1 = time.time()
        avg = (t1 - t0) / 50
        print(f"\n  [S1] Data collection ({self.N_IMGS}imgs×{self.N_CLASSES}cls): {avg:.4f}s avg")
        assert avg < 1.0, f"Too slow: {avg:.2f}s"

    def test_profile_vstack(self):
        """Stage 2: np.vstack overhead."""
        all_dets, all_gts = self._make_synthetic_data()
        t0 = time.time()
        for _ in range(500):
            for c in range(self.N_CLASSES):
                np.vstack(all_dets[c])
                np.vstack(all_gts[c])
        t1 = time.time()
        avg = (t1 - t0) / 500
        print(f"\n  [S2] vstack ({self.N_CLASSES}cls): {avg:.4f}s avg")
        assert avg < 0.1, f"Too slow: {avg:.2f}s"

    def test_profile_poly_iou(self):
        """Stage 3: poly_iou per-class timing."""
        from engine.eval.poly_iou import poly_iou

        N_DET, N_GT = 300, 90
        d = torch.rand(N_DET, 5) * 640
        d[:, 4] *= torch.pi
        g = torch.rand(N_GT, 5) * 640
        g[:, 4] *= torch.pi

        t0 = time.time()
        iou = poly_iou(d, g)
        t1 = time.time()
        per_pair_ms = (t1 - t0) / (N_DET * N_GT) * 1000
        print(f"\n  [S3] poly_iou {N_DET}×{N_GT}: {t1-t0:.4f}s ({per_pair_ms:.2f}ms/pair)")
        print(f"       extrapolated (15cls×2iou): {(t1-t0)*30:.1f}s")

    def test_profile_probiou(self):
        """Stage 4: batch_probiou for comparison."""
        from engine.deim.obb_ops import batch_probiou

        N_DET, N_GT = 300, 90
        d = torch.rand(N_DET, 5) * 640
        d[:, 4] *= torch.pi
        g = torch.rand(N_GT, 5) * 640
        g[:, 4] *= torch.pi

        t0 = time.time()
        for _ in range(10):
            iou = batch_probiou(d, g)
        t1 = time.time()
        avg = (t1 - t0) / 10
        print(f"\n  [S4] batch_probiou {N_DET}×{N_GT}: {avg:.4f}s avg")
        print(f"       extrapolated (15cls×2iou): {avg*30:.2f}s")
        print(f"       speedup vs poly_iou: {(17.9/avg):.0f}x")

    def test_profile_total_pipeline_estimate(self):
        """Stage 5: end-to-end estimate."""
        collection = 0.015   # from S1
        vstack     = 0.001   # from S2
        poly_iou   = 18.0    # from S3 (per class)
        probiou    = 0.003   # from S4 (per class, estimated)

        poly_total = collection + vstack + poly_iou * 15 * 2
        prob_total = collection + vstack + probiou * 15 * 2

        print(f"\n  [S5] Estimated total per validation:")
        print(f"       poly_iou:     {poly_total:.1f}s")
        print(f"       batch_probiou: {prob_total:.1f}s")
        print(f"       ratio:         {poly_total/prob_total:.0f}x")

    def test_profile_end_to_end_obb_evaluate(self):
        """Stage 6: end-to-end obb_evaluate with mock model."""
        import torch.nn as nn
        from engine.eval.obb_eval import obb_evaluate

        # Mock model / postprocessor / dataloader
        class MockModel(nn.Module):
            def forward(self, x):
                bs = x.shape[0] if isinstance(x, torch.Tensor) else len(x)
                return {"pred_logits": torch.randn(bs, 300, self.num_classes),
                        "pred_boxes": torch.rand(bs, 300, 5)}

        class MockPostprocessor(nn.Module):
            def __init__(self):
                super().__init__()
                self.num_classes = 15
            def forward(self, outputs, orig_sizes):
                bs = outputs["pred_boxes"].shape[0]
                nq = 300
                return [{"boxes": torch.rand(nq, 5) * 640,
                         "scores": torch.rand(nq),
                         "labels": (torch.rand(nq) * 15).long()}
                        for _ in range(bs)]

        # Build a small synthetic loader
        images = [torch.randn(3, 640, 640) for _ in range(self.N_IMGS)]
        targets = [{"boxes": torch.rand(self.GTS_PER_CLS * self.N_CLASSES, 5) * 640,
                     "labels": (torch.rand(self.GTS_PER_CLS * self.N_CLASSES) * self.N_CLASSES).long(),
                     "orig_size": torch.tensor([640, 640])}
                   for _ in range(self.N_IMGS)]
        # Batch loader: 4 images per batch
        bs = 4
        batches = []
        for b in range(0, self.N_IMGS, bs):
            b_imgs = torch.stack(images[b:b+bs])
            b_tgts = targets[b:b+bs]
            batches.append((b_imgs, b_tgts))

        model = MockModel()
        model.num_classes = self.N_CLASSES
        post = MockPostprocessor()

        t0 = time.time()
        stats = obb_evaluate(model, post, batches, device="cpu",
                             iou_thrs=(0.5,), num_classes=self.N_CLASSES)
        t1 = time.time()
        print(f"\n  [S6] End-to-end obb_evaluate ({self.N_IMGS} imgs): {t1-t0:.3f}s")
        print(f"       AP50={stats['AP50']:.4f}  mAP={stats['mAP']:.4f}  "
              f"P={stats['precision']:.4f}  R={stats['recall']:.4f}")
        assert 'AP50' in stats
        assert stats['AP50'] >= 0.0 and stats['AP50'] <= 1.0


if __name__ == '__main__':
    import pytest
    pytest.main([__file__, '-v', '-s'])
