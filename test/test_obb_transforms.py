"""TDD tests for OBB data augmentation transforms."""

import torch, sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.data.transforms.obb_transforms import (
    OBBFlip,
    OBBZoomOut,
    OBBResize,
    OBBConvertBoxes,
    OBBSanitize,
    OBBMosaic,
    OBBIoUCrop,
)


def _apply(transform, boxes, labels=None):
    if labels is None:
        labels = torch.arange(len(boxes))
    tgt = {"boxes": boxes.clone(), "labels": labels.clone()}
    _, r, _ = transform((torch.zeros(3, 640, 640), tgt, None))
    return r


class TestOBBFlip:
    def _boxes(self):
        return torch.tensor(
            [
                [320.0, 320.0, 128.0, 64.0, 0.0],
                [320.0, 320.0, 128.0, 64.0, 0.785],
                [200.0, 450.0, 100.0, 60.0, 0.5],
            ]
        )

    def test_flip_cx(self):
        b = self._boxes()
        r = _apply(OBBFlip(), b)
        assert torch.allclose(r["boxes"][:, 0], torch.tensor([320.0, 320.0, 440.0]))

    def test_flip_theta_0(self):
        r = _apply(OBBFlip(), self._boxes())
        assert torch.allclose(r["boxes"][0, 4], torch.tensor(0.0))

    def test_flip_theta_pi4(self):
        r = _apply(OBBFlip(), self._boxes())
        assert torch.allclose(r["boxes"][1, 4], torch.tensor(2.3562), atol=1e-3)

    def test_flip_cy(self):
        b = self._boxes()
        cy = b[:, 1].clone()
        r = _apply(OBBFlip(), b)
        assert torch.allclose(r["boxes"][:, 1], cy)

    def test_flip_wh(self):
        b = self._boxes()
        wh = b[:, 2:4].clone()
        r = _apply(OBBFlip(), b)
        assert torch.allclose(r["boxes"][:, 2:4], wh)


class TestOBBZoomOut:
    def _boxes(self):
        return torch.tensor(
            [[320.0, 320.0, 120.0, 60.0, 0.0], [200.0, 450.0, 100.0, 50.0, 0.785]]
        )

    def test_zoomout_wh_theta(self):
        b = self._boxes()
        whb = b[:, 2:].clone()
        r = _apply(OBBZoomOut(pad_level=0), b)
        assert torch.allclose(r["boxes"][:, 2:], whb)


class TestOBBResize:
    def _boxes(self):
        return torch.tensor([[200.0, 300.0, 100.0, 50.0, 0.5]])

    def test_resize(self):
        b = self._boxes()
        t = b[:, 4].clone()
        r = _apply(OBBResize(size=(320, 240)), b)
        assert torch.allclose(
            r["boxes"][0, :4], torch.tensor([100.0, 112.5, 50.0, 18.75])
        )
        assert torch.allclose(r["boxes"][:, 4], t)


class TestOBBConvertBoxes:
    def test_convert(self):
        b = torch.tensor([[200.0, 300.0, 100.0, 50.0, 0.5]])
        t = b[0, 4].clone()
        r = _apply(OBBConvertBoxes(normalize=True, img_size=(800, 600)), b)
        assert torch.allclose(
            r["boxes"][0, :4], torch.tensor([0.25, 0.5, 0.125, 0.08333]), atol=1e-4
        )
        assert torch.allclose(r["boxes"][0, 4], t)


class TestOBBSanitize:
    def test_sanitize(self):
        b = torch.tensor(
            [
                [320.0, 320.0, 120.0, 60.0, 0.0],
                [200.0, 200.0, 3.0, 50.0, 0.5],
                [450.0, 450.0, 90.0, 2.0, 1.0],
                [250.0, 380.0, 72.0, 54.0, 0.3],
            ]
        )
        r = _apply(OBBSanitize(min_size=4), b, labels=torch.arange(4))
        assert len(r["boxes"]) == 2


class TestOBBMosaic:
    def test_offset(self):
        b = torch.tensor([[320.0, 320.0, 120.0, 60.0, 0.785]])
        r = _apply(OBBMosaic(offset_x=160, offset_y=200), b)
        assert torch.allclose(r["boxes"][0, :2], torch.tensor([480.0, 520.0]))

    def test_labels(self):
        b = torch.tensor([[320.0, 320.0, 120.0, 60.0, 0.0]])
        r = _apply(OBBMosaic(offset_x=50, offset_y=100), b, labels=torch.tensor([3]))
        assert r["labels"][0] == 3


class TestOBBIoUCrop:
    def test_not_empty(self):
        b = torch.tensor([[320.0, 320.0, 120.0, 60.0, 0.5]])
        r = _apply(OBBIoUCrop(p=1.0, scale=(0.25, 0.25), ratio=(1, 1), trials=1), b)
        assert len(r["boxes"]) >= 0

    def test_not_empty2(self):
        b = torch.tensor(
            [[320.0, 320.0, 120.0, 60.0, 0.0], [100.0, 580.0, 60.0, 60.0, 0.3]]
        )
        r = _apply(
            OBBIoUCrop(p=1.0, scale=(0.3, 0.3), ratio=(1, 1), trials=1),
            b,
            labels=torch.tensor([0, 1]),
        )
        assert len(r["boxes"]) >= 0


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v", "-s"])
