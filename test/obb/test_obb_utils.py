"""Focused tests for tools/model_compare/obb_utils.py visualization helpers."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from PIL import Image
import torch

from engine.deim.obb_geometry import affine_obb
from engine.deim.obb_ops import probiou
from tools.model_compare.obb_inference_geometry import rescale_obb_to_original
from tools.model_compare.obb_utils import visualize_dota_predictions


def _write_dota(path, lines):
    with open(path, "w") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))


def test_visualize_draws_image_with_matching_dota_file(tmp_path):
    img_dir = tmp_path / "imgs"
    dota_dir = tmp_path / "dota"
    vis_dir = tmp_path / "vis"
    img_dir.mkdir()
    dota_dir.mkdir()

    Image.new("RGB", (100, 80), (255, 255, 255)).save(img_dir / "a.png")
    _write_dota(
        dota_dir / "a.txt",
        ["10 10 30 10 30 40 10 40 cable 0.90"],
    )

    written = visualize_dota_predictions(
        str(img_dir), str(dota_dir), str(vis_dir), score_threshold=0.0
    )

    assert written == 1
    out = vis_dir / "a.png"
    assert out.exists()
    drawn = Image.open(out)
    assert drawn.size == (100, 80)
    assert drawn.getextrema() != ((255, 255), (255, 255), (255, 255))


def test_visualize_score_threshold_skips_low_confidence(tmp_path):
    img_dir = tmp_path / "imgs"
    dota_dir = tmp_path / "dota"
    vis_dir = tmp_path / "vis"
    img_dir.mkdir()
    dota_dir.mkdir()

    Image.new("RGB", (100, 80), (255, 255, 255)).save(img_dir / "a.png")
    _write_dota(
        dota_dir / "a.txt",
        ["10 10 30 10 30 40 10 40 cable 0.10"],
    )

    written = visualize_dota_predictions(
        str(img_dir), str(dota_dir), str(vis_dir), score_threshold=0.5
    )

    assert written == 0
    assert not (vis_dir / "a.png").exists()


def test_visualize_skips_image_without_dota_file(tmp_path):
    img_dir = tmp_path / "imgs"
    dota_dir = tmp_path / "dota"
    vis_dir = tmp_path / "vis"
    img_dir.mkdir()
    dota_dir.mkdir()

    Image.new("RGB", (100, 80), (255, 255, 255)).save(img_dir / "a.png")
    Image.new("RGB", (100, 80), (255, 255, 255)).save(img_dir / "b.png")
    _write_dota(dota_dir / "a.txt", ["10 10 30 10 30 40 10 40 cable 0.90"])

    written = visualize_dota_predictions(
        str(img_dir), str(dota_dir), str(vis_dir), score_threshold=0.0
    )

    assert written == 1
    assert (vis_dir / "a.png").exists()
    assert not (vis_dir / "b.png").exists()


def test_rescale_obb_to_original_refits_theta_for_non_uniform_scale():
    original_box = torch.tensor([[2000.0, 1500.0, 800.0, 400.0, 0.35]])
    inference_box = affine_obb(original_box, sx=640 / 4000, sy=640 / 3000)
    expected_box = affine_obb(inference_box, sx=4000 / 640, sy=3000 / 640)

    restored_box = rescale_obb_to_original(
        inference_box,
        original_size=(3000, 4000),
        inference_size=(640, 640),
    )

    assert torch.allclose(restored_box, expected_box, atol=1e-4, rtol=1e-5)
    assert not torch.isclose(restored_box[0, 4], inference_box[0, 4])


def test_rescale_obb_to_original_preserves_theta_for_uniform_scale():
    inference_box = torch.tensor([[320.0, 240.0, 160.0, 80.0, 0.7]])

    restored_box = rescale_obb_to_original(
        inference_box,
        original_size=(1280, 1280),
        inference_size=(640, 640),
    )

    assert torch.isclose(restored_box[0, 4], inference_box[0, 4], atol=1e-6)
    assert torch.allclose(
        restored_box[0, :4],
        inference_box[0, :4] * 2,
        atol=1e-4,
        rtol=1e-5,
    )


def test_rescale_obb_to_original_preserves_empty_input():
    boxes = torch.empty((0, 5), dtype=torch.float64)

    restored_boxes = rescale_obb_to_original(
        boxes,
        original_size=(3000, 4000),
        inference_size=(640, 640),
    )

    assert restored_boxes is boxes
    assert restored_boxes.shape == (0, 5)
    assert restored_boxes.dtype == torch.float64


def test_rescale_obb_to_original_keeps_perfect_prediction_probiou_near_one():
    original_box = torch.tensor([[2000.0, 1500.0, 800.0, 400.0, 0.35]])
    inference_box = affine_obb(original_box, sx=640 / 4000, sy=640 / 3000)
    expected_box = affine_obb(inference_box, sx=4000 / 640, sy=3000 / 640)

    restored_box = rescale_obb_to_original(
        inference_box,
        original_size=(3000, 4000),
        inference_size=(640, 640),
    )

    assert probiou(restored_box, expected_box).item() > 0.999
