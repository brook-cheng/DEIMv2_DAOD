"""Focused tests for tools/model_compare/obb_utils.py visualization helpers."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from PIL import Image

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
