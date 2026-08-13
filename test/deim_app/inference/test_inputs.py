"""Tests for ``deim_app.inference.inputs`` (Task 6, Step 1).

Covers:
  - supported image file → one input
  - directory → sorted supported images, ignores unrelated, non-recursive
  - in-memory ``PIL.Image.Image`` → one RGB input with ``memory-000001`` id
  - missing path → ``InputSourceError``
  - empty dir → ``InputSourceError``
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from deim_app.errors import InputSourceError
from deim_app.inference.inputs import InputImage, list_inputs


def _write_image(path: Path, size: tuple[int, int] = (8, 8), color=(0, 0, 0)) -> None:
    Image.new("RGB", size, color).save(path)


def test_supported_image_file_returns_one_input(tmp_path: Path) -> None:
    img_path = tmp_path / "P0001.png"
    _write_image(img_path, size=(10, 12))

    inputs = list_inputs(str(img_path))

    assert len(inputs) == 1
    only = inputs[0]
    assert isinstance(only, InputImage)
    assert only.image_id == "P0001"
    assert only.source == str(img_path.resolve())
    assert only.image.mode == "RGB"
    assert only.image.size == (10, 12)


def test_directory_returns_sorted_supported_images(tmp_path: Path) -> None:
    # Supported images out of insertion order.
    _write_image(tmp_path / "b.png", size=(4, 4))
    _write_image(tmp_path / "a.jpg", size=(4, 4))
    _write_image(tmp_path / "c.bmp", size=(4, 4))
    # Unrelated files ignored.
    (tmp_path / "notes.txt").write_text("ignore me")
    (tmp_path / "a.json").write_text("{}")

    inputs = list_inputs(str(tmp_path))

    ids = [inp.image_id for inp in inputs]
    # Sorted by filename stem; extensions preserved on disk but id is the stem.
    assert ids == ["a", "b", "c"]
    for inp in inputs:
        assert inp.image.mode == "RGB"


def test_directory_enumeration_is_non_recursive(tmp_path: Path) -> None:
    _write_image(tmp_path / "top.png", size=(4, 4))
    nested = tmp_path / "nested"
    nested.mkdir()
    _write_image(nested / "deep.png", size=(4, 4))

    inputs = list_inputs(str(tmp_path))

    assert [inp.image_id for inp in inputs] == ["top"]


def test_in_memory_pil_image_returns_rgb_input_with_memory_id() -> None:
    raw = Image.new("RGBA", (5, 6), (255, 0, 0, 255))

    inputs = list_inputs(raw)

    assert len(inputs) == 1
    only = inputs[0]
    assert only.image_id == "memory-000001"
    assert only.source == "<memory>"
    # Converted to RGB regardless of the source mode.
    assert only.image.mode == "RGB"
    assert only.image.size == (5, 6)


def test_missing_path_raises_input_source_error(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist.png"
    with pytest.raises(InputSourceError):
        list_inputs(str(missing))


def test_empty_directory_raises_input_source_error(tmp_path: Path) -> None:
    with pytest.raises(InputSourceError):
        list_inputs(str(tmp_path))


def test_unsupported_extension_file_raises(tmp_path: Path) -> None:
    txt = tmp_path / "data.txt"
    txt.write_text("not an image")
    with pytest.raises(InputSourceError):
        list_inputs(str(txt))


def test_directory_with_duplicate_stem_across_extensions_raises(tmp_path: Path) -> None:
    """Two supported files sharing a stem would collide on ``image_id``.

    Given a directory containing ``same.jpg`` and ``same.png``, the writer
    layer (which derives its output filename from ``image_id``) would silently
    overwrite one result with the other. The input enumeration boundary must
    reject the directory before any image is loaded or inference runs, naming
    the colliding stem so the user can disambiguate the sources.
    """
    # Given: a directory with two supported images sharing stem "same".
    _write_image(tmp_path / "same.jpg", size=(4, 4))
    _write_image(tmp_path / "same.png", size=(4, 4))
    # And: an unrelated non-colliding supported image that must NOT suppress
    # the collision check (it would be silently dropped if we just deduped).
    _write_image(tmp_path / "other.bmp", size=(4, 4))

    # When: enumerating the directory.
    with pytest.raises(InputSourceError) as excinfo:
        list_inputs(str(tmp_path))

    # Then: the error names the colliding stem deterministically.
    message = str(excinfo.value)
    assert "same" in message
    # And: the message is deterministic (sorted, no insertion-order leak).
    assert "other" not in message
