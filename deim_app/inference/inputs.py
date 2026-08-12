"""Input enumeration for the shared inference pipeline.

``list_inputs`` accepts a single image path, a directory of images, or an
in-memory ``PIL.Image.Image`` and returns a tuple of :class:`InputImage`
records. Every record carries a stable ``image_id`` (file stem for paths, a
monotonic ``memory-000001``-style id for in-memory images) plus the already-
loaded RGB ``PIL.Image.Image`` so downstream stages never need to reopen the
source.

Directory enumeration is non-recursive in v1 and limited to the four
extensions the rest of the application layer treats as supported
(``.jpg``, ``.jpeg``, ``.png``, ``.bmp``). Missing paths and empty directories
raise :class:`InputSourceError`.

Boundary: this module imports only from ``deim_app`` and ``PIL`` — never from
``engine``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Union

from PIL import Image

from deim_app.errors import InputSourceError

__all__ = ["InputImage", "InputSource", "list_inputs"]


#: Supported image extensions for directory enumeration (lowercase, dot-prefixed).
SUPPORTED_IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp"}
)


#: Union alias for anything ``list_inputs`` accepts.
InputSource = Union[str, Path, Image.Image]


@dataclass(frozen=True, slots=True)
class InputImage:
    """A single enumerated input carrying its loaded RGB image.

    Attributes:
        image_id: Stable label for downstream output naming. File stem for
            path / directory inputs (e.g. ``"P0001"`` for ``P0001.png``);
            ``"memory-000001"``-style monotonic id for in-memory images.
        source: Human-readable source label. The resolved path string for
            file / directory inputs; ``"<memory>"`` for in-memory images.
        image: The loaded RGB PIL image. Always converted to mode ``"RGB"``
            so downstream tensor conversion sees a 3-channel image.
    """

    image_id: str
    source: str
    image: Image.Image


def list_inputs(source: InputSource) -> tuple[InputImage, ...]:
    """Enumerate inputs from a path, directory, or in-memory PIL image.

    Resolution rules:
      * ``PIL.Image.Image`` → one :class:`InputImage` with a monotonic
        ``memory-%06d`` id (counter starts at 1, local to this call).
      * File path → one :class:`InputImage` (id = file stem).
      * Directory → sorted supported images (non-recursive), id = stem each.

    Raises:
        InputSourceError: if a path does not exist, a file path is not a
            supported image, or a directory contains zero supported images.
    """
    # In-memory PIL image.
    if isinstance(source, Image.Image):
        rgb = _as_rgb(source)
        return (InputImage(image_id="memory-000001", source="<memory>", image=rgb),)

    # Str / Path → resolve.
    raw_path = Path(str(source)) if isinstance(source, str) else Path(source)
    if not raw_path.exists():
        raise InputSourceError(
            f"input source '{raw_path}' does not exist"
        )

    raw_path = raw_path.resolve()

    if raw_path.is_dir():
        return _enumerate_directory(raw_path)

    if raw_path.is_file():
        if raw_path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
            raise InputSourceError(
                f"input file '{raw_path}' has unsupported extension "
                f"'{raw_path.suffix}'; supported: "
                f"{sorted(SUPPORTED_IMAGE_EXTENSIONS)}"
            )
        return (_input_from_file(raw_path),)

    # Neither file nor dir (e.g. a socket / device node) — treat as unusable.
    raise InputSourceError(
        f"input source '{raw_path}' is neither a file nor a directory"
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _enumerate_directory(directory: Path) -> tuple[InputImage, ...]:
    """Return supported images in ``directory`` sorted by filename.

    Non-recursive. Raises :class:`InputSourceError` when no supported images
    are found so a caller never silently receives an empty collection.
    """
    candidates = sorted(
        entry
        for entry in directory.iterdir()
        if entry.is_file() and entry.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    )
    if not candidates:
        raise InputSourceError(
            f"directory '{directory}' contains no supported images "
            f"(extensions: {sorted(SUPPORTED_IMAGE_EXTENSIONS)})"
        )
    return tuple(_input_from_file(entry) for entry in candidates)


def _input_from_file(path: Path) -> InputImage:
    """Build an :class:`InputImage` from a file path (stem id, RGB-loaded)."""
    with Image.open(path) as handle:
        rgb = _as_rgb(handle)
    return InputImage(image_id=path.stem, source=str(path), image=rgb)


def _as_rgb(image: Image.Image) -> Image.Image:
    """Return an RGB copy of ``image`` (covers 'L', 'RGBA', 'P', etc.)."""
    if image.mode == "RGB":
        # Copy so the caller owns an independent buffer (the file handle may
        # close after returning from ``Image.open``'s context manager).
        return image.copy()
    return image.convert("RGB")
