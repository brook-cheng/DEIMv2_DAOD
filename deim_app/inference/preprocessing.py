"""Canonical resize / tensor / normalize preprocessing for inference.

``Preprocessor`` turns an :class:`~deim_app.inference.inputs.InputImage` into a
:class:`PreparedImage` whose ``tensor`` is a ``(1, 3, H, W)`` float32,
ImageNet-normalized batch ready for the model. The original PIL image and its
``(height, width)`` are retained on the :class:`PreparedImage` so downstream
visualization never reopens or re-resizes the source.

Pipeline choice (documented per the task brief):

  The engine's ``ConvertPILImage`` (``engine/data/transforms/_transforms.py``)
  is a ``torchvision.transforms.v2`` ``Transform`` whose ``_transformed_types``
  is ``(PIL.Image.Image,)`` and whose output is wrapped in the engine's own
  ``Image`` ``tv_tensors`` subclass. That wrapper is designed for the engine's
  sample-tuple training pipeline, not for bare-PIL inference, and unwrapping it
  at the application boundary is awkward. The task brief explicitly permits a
  minimal equivalent built on the public torchvision functional API:

      resize → pil_to_tensor → /255 → normalize

  This keeps ``deim_app/inference/`` completely free of ``engine`` imports
  (the dep-guard permits engine imports only under ``deim_app/adapters/``) and
  produces byte-identical numerics to ``ConvertPILImage`` for an RGB input.

The canonical DINOv3 / DEIM normalization constants (ImageNet mean/std) are
used by default.

Boundary: this module imports only from ``deim_app``, ``torch``, and
``torchvision`` — never from ``engine``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional  # noqa: F401  (ensures torch ops dispatch correctly)
import torchvision.transforms as T
import torchvision.transforms.functional as F
from PIL import Image

from deim_app.inference.inputs import InputImage

__all__ = ["IMAGENET_MEAN", "IMAGENET_STD", "PreparedImage", "Preprocessor"]


#: ImageNet RGB mean — the DINOv3 / DEIM preset normalization.
IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)

#: ImageNet RGB std — the DINOv3 / DEIM preset normalization.
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)


@dataclass(frozen=True, slots=True)
class PreparedImage:
    """A preprocessed input ready for the model plus retained originals.

    Attributes:
        image_id: Forwarded from the source :class:`InputImage`.
        source: Forwarded from the source :class:`InputImage`.
        original_image: The original (unresized) RGB PIL image, retained so the
            visualization writer can redraw detections without re-reading disk.
        original_size_hw: Original image size as ``(height, width)`` (the
            convention used everywhere in ``deim_app`` except the postprocessor
            tensor, which is built as ``[width, height]`` — see
            :class:`~deim_app.inference.torch_backend.TorchBackend`).
        tensor: Model input of shape ``(1, 3, H, W)`` float32, normalized.
    """

    image_id: str
    source: str
    original_image: Image.Image
    original_size_hw: tuple[int, int]
    tensor: torch.Tensor


class Preprocessor:
    """Canonical resize → tensor → normalize pipeline.

    Args:
        input_size: Target ``(height, width)`` for the resize step, matching the
            ``runtime.input_size`` field of the application config. The
            torchvision functional API interprets a length-2 sequence as
            ``[H, W]``.
        normalize_mean: RGB mean for normalization (default: ImageNet).
        normalize_std: RGB std for normalization (default: ImageNet).
    """

    def __init__(
        self,
        input_size: Sequence[int],
        normalize_mean: Sequence[float] = IMAGENET_MEAN,
        normalize_std: Sequence[float] = IMAGENET_STD,
    ) -> None:
        if len(input_size) != 2:
            raise ValueError(
                f"input_size must have exactly two elements (H, W), "
                f"got {len(input_size)}"
            )
        self._input_size: tuple[int, int] = (int(input_size[0]), int(input_size[1]))
        self._mean: tuple[float, ...] = tuple(normalize_mean)
        self._std: tuple[float, ...] = tuple(normalize_std)
        # ``transforms.Resize`` (the Transform class) carries proper PIL/Tensor
        # overloads, unlike ``functional.resize`` whose stub is Tensor-only.
        self._resize: T.Resize = T.Resize(list(self._input_size))

    @property
    def input_size(self) -> tuple[int, int]:
        """The configured ``(height, width)`` resize target."""
        return self._input_size

    def __call__(self, inp: InputImage) -> PreparedImage:
        original = inp.image
        # PIL ``Image.size`` is ``(width, height)``.
        orig_w, orig_h = original.size

        resized = self._resize(original)
        tensor = F.pil_to_tensor(resized).float().div(255.0)
        tensor = F.normalize(tensor, list(self._mean), list(self._std))

        return PreparedImage(
            image_id=inp.image_id,
            source=inp.source,
            original_image=original,
            original_size_hw=(int(orig_h), int(orig_w)),
            tensor=tensor.unsqueeze(0),
        )
