"""Shared PyTorch inference pipeline for the DEIMv2 application layer.

Public surface (re-exported here for convenience):
    - :class:`InputImage` / :data:`InputSource` / :func:`list_inputs` — input
      enumeration over paths, directories, and in-memory PIL images.
    - :class:`PreparedImage` / :class:`Preprocessor` — canonical resize /
      tensor / normalize preprocessing.
    - :class:`TorchBackend` — batched PyTorch inference returning a
      :class:`~deim_app.predictions.collection.PredictionCollection`.

Boundary: this package imports only from ``deim_app`` and ``torch`` /
``torchvision`` / ``PIL``. It MUST NOT import ``engine.*`` — the dependency
guard (``test/deim_app/test_dependency_boundaries.py``) forbids engine imports
outside ``deim_app/adapters/``. Preprocessing uses the torchvision functional
API directly (see ``preprocessing.py`` for the rationale).
"""

from deim_app.inference.inputs import InputImage, InputSource, list_inputs
from deim_app.inference.preprocessing import PreparedImage, Preprocessor
from deim_app.inference.torch_backend import TorchBackend

__all__ = [
    "InputImage",
    "InputSource",
    "PreparedImage",
    "Preprocessor",
    "TorchBackend",
    "list_inputs",
]
