"""Immutable prediction types and collections for the DEIMv2 application layer.

Public surface (re-exported here for convenience):
    - ``HBBDetection`` / ``OBBDetection`` — frozen detection dataclasses.
    - ``Detection`` — Union alias for either kind.
    - ``Timings`` — per-image inference timings.
    - ``ImagePrediction`` — per-image aggregate (detections + metadata + image).
    - ``PredictionCollection`` — frozen collection with non-mutating
      ``filter`` / ``top_k`` and lazy ``export_json`` / ``export_dota`` /
      ``save_images`` methods.

Boundary: this package imports only from ``deim_app`` — never from ``engine``
or ``tools.model_compare`` (enforced by the dep-guard test).
"""

from deim_app.predictions.collection import PredictionCollection
from deim_app.predictions.types import (
    Detection,
    HBBDetection,
    ImagePrediction,
    OBBDetection,
    Timings,
)

__all__ = [
    "Detection",
    "HBBDetection",
    "ImagePrediction",
    "OBBDetection",
    "PredictionCollection",
    "Timings",
]
