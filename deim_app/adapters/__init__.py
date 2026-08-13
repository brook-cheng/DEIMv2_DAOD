"""Adapter modules bridging ``deim_app`` and engine internals.

Modules in this package are the ONLY ``deim_app`` modules permitted to import
from ``engine``, ``tools.model_compare``, or other internal packages. The
boundary is enforced by ``test/deim_app/test_dependency_boundaries.py``.

Non-adapter ``deim_app`` code must route engine access through
``deim_app.adapters._engine_yaml`` (and future adapter siblings) rather than
importing engine modules directly.

Task 5 additions:
  - :class:`DetectionAdapter` — Protocol the facade (Task 7) type-hints against.
  - :class:`DeimDetectionAdapter` — concrete DEIM adapter; the only module
    that builds engine objects and loads checkpoints.
  - :func:`select_model_state` — checkpoint EMA/model selection + module-prefix
    normalization.
"""

from deim_app.adapters.base import DetectionAdapter
from deim_app.adapters.checkpoint import select_model_state
from deim_app.adapters.deim import DeimDetectionAdapter

__all__ = [
    "DeimDetectionAdapter",
    "DetectionAdapter",
    "select_model_state",
]
