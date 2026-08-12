"""Adapter modules bridging ``deim_app`` and engine internals.

Modules in this package are the ONLY ``deim_app`` modules permitted to import
from ``engine``, ``tools.model_compare``, or other internal packages. The
boundary is enforced by ``test/deim_app/test_dependency_boundaries.py``.

Non-adapter ``deim_app`` code must route engine access through
``deim_app.adapters._engine_yaml`` (and future adapter siblings) rather than
importing engine modules directly.
"""
