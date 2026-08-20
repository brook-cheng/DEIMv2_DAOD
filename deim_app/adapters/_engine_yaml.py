"""Adapter wrapping ``engine.core.yaml_utils`` for the application layer.

This module is the ONLY place inside ``deim_app/`` (outside the dep-guard's
purview) permitted to import engine YAML utilities. Every other ``deim_app``
module must route engine-YAML access through the two thin functions exposed
here. The boundary is enforced by
``test/deim_app/test_dependency_boundaries.py``.

Why an adapter?

- ``engine.core.yaml_utils.load_config`` carries a *mutable default argument*
  (``cfg=dict()``) that silently leaks state across calls. Forcing ``cfg={}``
  here isolates every load.
- Centralising the engine touch-point means a future engine refactor (or a
  pure-Python replacement) only edits one file.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from engine.core.yaml_utils import load_config as _engine_load_config
from engine.core.yaml_utils import merge_dict as _engine_merge_dict

__all__ = ["load_engine_config", "merge_dict"]


def load_engine_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config through the engine loader with a guaranteed-fresh accumulator.

    The engine's ``load_config(file_path, cfg=dict())`` shares its default
    accumulator across calls (classic mutable-default bug). We always pass a
    fresh ``cfg={}`` so one load cannot poison another.
    """
    return _engine_load_config(str(path), cfg={})


def merge_dict(target: dict[str, Any], source: Mapping[str, Any]) -> dict[str, Any]:
    """Deep-merge ``source`` into ``target`` in place and return ``target``.

    Thin delegate over ``engine.core.yaml_utils.merge_dict`` with the default
    ``inplace=True`` semantics; callers are responsible for deepcopying
    ``target`` first when they need to preserve the original.
    """
    return _engine_merge_dict(target, dict(source), inplace=True)
