"""Shared fixtures for ``deim_app.inference`` tests (Task 6).

Only pytest fixtures live here; stub classes and helper functions are defined
inline in the test modules that use them, avoiding implicit-relative imports.
"""

from __future__ import annotations

import pytest

from deim_app.inference.preprocessing import Preprocessor


@pytest.fixture
def small_preprocessor() -> Preprocessor:
    """A Preprocessor with a 16x16 input size (fast, no CUDA needed)."""
    return Preprocessor(input_size=(16, 16))
