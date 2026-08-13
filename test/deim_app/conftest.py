"""Shared fixtures for ``deim_app`` facade tests.

The ``fake_adapter`` / ``unloaded_fake_adapter`` fixtures and the
:class:`FakeAdapter` spy are defined in the sibling ``_facade_fakes`` module
and re-exported here so pytest discovers them as conftest-scoped fixtures.

Why this is not registered via ``pytest_plugins``: pytest 9 forbids
``pytest_plugins`` in a non-top-level conftest because it pollutes the
session-wide fixture namespace. ``test/deim_app`` is a nested directory under
``test/``, so the previous ``pytest_plugins = ("_facade_fakes",)`` broke
``python -m pytest test/`` collection with::

    Defining 'pytest_plugins' in a non-top-level conftest is no longer supported

Importing the fixture functions directly exposes them in this module's
namespace; pytest then treats them exactly as if they were defined with
``@pytest.fixture`` here, with no plugin registration involved.

The ``from _facade_fakes import ...`` form is the same one already used by
``test_api.py`` and ``test_cli.py``. It resolves at runtime because pytest
inserts each conftest's directory onto ``sys.path``; basedpyright's
``reportImplicitRelativeImport`` flags it as a sibling-import ambiguity (the
same baseline diagnostic those test modules already carry), which is the
accepted trade-off for not making ``test/deim_app`` a package.
"""

from __future__ import annotations

from _facade_fakes import fake_adapter, unloaded_fake_adapter

__all__ = ("fake_adapter", "unloaded_fake_adapter")
