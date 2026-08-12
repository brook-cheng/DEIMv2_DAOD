"""Shared fixtures for ``deim_app`` facade tests (Task 7).

The ``fake_adapter`` / ``unloaded_fake_adapter`` fixtures and the
:class:`~_facade_fakes.FakeAdapter` spy live in the ``_facade_fakes`` module.
They are registered here via ``pytest_plugins`` so pytest auto-discovers them
as if they were defined directly in this conftest.

This split avoids two problems:
  1. The ``conftest`` module-name collision between ``test/deim_app/`` and
     ``test/deim_app/predictions/`` (both have ``conftest.py`` without a
     parent ``__init__.py``).
  2. basedpyright's ``reportImplicitRelativeImport`` rule, which treats
     conftest.py as a package member and flags ``from <sibling> import ...``
     even though pytest's sys.path convention makes it a valid absolute
     import at runtime.
"""

pytest_plugins = ("_facade_fakes",)
