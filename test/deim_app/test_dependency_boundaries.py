from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def test_engine_never_imports_deim_app() -> None:
    violations = []
    for path in (ROOT / "engine").rglob("*.py"):
        if any(name == "deim_app" or name.startswith("deim_app.") for name in imported_modules(path)):
            violations.append(path.relative_to(ROOT))
    assert violations == []


def test_only_adapters_import_engine_solver_or_model_internals() -> None:
    violations = []
    for path in (ROOT / "deim_app").rglob("*.py"):
        if "adapters" in path.parts:
            continue
        forbidden = {
            name
            for name in imported_modules(path)
            if name == "engine"
            or name.startswith("engine.")
            or name.startswith("tools.model_compare")
        }
        if forbidden:
            violations.append((path.relative_to(ROOT), sorted(forbidden)))
    assert violations == []
