"""Shared fixtures for ``deim_app.adapters`` tests (Task 5).

The stub engine objects live in ``_stubs.py`` so both this conftest and
``test_deim_adapter.py`` can import them without ``test/`` being a package.
The adapter unit tests NEVER construct real engine objects — those require a
GPU + trained weights. Instead we exercise orchestration via the stubs.

Canned ``ResolvedAlgorithmConfig`` / ``LoadedAppConfig`` fixtures let tests
build the adapter without touching the filesystem.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from _stubs import StubYAMLConfig
from deim_app.config.loader import LoadedAppConfig
from deim_app.config.mapping import ResolvedAlgorithmConfig
from deim_app.config.metadata import DatasetMetadata
from deim_app.config.schema import AppConfig


@pytest.fixture(autouse=True)
def _clear_stub_instances() -> None:
    """Reset the class-level stub registry before every test."""
    StubYAMLConfig.instances.clear()
    yield
    StubYAMLConfig.instances.clear()


def _canned_metadata(box_mode: str = "obb", num_classes: int = 15) -> DatasetMetadata:
    names: dict[int, str] = {i: f"cls{i}" for i in range(num_classes)}
    return DatasetMetadata(
        box_mode=box_mode,
        num_classes=num_classes,
        class_names_by_label=names,
        output_names_by_id=dict(names),
    )


@pytest.fixture
def canned_resolved() -> ResolvedAlgorithmConfig:
    """Synthetic resolved config carrying HGNetv2.pretrained=True (the flag the
    adapter must flip to False) and box_mode='obb' / num_classes=15."""
    return ResolvedAlgorithmConfig(
        config_path=Path("/synthetic/app.yml"),
        overrides={
            "HGNetv2": {"pretrained": True},
            "eval_spatial_size": [576, 1024],
            "num_classes": 15,
        },
        metadata=_canned_metadata("obb", 15),
        app=AppConfig(),
    )


@pytest.fixture
def canned_loaded(canned_resolved: ResolvedAlgorithmConfig) -> LoadedAppConfig:
    """Synthetic LoadedAppConfig mirroring ``canned_resolved``."""
    return LoadedAppConfig(
        app=canned_resolved.app,
        engine_base=dict(canned_resolved.overrides),
        source=canned_resolved.config_path,
        app_base=Path("/synthetic/base.yml"),
    )


@pytest.fixture
def call_log() -> list[str]:
    """A fresh shared call-order log per test."""
    return []
