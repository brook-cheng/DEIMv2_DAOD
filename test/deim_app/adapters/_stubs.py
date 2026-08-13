"""Stub engine objects for ``deim_app.adapters`` tests (Task 5).

Importable from both ``conftest.py`` (for the autouse registry-clear fixture)
and ``test_deim_adapter.py`` (for the inline stub builders). Kept out of
``conftest.py`` because ``test/`` is not a package and pytest's conftest is not
meant to be imported by name from test modules.
"""

from __future__ import annotations

from typing import Any

import torch


class StubModel:
    """Records load_state_dict / deploy / to calls; serves a canned state_dict."""

    def __init__(
        self,
        call_log: list[str],
        state: dict[str, torch.Tensor] | None = None,
    ) -> None:
        self._call_log = call_log
        self._state = dict(state) if state is not None else {}
        self.last_loaded_state: Any = None
        self.last_strict: Any = None
        self.load_state_dict_calls: int = 0
        self.deploy_calls: int = 0
        self.to_calls: int = 0
        self.last_device: Any = None

    def state_dict(self) -> dict[str, torch.Tensor]:
        return dict(self._state)

    def load_state_dict(self, state: Any, strict: bool = True) -> "_MissingKeys":
        self._call_log.append("model.load_state_dict")
        self.last_loaded_state = state
        self.last_strict = strict
        self.load_state_dict_calls += 1
        return _MissingKeys(missing_keys=[], unexpected_keys=[])

    def deploy(self) -> "StubModel":
        self._call_log.append("model.deploy")
        self.deploy_calls += 1
        return self

    def to(self, *args: Any, **kwargs: Any) -> "StubModel":
        """Mirror nn.Module.to(); record the device argument and return self."""
        self._call_log.append("model.to")
        self.to_calls += 1
        self.last_device = args[0] if args else kwargs.get("device")
        return self


class StubPostprocessor:
    """Records its deploy() and to() calls."""

    def __init__(self, call_log: list[str]) -> None:
        self._call_log = call_log
        self.deploy_calls: int = 0
        self.to_calls: int = 0
        self.last_device: Any = None

    def deploy(self) -> "StubPostprocessor":
        self._call_log.append("postprocessor.deploy")
        self.deploy_calls += 1
        return self

    def to(self, *args: Any, **kwargs: Any) -> "StubPostprocessor":
        """Mirror nn.Module.to(); record the device argument and return self."""
        self._call_log.append("postprocessor.to")
        self.to_calls += 1
        self.last_device = args[0] if args else kwargs.get("device")
        return self


class _MissingKeys:
    """Stand-in for the NamedTuple returned by nn.Module.load_state_dict."""

    def __init__(self, missing_keys: list[str], unexpected_keys: list[str]) -> None:
        self.missing_keys = missing_keys
        self.unexpected_keys = unexpected_keys


class StubYAMLConfig:
    """Records cfg_path + kwargs; exposes a mutable yaml_cfg and stub children.

    ``instances`` is a class-level registry so tests can retrieve the single
    stub the adapter constructed during ``load()``.
    """

    instances: list["StubYAMLConfig"] = []

    def __init__(
        self,
        cfg_path: str,
        call_log: list[str] | None = None,
        yaml_cfg: dict[str, Any] | None = None,
        model: StubModel | None = None,
        postprocessor: StubPostprocessor | None = None,
        **kwargs: Any,
    ) -> None:
        self.cfg_path = cfg_path
        self.kwargs = dict(kwargs)
        self._call_log = call_log if call_log is not None else []
        self.yaml_cfg = (
            yaml_cfg if yaml_cfg is not None else {"HGNetv2": {"pretrained": True}}
        )
        self.model = model if model is not None else StubModel(self._call_log)
        self.postprocessor = (
            postprocessor
            if postprocessor is not None
            else StubPostprocessor(self._call_log)
        )
        StubYAMLConfig.instances.append(self)
