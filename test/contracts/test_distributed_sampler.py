"""Sampler ownership contract: warp_loader owns DistributedSampler wiring.

Layer map (verified on server 20260820_152043):
- ``YAMLConfig.build_dataloader``: per-rank batch size only, NO sampler
  (upstream design).
- ``DetSolver.train`` -> ``dist_utils.warp_loader``: rebuilds the loader with
  a ``DistributedSampler``; shuffle lives in the sampler, NOT the DataLoader —
  torch rejects ``DataLoader(shuffle=True, sampler=...)``.
- ``det_solver`` per-epoch: ``loader.set_epoch`` + ``sampler.set_epoch``.

Attaching a sampler inside ``build_dataloader`` double-wires the chain and
crashes on configs with ``shuffle: true`` (server 20260820_152043).

Run:
    pytest test/contracts/test_distributed_sampler.py -v
"""

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.core import yaml_config as yc  # noqa: E402
from engine.core.yaml_config import YAMLConfig  # noqa: E402

_CFG = os.path.join(ROOT, "configs/custom_obb/synthetic_configs/synthetic_exp_002.yml")


class _FakeLoader:
    def __init__(self, **kw):
        self.dataset = list(range(100))
        self.sampler = None
        self.shuffle = False
        self.__dict__.update(kw)


@pytest.fixture()
def _capture_create(monkeypatch):
    calls: list = []

    def fake_create(name, global_cfg, **kw):
        calls.append({"name": name, **kw})
        return _FakeLoader(batch_size=kw.get("batch_size", 1))

    monkeypatch.setattr(yc, "create", fake_create)
    return calls


@pytest.fixture()
def _two_rank_world(monkeypatch):
    import torch.distributed as dist

    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(dist, "get_rank", lambda: 0)


@pytest.fixture()
def _single_process(monkeypatch):
    import torch.distributed as dist

    monkeypatch.setattr(dist, "is_initialized", lambda: False)


def _cfg_with_shuffle(flag):
    cfg = YAMLConfig(_CFG)
    cfg.yaml_cfg["train_dataloader"]["shuffle"] = flag
    return cfg


def test_two_rank_build_leaves_sampler_to_warp_loader(
    _two_rank_world, _capture_create
):
    loader_calls = []
    _cfg_with_shuffle(True).build_dataloader("train_dataloader")

    loader_calls = [c for c in _capture_create if "batch_size" in c]
    assert loader_calls, "build_dataloader must create the loader"
    assert "sampler" not in loader_calls[0], (
        "sampler wiring belongs to dist_utils.warp_loader (called by the "
        "solver); attaching it in build_dataloader double-wires the chain "
        "and conflicts with config shuffle:true "
        "(server 20260820_152043 mutual-exclusion crash)"
    )
    assert "dataset" not in loader_calls[0]


def test_single_process_build_stays_sampler_free(_single_process, _capture_create):
    from torch.utils.data import DistributedSampler

    loader = _cfg_with_shuffle(True).build_dataloader("train_dataloader")
    assert not isinstance(loader.sampler, DistributedSampler)


def test_warp_loader_attaches_distributed_sampler_without_shuffle_conflict(
    monkeypatch,
):
    import torch.distributed as dist
    from engine.misc import dist_utils
    from torch.utils.data import DistributedSampler

    monkeypatch.setattr(dist_utils, "is_dist_available_and_initialized", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(dist, "get_rank", lambda: 0)

    raw = SimpleNamespace(
        dataset=list(range(101)),
        batch_size=4,
        drop_last=True,
        collate_fn=None,
        pin_memory=False,
        num_workers=0,
    )
    warped = dist_utils.warp_loader(raw, shuffle=True)

    assert isinstance(warped.sampler, DistributedSampler)
    assert warped.sampler.shuffle is True
    assert warped.sampler.num_samples == 51


def test_create_kwargs_override_injected_dataset(monkeypatch):
    from engine.core.workspace import create
    import engine.data.dataloader as dataloader_module

    global_cfg = {
        "DataLoader": {
            "_name": "DataLoader",
            "_pymodule": dataloader_module,
        }
    }

    class _Dataset:
        pass

    configured = _Dataset()
    explicit = _Dataset()
    global_cfg["DataLoader"].update(
        {
            "_inject": ["dataset"],
            "_share": [],
            "dataset": "configured_dataset",
            "batch_size": 1,
        }
    )
    global_cfg["configured_dataset"] = configured

    monkeypatch.setattr(yc, "create", create)
    loader = create("DataLoader", global_cfg, dataset=explicit)

    assert loader.dataset is explicit
