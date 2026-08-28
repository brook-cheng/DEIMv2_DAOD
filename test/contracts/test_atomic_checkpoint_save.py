"""Atomic checkpoint write contract: power loss must never corrupt the
recovery point.

``save_on_master`` streams ``torch.save`` directly into the target file — a
crash/power loss mid-write leaves a truncated ``last.pth`` and the recovery
mechanism itself becomes unusable (the same EOFError class as the epoch-50
read race). ``save_on_master_atomic`` must write to ``<path>.tmp`` and
``os.replace`` it into place (atomic on POSIX), so the target always holds
either the old or the new complete content.

Contract:
- Main process: file written, loadable, no ``.tmp`` residue.
- Write failure: previous target intact, ``.tmp`` cleaned up.
- Non-main process: nothing written.

Run:
    pytest test/contracts/test_atomic_checkpoint_save.py -v
"""

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.misc import dist_utils  # noqa: E402


def test_atomic_save_writes_loadable_file_without_tmp_residue(tmp_path):
    target = tmp_path / "last.pth"

    dist_utils.save_on_master_atomic({"last_epoch": 41}, target)

    assert target.exists()
    assert not Path(str(target) + ".tmp").exists()
    assert torch.load(target, weights_only=False)["last_epoch"] == 41


def test_atomic_save_failure_keeps_previous_file_intact(tmp_path, monkeypatch):
    target = tmp_path / "last.pth"
    dist_utils.save_on_master_atomic({"last_epoch": 7}, target)

    def exploding_save(state, path):
        Path(path).write_bytes(b"garbage-partial-write")
        raise OSError("simulated power loss mid-write")

    monkeypatch.setattr(dist_utils, "is_main_process", lambda: True)
    monkeypatch.setattr(dist_utils.torch, "save", exploding_save)

    with pytest.raises(OSError):
        dist_utils.save_on_master_atomic({"last_epoch": 8}, target)

    assert torch.load(target, weights_only=False)["last_epoch"] == 7, (
        "previous checkpoint must survive a failed write"
    )
    assert not Path(str(target) + ".tmp").exists(), "tmp must be cleaned up"


def test_atomic_save_skips_non_main_process(tmp_path, monkeypatch):
    target = tmp_path / "last.pth"

    def exploding_save(*a, **kw):
        raise AssertionError("non-main process must not write")

    monkeypatch.setattr(dist_utils, "is_main_process", lambda: False)
    monkeypatch.setattr(dist_utils.torch, "save", exploding_save)

    dist_utils.save_on_master_atomic({"last_epoch": 1}, target)

    assert not target.exists()
