"""``setup_distributed`` failure visibility (multi-GPU diagnosis finding #0).

On the training server, torchrun started both ranks yet each printed
``Not init distributed mode.`` — ``init_process_group`` failed and the bare
``except`` swallowed the reason. Both ranks then silently degraded to
independent single-GPU processes squeezed onto cuda:0 (GPU1 stayed at 0 MiB,
batch_size was never divided by world_size). The failure cause itself stayed
invisible.

Contract:
* torchrun context (``RANK`` set) + init failure → the exception MUST
  propagate with a printed diagnosis (env vars + traceback). Silent
  degradation to N competing single-GPU processes is the worst outcome.
* plain single-process context (no ``RANK``) → init failure stays silent
  (that IS the normal non-distributed path).
* success path sets the CUDA device from ``LOCAL_RANK`` (multi-node correct),
  not the global rank.

Run:
    pytest test/contracts/test_setup_distributed_visibility.py -v
"""

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from engine.misc import dist_utils  # noqa: E402


@pytest.fixture(autouse=True)
def _restore_builtin_print():
    """``setup_distributed`` → ``setup_print`` permanently rebinds
    ``builtins.print``; without restoring it, this module leaks a filtered
    print into the whole test process (CLI stderr assertions die)."""
    import builtins

    original = builtins.print
    yield
    builtins.print = original


def _clear_dist_env(monkeypatch):
    for var in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
        monkeypatch.delenv(var, raising=False)


def test_torchrun_context_init_failure_raises(monkeypatch, capsys):
    _clear_dist_env(monkeypatch)
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "2")

    def _boom(*a, **k):
        raise RuntimeError("nccl init exploded")

    monkeypatch.setattr(torch.distributed, "init_process_group", _boom)
    with pytest.raises(RuntimeError, match="nccl init exploded"):
        dist_utils.setup_distributed()
    captured = capsys.readouterr()
    out = captured.out + captured.err
    assert "RANK=1" in out and "WORLD_SIZE=2" in out, (
        "failure diagnostics must print the torchrun env for post-mortem"
    )
    assert "Traceback" in out


def test_plain_single_process_stays_silent(monkeypatch, capsys):
    _clear_dist_env(monkeypatch)

    def _boom(*a, **k):
        raise RuntimeError("no master addr")

    monkeypatch.setattr(torch.distributed, "init_process_group", _boom)
    enabled = dist_utils.setup_distributed()
    out = capsys.readouterr().out
    assert enabled is False
    assert "Not init distributed mode." in out
    assert "Traceback" not in out


def test_success_path_sets_device_from_local_rank(monkeypatch, capsys):
    _clear_dist_env(monkeypatch)
    monkeypatch.setenv("RANK", "3")
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "4")

    monkeypatch.setattr(
        torch.distributed, "init_process_group", lambda *a, **k: None
    )
    monkeypatch.setattr(torch.distributed, "barrier", lambda *a, **k: None)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 3)
    monkeypatch.setattr(dist_utils, "get_rank", lambda: 3)
    set_device_calls = []
    monkeypatch.setattr(
        torch.cuda, "set_device", lambda d: set_device_calls.append(d)
    )
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)

    enabled = dist_utils.setup_distributed()
    assert enabled is True
    assert set_device_calls == [1], "device must come from LOCAL_RANK, not RANK"
