"""Auto-recovery contract: resume from output_dir/last.pth after power loss.

Motivation: external interruptions (power outages) kill training mid-run.
With ``recovery`` enabled (``--resume auto`` or yaml ``recovery: true``),
``DetSolver.fit()`` must auto-resume BEFORE the pre-resume evaluation and
``start_epoch`` computation, so ``last_epoch`` flows through the existing
resume path.

Tier order: ``last.pth`` (atomic per-epoch save) → newest
``checkpoint{NNNN}.pth`` snapshot → fresh start. An unusable tier falls
through instead of aborting. With dist initialized, a barrier precedes any
load (all ranks make the same decision on the same file state).

Run:
    pytest test/contracts/test_auto_recovery.py -v
"""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.solver.det_solver import DetSolver  # noqa: E402

_TRAIN_CLI = ROOT / "tools" / "train" / "train.py"


def _load_train_cli():
    spec = importlib.util.spec_from_file_location("train_cli", _TRAIN_CLI)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _stub(tmp_path, load_impl):
    return SimpleNamespace(
        output_dir=tmp_path,
        last_epoch=-1,
        load_resume_state=load_impl,
    )


def test_recovers_from_last_pth(tmp_path):
    (tmp_path / "last.pth").write_bytes(b"ckpt")
    events = []

    def load(path):
        events.append(str(path))
        stub.last_epoch = 41

    stub = _stub(tmp_path, load)
    DetSolver._maybe_auto_recover(stub)

    assert events == [str(tmp_path / "last.pth")]
    assert stub.last_epoch == 41


def test_falls_back_to_newest_snapshot_when_last_pth_unusable(tmp_path):
    (tmp_path / "last.pth").write_bytes(b"truncated")
    (tmp_path / "checkpoint0039.pth").write_bytes(b"ckpt")
    (tmp_path / "checkpoint0049.pth").write_bytes(b"ckpt")
    attempts = []

    def load(path):
        attempts.append(Path(path).name)
        if Path(path).name == "last.pth":
            raise EOFError("Ran out of input")
        stub.last_epoch = 49

    stub = _stub(tmp_path, load)
    DetSolver._maybe_auto_recover(stub)

    assert attempts == ["last.pth", "checkpoint0049.pth"], (
        "tier order must be last.pth then the NEWEST snapshot"
    )
    assert stub.last_epoch == 49


def test_no_checkpoint_starts_fresh_without_error(tmp_path):
    events = []

    stub = _stub(tmp_path, lambda path: events.append(path))
    DetSolver._maybe_auto_recover(stub)

    assert events == []


def test_barrier_precedes_load_in_dist_mode(tmp_path, monkeypatch):
    import torch.distributed as dist
    from engine.misc import dist_utils

    (tmp_path / "last.pth").write_bytes(b"ckpt")
    order = []

    monkeypatch.setattr(
        dist_utils, "is_dist_available_and_initialized", lambda: True
    )
    monkeypatch.setattr(dist, "barrier", lambda *a, **k: order.append("barrier"))

    def load(path):
        order.append("load")

    stub = _stub(tmp_path, load)
    DetSolver._maybe_auto_recover(stub)

    assert order == ["barrier", "load"]


def test_unusable_everything_starts_fresh_with_warning(tmp_path, capsys):
    (tmp_path / "last.pth").write_bytes(b"truncated")

    def load(path):
        raise EOFError("Ran out of input")

    stub = _stub(tmp_path, load)
    DetSolver._maybe_auto_recover(stub)

    assert stub.last_epoch == -1
    assert "starting fresh" in capsys.readouterr().out


def test_none_output_dir_is_a_no_op():
    events = []
    stub = SimpleNamespace(
        output_dir=None,
        last_epoch=-1,
        load_resume_state=lambda path: events.append(path),
    )
    DetSolver._maybe_auto_recover(stub)
    assert events == []


def test_resolve_recovery_maps_auto_sentinel():
    train_cli = _load_train_cli()
    assert train_cli.resolve_recovery("auto") == (None, True)
    assert train_cli.resolve_recovery("outputs/x/last.pth") == (
        "outputs/x/last.pth",
        False,
    )
    assert train_cli.resolve_recovery(None) == (None, False)
