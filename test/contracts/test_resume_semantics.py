"""-r <file> checkpoint-training semantics: load state, train a FRESH schedule.

User contract (chosen semantics, "A 字面理想式"):
- ``-r auto``  → recovery: restore everything INCLUDING the epoch counter and
  continue from where the run stopped (absolute epoches).
- ``-r <file>`` → 断点训练: the checkpoint seeds weights/EMA/optimizer
  (+kendall), but training follows the REGULAR flow — the epoch schedule
  restarts from scratch. A checkpoint saved at epoch 99 with ``epoches=50``
  must train 50 epochs, not exit early (the old absolute-epoches behavior
  made ``range(100, 50)`` empty and silently ended the run).

Contract for ``_load_checkpoint_fresh_schedule``:
- Loads via load_resume_state (full state incl. kendall — the fit-path load
  is deferred past kendall construction).
- Resets ``last_epoch = -1`` (fresh epoch flow from 0).
- Re-initializes early stopping (fresh state, not the restored one).
- Rebuilds the LR warmup scheduler (cleared cfg cache → fresh warmup).

Run:
    pytest test/contracts/test_resume_semantics.py -v
"""

import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.solver.det_solver import DetSolver  # noqa: E402


def _stub(tmp_path):
    calls = []

    def load_resume_state(path):
        calls.append(("load", str(path)))
        stub.last_epoch = 99

    def init_early_stopping():
        calls.append(("early_stopping_fresh", None))

    stub = SimpleNamespace(
        cfg=SimpleNamespace(
            epoches=2,
            _lr_warmup_scheduler=object(),
            lr_warmup_scheduler=None,
        ),
        last_epoch=-1,
        load_resume_state=load_resume_state,
        _init_early_stopping=init_early_stopping,
    )
    return stub, calls


def test_file_mode_loads_then_resets_epoch_schedule(tmp_path):
    stub, calls = _stub(tmp_path)
    ckpt = tmp_path / "best_stg2.pth"

    DetSolver._load_checkpoint_fresh_schedule(stub, str(ckpt))

    assert calls[0] == ("load", str(ckpt)), "must load the given checkpoint"
    assert stub.last_epoch == -1, (
        "file mode must reset the epoch counter so a saved-epoch-99 "
        "checkpoint with epoches=50 trains a fresh 50-epoch schedule"
    )
    assert ("early_stopping_fresh", None) in calls, (
        "early stopping must be re-initialized for the fresh schedule"
    )


def test_file_mode_rebuilds_warmup_scheduler(tmp_path):
    stub, calls = _stub(tmp_path)

    DetSolver._load_checkpoint_fresh_schedule(stub, str(tmp_path / "last.pth"))

    assert stub.cfg._lr_warmup_scheduler is None, (
        "cfg warmup cache must be cleared so the property rebuilds a fresh "
        "warmup scheduler for the regular training flow"
    )


def test_file_mode_announces_fresh_schedule(tmp_path, capsys):
    stub, _ = _stub(tmp_path)

    DetSolver._load_checkpoint_fresh_schedule(stub, str(tmp_path / "last.pth"))

    out = capsys.readouterr().out
    assert "fresh schedule" in out and "2 epochs" in out
