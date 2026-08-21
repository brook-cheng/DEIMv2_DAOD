"""stop_epoch stage-checkpoint refresh contract: barrier before best_stg1.pth load.

Server run 20260820_182623: rank1 died at epoch 50 with
``EOFError: Ran out of input`` inside ``torch.load`` of ``best_stg1.pth``
while rank0 loaded it fine. ``best_stg1.pth`` is written by rank0 only
(``save_on_master`` at the previous epoch's validation); the epoch-50
refresh loads it on EVERY rank with no barrier in between — rank1 read the
file while rank0 was still streaming it.

Contract after fix:

- With dist initialized: ``torch.distributed.barrier()`` BEFORE
  ``load_resume_state`` (same pattern as the ``best.pth`` restore at the end
  of ``fit()``).
- Missing checkpoint file: skip the load with a warning (no crash), still
  update ``ema.decay`` and preserve ``last_epoch``.
- ``last_epoch`` preserved across the refresh; ``ema.decay`` set to
  ``collate_fn.ema_restart_decay``; early-stopping patience reset preserved.

Run:
    pytest test/contracts/test_stage_checkpoint_barrier.py -v
"""

from types import SimpleNamespace

from engine.solver.det_solver import DetSolver


def _stub(tmp_path, **overrides):
    events = []
    stub = SimpleNamespace(
        output_dir=tmp_path,
        last_epoch=49,
        load_resume_state=lambda path: events.append(("load", str(path))),
        ema=SimpleNamespace(decay=0.9999),
        train_dataloader=SimpleNamespace(
            collate_fn=SimpleNamespace(ema_restart_decay=0.999)
        ),
        early_stopping=None,
    )
    for key, value in overrides.items():
        setattr(stub, key, value)
    stub.events = events
    return stub


def test_barrier_runs_before_checkpoint_load(tmp_path, monkeypatch):
    import torch.distributed as dist
    from engine.misc import dist_utils

    order = []
    monkeypatch.setattr(
        dist_utils, "is_dist_available_and_initialized", lambda: True
    )
    monkeypatch.setattr(dist, "barrier", lambda *a, **k: order.append("barrier"))

    (tmp_path / "best_stg1.pth").write_bytes(b"placeholder")

    def load(path):
        order.append("load")

    stub = _stub(tmp_path, load_resume_state=load)

    DetSolver._load_stage_checkpoint(stub, 50)

    assert order == ["barrier", "load"], "barrier must precede the file read"
    assert stub.last_epoch == 49, "refresh must not advance last_epoch"
    assert stub.ema.decay == 0.999, "ema decay must switch to restart decay"


def test_missing_checkpoint_skips_load_and_keeps_epoch(tmp_path, monkeypatch):
    import torch.distributed as dist
    from engine.misc import dist_utils

    barriers = []
    monkeypatch.setattr(
        dist_utils, "is_dist_available_and_initialized", lambda: True
    )
    monkeypatch.setattr(dist, "barrier", lambda *a, **k: barriers.append(1))

    stub = _stub(tmp_path)  # no best_stg1.pth written
    DetSolver._load_stage_checkpoint(stub, 50)

    assert stub.events == [], "missing checkpoint must not trigger a load"
    assert barriers, "barrier still runs before the existence check"
    assert stub.last_epoch == 49
    assert stub.ema.decay == 0.999


def test_single_process_loads_without_barrier(tmp_path, monkeypatch):
    import torch.distributed as dist
    from engine.misc import dist_utils

    barriers = []
    monkeypatch.setattr(
        dist_utils, "is_dist_available_and_initialized", lambda: False
    )
    monkeypatch.setattr(dist, "barrier", lambda *a, **k: barriers.append(1))

    (tmp_path / "best_stg1.pth").write_bytes(b"placeholder")
    stub = _stub(tmp_path)
    DetSolver._load_stage_checkpoint(stub, 50)

    assert barriers == [], "no barrier in single-process mode"
    assert len(stub.events) == 1 and stub.events[0][0] == "load"
    assert stub.last_epoch == 49
    assert stub.ema.decay == 0.999


def test_early_stopping_patience_reset_is_preserved(tmp_path, monkeypatch):
    from engine.misc import dist_utils

    monkeypatch.setattr(
        dist_utils, "is_dist_available_and_initialized", lambda: False
    )
    resets = []
    early_stopping = SimpleNamespace(
        best_epoch=12,
        best_observed_metric=0.3456,
        reset_patience=lambda: resets.append(1),
    )
    (tmp_path / "best_stg1.pth").write_bytes(b"placeholder")
    stub = _stub(tmp_path, early_stopping=early_stopping)

    DetSolver._load_stage_checkpoint(stub, 50)

    assert resets == [1], "early-stopping patience reset must be preserved"
