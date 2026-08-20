"""reduce_dict schema contract: rank-divergent loss keys must not desync the
collective stream.

Root cause of the epoch-10 2-GPU hang: once Mosaic switches on
(``policy.epoch[0] = 10``), one rank can draw a batch with zero GT boxes;
``denoising.py:30`` then returns None on that rank, the decoder omits
``dn_outputs``/``dn_pre_outputs``, and the criterion omits every ``*_dn_*``
loss key. ``reduce_dict`` stacked the rank-LOCAL keys and called
``all_reduce`` on them, so the two ranks entered the same collective with
different numel -> NCCL desync (flight recorder: rank0 stuck in
``_ALLGATHER_BASE``, rank1 in ``ALLREDUCE NumelIn=97``).

Contract after fix:

- ``reduce_dict`` builds ONE rank-consistent key schema (union of all ranks'
  keys in deterministic sorted order) before the collective.
- A rank missing a key contributes a zero tensor of the same dtype/device.
- Every rank returns the union key set; values are the cross-rank average
  (``avg=True``) exactly as before for keys present on all ranks.
- ``world_size < 2`` keeps the pass-through behavior (returns input dict).

Run:
    pytest test/contracts/test_reduce_dict_schema.py -v
"""

import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.misc import dist_utils  # noqa: E402

_WORKER = ROOT / "test" / "contracts" / "_reduce_dict_worker.py"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# ---------------------------------------------------------------- fake-dist --
@pytest.fixture()
def fake_two_rank_exchange(monkeypatch):
    """Simulate the two-rank key exchange + identity all_reduce in-process.

    ``peer_keys`` is the key set held by the other rank; after the fake
    all_reduce (identity), the locally returned values are simply the
    zero-padded local values divided by world_size.
    """

    def _setup(peer_keys):
        import torch.distributed as dist

        state = {"peer_keys": sorted(peer_keys), "reduced": []}

        def fake_all_gather_object(object_list, obj):
            object_list[:] = [sorted(obj), state["peer_keys"]]

        def fake_all_reduce(tensor, *args, **kwargs):
            state["reduced"].append(tensor.clone())
            # identity: single-process simulation of a no-op reduce

        monkeypatch.setattr(dist_utils, "get_world_size", lambda: 2)
        monkeypatch.setattr(dist, "all_gather_object", fake_all_gather_object)
        monkeypatch.setattr(dist, "all_reduce", fake_all_reduce)
        return state

    return _setup


def test_reduce_dict_unifies_divergent_key_sets(fake_two_rank_exchange):
    state = fake_two_rank_exchange(peer_keys=["loss_main", "loss_dn_0"])

    data = {
        "loss_main": torch.tensor(2.0),
        "loss_boxes": torch.tensor(4.0),
    }
    out = dist_utils.reduce_dict(data)

    assert set(out.keys()) == {"loss_main", "loss_boxes", "loss_dn_0"}
    assert state["reduced"], "reduce_dict must still all_reduce the values"
    # union of 3 keys, deterministic sorted order, zero-padded on this rank
    assert state["reduced"][0].numel() == 3
    # identity reduce + /world_size: present halved, missing stays 0.0
    assert abs(out["loss_main"].item() - 1.0) < 1e-6
    assert abs(out["loss_boxes"].item() - 2.0) < 1e-6
    assert out["loss_dn_0"].item() == 0.0


def test_reduce_dict_pads_missing_keys_with_matching_dtype_device(
    fake_two_rank_exchange,
):
    state = fake_two_rank_exchange(peer_keys=["a", "b"])

    data = {"a": torch.tensor(1.0, dtype=torch.float64)}
    out = dist_utils.reduce_dict(data)

    stacked = state["reduced"][0]
    assert stacked.dtype == torch.float64, "pad must match the present dtype"
    assert stacked.device == torch.device("cpu")
    assert out["b"].item() == 0.0
    assert out["b"].dtype == torch.float64


def test_reduce_dict_avg_false_keeps_values_undivided(fake_two_rank_exchange):
    state = fake_two_rank_exchange(peer_keys=["loss_main", "loss_dn_0"])

    data = {"loss_main": torch.tensor(2.0)}
    out = dist_utils.reduce_dict(data, avg=False)

    assert set(out.keys()) == {"loss_main", "loss_dn_0"}
    assert abs(out["loss_main"].item() - 2.0) < 1e-6
    assert out["loss_dn_0"].item() == 0.0


def test_reduce_dict_single_process_returns_original_dict():
    data = {"loss": torch.tensor(3.0)}
    out = dist_utils.reduce_dict(data)
    assert out is data


# ------------------------------------------------- real multiprocess (gloo) --
def test_reduce_dict_two_processes_with_divergent_keys():
    """True reproduction: two OS processes, different loss keys per rank.

    Against the unfixed reduce_dict, both processes enter ``all_reduce``
    with different numel — gloo raises (NCCL hangs); either way the test
    fails. With the fix, both ranks return the union key set and the
    cross-rank average.
    """
    port = _free_port()
    env = os.environ.copy()
    env.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        WORLD_SIZE="2",
    )

    procs = []
    for rank in (0, 1):
        rank_env = dict(env)
        rank_env["RANK"] = str(rank)
        procs.append(
            subprocess.Popen(
                [sys.executable, str(_WORKER)],
                env=rank_env,
                cwd=str(ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
        )

    outputs = []
    try:
        for proc in procs:
            out, _ = proc.communicate(timeout=180)
            outputs.append(out)
    except subprocess.TimeoutExpired:
        for proc in procs:
            proc.kill()
        pytest.fail(
            "reduce_dict multiprocess test timed out — ranks desynced "
            f"(hung collective). Outputs: {outputs}"
        )

    failures = [p.returncode for p in procs if p.returncode != 0]
    assert not failures, (
        f"worker processes failed (exit codes {failures}):\n{''.join(outputs)}"
    )
    for out in outputs:
        assert "REDUCE_DICT_OK" in out, out
