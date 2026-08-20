"""Subprocess worker for the real multiprocess reduce_dict regression test.

Spawned as two independent OS processes (torchrun-style) by
``test_reduce_dict_schema.py``. The two ranks deliberately build loss dicts
with DIFFERENT key sets — rank 0 carries an extra ``loss_dn_0`` entry, the
exact data-dependent divergence observed in the epoch-10 NCCL hang (one rank
draws a zero-GT batch, denoising returns None, criterion omits all ``*_dn_*``
loss keys).

Backend: gloo on CPU — the collective ORDER/SIZE mismatch reproduces the
NCCL desync without needing two GPUs (NCCL hangs on numel mismatch; gloo
raises — both are failures, and the fixed code passes on either).

Exit 0 + ``REDUCE_DICT_OK rank=<n>`` on success.
"""

import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

from engine.misc import dist_utils  # noqa: E402


def main() -> None:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group(
        "gloo", init_method="env://", rank=rank, world_size=world_size
    )

    if rank == 0:
        loss = {
            "loss_main": torch.tensor(1.0),
            "loss_boxes": torch.tensor(2.0),
            # DN loss exists only on this rank (peer drew a zero-GT batch).
            "loss_dn_0": torch.tensor(4.0),
        }
    else:
        loss = {
            "loss_main": torch.tensor(3.0),
            "loss_boxes": torch.tensor(6.0),
        }

    out = dist_utils.reduce_dict(loss)

    assert set(out.keys()) == {"loss_main", "loss_boxes", "loss_dn_0"}, (
        rank,
        sorted(out.keys()),
    )
    assert abs(out["loss_main"].item() - 2.0) < 1e-6, (rank, out)
    assert abs(out["loss_boxes"].item() - 4.0) < 1e-6, (rank, out)
    # rank0 contributes 4.0, rank1 contributes a zero pad -> mean over 2 = 2.0
    assert abs(out["loss_dn_0"].item() - 2.0) < 1e-6, (rank, out)

    dist.destroy_process_group()
    print(f"REDUCE_DICT_OK rank={rank}")


if __name__ == "__main__":
    main()
