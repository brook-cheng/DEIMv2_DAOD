"""Multi-GPU diagnosis helpers (env-gated, zero impact when disabled).

Symptom under investigation: 2-GPU torchrun runs drift apart after several
epochs and abort on the lagging rank's watchdog timeout. Reproduction and
root-cause need per-rank iteration timestamps (which rank stalled, at which
iteration) plus NCCL flight-recorder dumps — both wired here.

Env switches (all default OFF):

* ``DEIM_DIAG_HEARTBEAT=1`` — per-iteration JSONL heartbeat per rank, see
  :class:`Heartbeat`.
* ``DEIM_DIAG_DIR`` — override the heartbeat directory (default
  ``<output_dir>/diag``).
* ``DEIM_DIAG_USR1=1`` — register SIGUSR1 to dump all thread stacks; while
  hung, ``kill -USR1 <pid>`` captures the in-flight state of every rank.
"""

from __future__ import annotations

import faulthandler
import json
import os
import signal
import time


class Heartbeat:
    """Append one JSON line per training iteration, per rank.

    Post-hoc, align the two ranks' ``global_step`` timestamps: the first
    iteration where the inter-rank gap explodes is the divergence point.
    """

    def __init__(self, output_dir: str) -> None:
        self.enabled = os.getenv("DEIM_DIAG_HEARTBEAT", "") == "1"
        if not self.enabled:
            self._fh = None
            return
        rank = os.getenv("RANK", "0")
        diag_dir = os.getenv("DEIM_DIAG_DIR") or os.path.join(output_dir, "diag")
        os.makedirs(diag_dir, exist_ok=True)
        self._fh = open(
            os.path.join(diag_dir, f"heartbeat_rank{rank}.jsonl"),
            "a",
            encoding="utf-8",
            buffering=1,
        )
        self._rank = int(rank)

    def beat(self, epoch: int, step: int, extra: dict) -> None:
        if self._fh is None:
            return
        record = {
            "ts": time.time(),
            "rank": self._rank,
            "epoch": epoch,
            "step": step,
            "global_step": None,
            **extra,
        }
        record["global_step"] = extra.get("global_step", epoch * 10**6 + step)
        self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def enable_usr1_stack_dump() -> None:
    """SIGUSR1 → dump all thread stacks to stderr (external trigger on hang)."""
    if os.getenv("DEIM_DIAG_USR1", "") != "1":
        return
    faulthandler.register(signal.SIGUSR1, all_threads=True, chain=False)
