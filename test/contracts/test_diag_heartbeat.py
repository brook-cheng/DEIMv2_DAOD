"""Rank heartbeat for multi-GPU hang diagnosis (T-diag infrastructure).

Symptom under investigation: on 2-GPU torchrun training, ranks drift apart
after some epochs; the lagging rank exceeds the watchdog timeout and the run
aborts. Post-mortem requires per-rank, per-iteration timestamps to find the
exact iteration where the ranks diverged.

``Heartbeat`` appends one JSON line per training iteration to
``<output_dir>/diag/heartbeat_rank{RANK}.jsonl`` when enabled via
``DEIM_DIAG_HEARTBEAT=1`` (plus ``DEIM_DIAG_DIR`` override). Disabled by
default — zero IO, zero effect on normal training.

Run:
    pytest test/contracts/test_diag_heartbeat.py -v
"""

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.misc.diag import Heartbeat  # noqa: E402


def test_disabled_by_default_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.delenv("DEIM_DIAG_HEARTBEAT", raising=False)
    hb = Heartbeat(output_dir=str(tmp_path))
    hb.beat(epoch=0, step=0, extra={})
    assert hb.enabled is False
    assert not (tmp_path / "diag").exists()


def test_enabled_writes_one_json_line_per_beat(tmp_path, monkeypatch):
    monkeypatch.setenv("DEIM_DIAG_HEARTBEAT", "1")
    monkeypatch.setenv("RANK", "1")
    hb = Heartbeat(output_dir=str(tmp_path))
    hb.beat(epoch=3, step=42, extra={"loss": 1.5})
    hb.beat(epoch=3, step=43, extra={})
    path = tmp_path / "diag" / "heartbeat_rank1.jsonl"
    lines = [json.loads(l) for l in path.read_text().splitlines()]
    assert len(lines) == 2
    assert lines[0]["rank"] == 1 and lines[0]["epoch"] == 3 and lines[0]["step"] == 42
    assert lines[0]["loss"] == 1.5
    assert lines[1]["step"] == 43
    assert all("ts" in l and "global_step" in l for l in lines)


def test_rank_defaults_to_zero_and_dir_override(tmp_path, monkeypatch):
    monkeypatch.setenv("DEIM_DIAG_HEARTBEAT", "1")
    monkeypatch.delenv("RANK", raising=False)
    override = tmp_path / "elsewhere"
    monkeypatch.setenv("DEIM_DIAG_DIR", str(override))
    hb = Heartbeat(output_dir=str(tmp_path))
    hb.beat(epoch=0, step=0, extra={})
    assert (override / "heartbeat_rank0.jsonl").exists()
