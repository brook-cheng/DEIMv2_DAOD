"""Compare-tool head sizing: the checkpoint is ground truth (acceptance T13).

``run_infer.py`` historically sized model heads from ``classes.txt``. The
synthetic-ellipse dataset is 3-class (r/g/b) while legacy density configs
carried ``num_classes: 15`` — the mismatched checkpoint then failed
``load_state_dict`` with size errors on every score head. The research tool
must instead derive ``num_classes`` from the checkpoint's actual
``decoder.dec_score_head.0.weight`` shape when one is provided, so any
historical checkpoint loads regardless of its config's class count.

Run:
    pytest test/eval/test_compare_head_sizing.py -v
"""

import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tools.compare.core import resolve_num_classes  # noqa: E402


def _ckpt(num_classes: int) -> dict:
    return {
        "model": {
            "decoder.dec_score_head.0.weight": torch.zeros(num_classes, 256),
            "decoder.enc_score_head.weight": torch.zeros(num_classes, 256),
        }
    }


def test_checkpoint_head_shape_wins_over_classes_txt_count():
    """A 15-class checkpoint must resolve to 15 even when classes.txt has 3."""
    assert resolve_num_classes(_ckpt(15), num_classes_from_txt=3) == 15


def test_txt_count_used_when_no_checkpoint_head():
    """Bare state (no head key) falls back to the classes.txt count."""
    assert resolve_num_classes({"model": {}}, num_classes_from_txt=3) == 3


def test_ema_module_layout_resolves_too():
    ckpt = {
        "ema": {
            "module": {
                "decoder.dec_score_head.0.weight": torch.zeros(7, 256),
            }
        }
    }
    assert resolve_num_classes(ckpt, num_classes_from_txt=3) == 7


def test_mismatch_is_detected_not_silenced():
    """When both sources exist and disagree, the checkpoint wins — but the
    helper reports the txt count so callers can warn about the discrepancy."""
    txt_n, ckpt_n = 3, 15
    resolved = resolve_num_classes(_ckpt(ckpt_n), num_classes_from_txt=txt_n)
    assert resolved == ckpt_n and resolved != txt_n
