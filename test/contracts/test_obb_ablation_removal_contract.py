"""Per-axis ablation removal contracts (plan Tasks 3-8).

One ``xfail(strict=False)`` test per removed constructor surface. Each is
RED (xfail) before the corresponding user production edit and XPASS after.
At Task 12 the ``xfail`` markers come off and these become hard guards.

Cheap by design: signature/import checks only, no model construction.

Task mapping:
  Task 3 -> offset_scale_source removed (dfine_utils x2, DEIMTransformer,
            TransformerDecoder, DEIMCriterion)
  Task 4 -> angle_step removed (DEIMTransformer)
  Task 5 -> use_gate_fusion + use_angle_first removed (DEIMTransformer,
            TransformerDecoder); engine.deim.gated_fusion module deleted
  Task 8 -> decoder_angle_encoding removed (DEIMTransformer,
            TransformerDecoder, TransformerDecoderLayer, MSDeformableAttention,
            get_contrastive_denoising_training_group);
            _VALID_DECODER_ANGLE_ENCODINGS module attr gone

See appendix A3/A4/A5/A8 of
``docs/superpowers/plans/2026-08-13-obb-ablation-cleanup.md``.
"""

import importlib
import inspect

import pytest

from engine.deim.deim_criterion import DEIMCriterion
from engine.deim.deim_decoder import (
    DEIMTransformer,
    TransformerDecoder,
    TransformerDecoderLayer,
)
from engine.deim.dfine_decoder import MSDeformableAttention
from engine.deim.dfine_utils import bbox2distance_obb, distance2bbox_obb
from engine.deim.denoising import get_contrastive_denoising_training_group


def _params(fn):
    return set(inspect.signature(fn).parameters)


def test_task3_offset_scale_source_removed():
    offenders = []
    for fn, label in [
        (distance2bbox_obb, "distance2bbox_obb"),
        (bbox2distance_obb, "bbox2distance_obb"),
        (DEIMTransformer.__init__, "DEIMTransformer.__init__"),
        (TransformerDecoder.__init__, "TransformerDecoder.__init__"),
        (DEIMCriterion.__init__, "DEIMCriterion.__init__"),
    ]:
        if "offset_scale_source" in _params(fn):
            offenders.append(label)
    assert not offenders, "offset_scale_source still accepted by: " + ", ".join(offenders)


def test_task4_angle_step_removed():
    assert "angle_step" not in _params(DEIMTransformer.__init__), (
        "angle_step still accepted by DEIMTransformer.__init__"
    )


def test_task5_gate_fusion_and_angle_first_removed():
    offenders = []
    for fn, label in [
        (DEIMTransformer.__init__, "DEIMTransformer.__init__"),
        (TransformerDecoder.__init__, "TransformerDecoder.__init__"),
    ]:
        for key in ("use_gate_fusion", "use_angle_first"):
            if key in _params(fn):
                offenders.append(f"{label}.{key}")
    import_failed = False
    try:
        importlib.import_module("engine.deim.gated_fusion")
    except ImportError:
        import_failed = True
    assert not offenders and import_failed, (
        f"gate/angle-first still present: {offenders}; "
        f"gated_fusion deleted={import_failed}"
    )


def test_task8_decoder_angle_encoding_removed():
    offenders = []
    for fn, label in [
        (DEIMTransformer.__init__, "DEIMTransformer.__init__"),
        (TransformerDecoder.__init__, "TransformerDecoder.__init__"),
        (TransformerDecoderLayer.__init__, "TransformerDecoderLayer.__init__"),
        (MSDeformableAttention.__init__, "MSDeformableAttention.__init__"),
        (get_contrastive_denoising_training_group, "get_contrastive_denoising_training_group"),
    ]:
        key = "decoder_angle_encoding" if "DEIMTransformer" in label else "angle_encoding"
        if key in _params(fn):
            offenders.append(f"{label}.{key}")
    import engine.deim.deim_decoder as dec_mod
    attr_gone = not hasattr(dec_mod, "_VALID_DECODER_ANGLE_ENCODINGS")
    assert not offenders and attr_gone, (
        f"decoder_angle_encoding still present: {offenders}; "
        f"_VALID_DECODER_ANGLE_ENCODINGS gone={attr_gone}"
    )
