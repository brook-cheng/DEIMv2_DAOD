"""Per-axis ablation removal contracts (plan Tasks 3-8).

One test per removed constructor surface. Each is ``xfail(strict=False)``
while the corresponding task is pending: RED (xfail) before the user's
production edit, XPASS after. At Task 12 the ``xfail`` markers come off and
these become hard guards ensuring the removed surfaces never return.

Task 3 (this file, initial): ``offset_scale_source`` removed from the two
``dfine_utils`` OBB geometry functions and from the ``DEIMTransformer`` /
``TransformerDecoder`` / ``DEIMCriterion`` constructors. See appendix A3 of
``docs/superpowers/plans/2026-08-13-obb-ablation-cleanup.md``.
"""

import inspect

from engine.deim.deim_criterion import DEIMCriterion
from engine.deim.deim_decoder import DEIMTransformer, TransformerDecoder
from engine.deim.dfine_utils import bbox2distance_obb, distance2bbox_obb

_REMOVAL_TARGETS_TASK3 = [
    (distance2bbox_obb, "distance2bbox_obb"),
    (bbox2distance_obb, "bbox2distance_obb"),
    (DEIMTransformer.__init__, "DEIMTransformer.__init__"),
    (TransformerDecoder.__init__, "TransformerDecoder.__init__"),
    (DEIMCriterion.__init__, "DEIMCriterion.__init__"),
]


def test_offset_scale_source_removed_from_signatures():
    """Task 3: offset_scale_source must not remain in any of the OBB
    geometry / decoder / criterion signatures."""
    offenders = []
    for fn, label in _REMOVAL_TARGETS_TASK3:
        params = inspect.signature(fn).parameters
        if "offset_scale_source" in params:
            offenders.append(label)
    assert not offenders, (
        "offset_scale_source still accepted by: " + ", ".join(offenders)
    )
