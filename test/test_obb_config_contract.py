"""OBB YAML configuration contract tests (plan Task 2).

Two layers:

1. **Enforcement net** (hard, green now): every key in the
   ``DEIMTransformer`` / ``DEIMCriterion`` section of every active OBB
   config must be an accepted constructor kwarg
   (``engine/core/workspace.py:176-178`` has signature filtering
   commented out, so a stale key raises ``TypeError`` at build). This
   turns RED the moment a cleanup task deletes a constructor parameter
   while a YAML key still references it -- the signal that keeps
   Tasks 3-8 honest. A documented pre-existing stale key
   (``decouple_angle``) is tolerated until Task 11 removes it.

2. **Cleanup completeness** (xfail while Tasks 3-8 are in progress):
   asserts removed ablation keys are gone and ``angle_rep`` is in
   ``{0, 3}``. These legitimately fail pre-cleanup; their failure
   message is the deletion manifest. Drop the ``xfail`` once complete.

Run:
    pytest test/test_obb_config_contract.py -v
"""

import inspect
import os
import sys
from pathlib import Path

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import engine.deim  # noqa: E402,F401  triggers @register() population
from engine.core.yaml_config import YAMLConfig  # noqa: E402
from engine.deim.deim_criterion import DEIMCriterion  # noqa: E402
from engine.deim.deim_decoder import DEIMTransformer  # noqa: E402

REPO = Path(ROOT)
AUDITED_SECTIONS = ("DEIMTransformer", "DEIMCriterion")

ACCEPTED_KWARGS = {
    "DEIMTransformer": set(inspect.signature(DEIMTransformer.__init__).parameters) - {"self"},
    "DEIMCriterion": set(inspect.signature(DEIMCriterion.__init__).parameters) - {"self"},
}

# Pre-existing stale key unrelated to this cleanup (only
# jyz/sp_ft_rep0.yml:97 defines it; it propagates via __include__).
# Task 11 removes it; until then the enforcement net tolerates it so the
# cleanup's own signals stay clean.
KNOWN_PRE_CLEANUP_STALE = {"decouple_angle"}

REMOVED_ABLATION_KEYS = {
    "offset_scale_source",
    "use_gate_fusion",
    "angle_step",
    "use_angle_first",
    "decoder_angle_encoding",
}


def _obb_config_paths():
    custom = [
        p
        for p in sorted((REPO / "configs" / "custom_obb").rglob("*.yml"))
        if "provenance" not in p.parts
    ]
    app_obb = [
        REPO / "configs" / "app" / "presets" / "deimv2_dinov3_sp_obb.yml",
    ]
    paths = custom + [p for p in app_obb if p.exists()]
    return paths, [str(p.relative_to(REPO)) for p in paths]


_PATHS, _IDS = _obb_config_paths()


def _resolved_sections(path):
    gc = YAMLConfig(str(path)).global_cfg
    out = {}
    for section in AUDITED_SECTIONS:
        v = gc.get(section)
        if isinstance(v, dict):
            out[section] = v
    return out


def _public_keys(resolved):
    return [k for k in resolved if not k.startswith("_") and k != "type"]


# ---------------------------------------------------------------------------
# Enforcement net
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", _PATHS, ids=_IDS)
def test_yaml_section_keys_match_constructor_signature(path):
    """Every public key in DEIMTransformer/DEIMCriterion must be an accepted
    constructor kwarg, except the documented pre-existing stale key."""
    problems = []
    for section, resolved in _resolved_sections(path).items():
        accepted = ACCEPTED_KWARGS[section]
        for key in _public_keys(resolved):
            if key in KNOWN_PRE_CLEANUP_STALE:
                continue
            if key not in accepted:
                problems.append(f"{section}.{key}")
    assert not problems, (
        f"{path.relative_to(REPO)}: unknown constructor keys -> {problems}"
    )


# ---------------------------------------------------------------------------
# Cleanup completeness (in progress)
# ---------------------------------------------------------------------------


def _collect_violations():
    key_violations = []
    rep_violations = []
    for path in _PATHS:
        rel = path.relative_to(REPO)
        sections = _resolved_sections(path)
        for section, resolved in sections.items():
            for key in _public_keys(resolved):
                if key in REMOVED_ABLATION_KEYS:
                    key_violations.append(f"{rel}:{section}.{key}")
        dt = sections.get("DEIMTransformer", {})
        if "angle_rep" in dt:
            val = dt["angle_rep"]
            if val not in (0, 3) or isinstance(val, bool):
                rep_violations.append(f"{rel}:DEIMTransformer.angle_rep={val!r}")
    return key_violations, rep_violations


@pytest.mark.xfail(
    strict=False,
    reason="Cleanup in progress (plan Tasks 3-8); fails until removed ablation "
    "keys are gone from every OBB config. The list below is the manifest.",
)
def test_no_removed_ablation_keys_in_any_obb_config():
    key_violations, _ = _collect_violations()
    assert not key_violations, (
        "OBB configs still reference removed ablation keys:\n  "
        + "\n  ".join(sorted(set(key_violations)))
    )


@pytest.mark.xfail(
    strict=False,
    reason="Cleanup in progress (plan Tasks 6-7); fails until every OBB config "
    "uses angle_rep in {0, 3}.",
)
def test_angle_rep_is_only_0_or_3_in_any_obb_config():
    _, rep_violations = _collect_violations()
    assert not rep_violations, (
        "OBB configs still use angle_rep outside {0, 3}:\n  "
        + "\n  ".join(sorted(set(rep_violations)))
    )
