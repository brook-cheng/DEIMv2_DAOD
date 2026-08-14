"""OBB YAML configuration contract tests (plan Task 2).

Two layers:

1. **Enforcement net** (hard): every key in the
   ``DEIMTransformer`` / ``DEIMCriterion`` section of every active OBB
   config must be an accepted constructor kwarg
   (``engine/core/workspace.py:176-178`` has signature filtering
   commented out, so a stale key raises ``TypeError`` at build). This
   turns RED the moment a cleanup task deletes a constructor parameter
   while a YAML key still references it -- the signal that keeps
   Tasks 3-8 honest.

2. **Forbidden keys** (hard): a permanently-banned set of keys that
   must never reappear in any OBB config, plus the removed ablation
   keys. ``decouple_angle`` is among the forbidden keys: it was a
   stale YAML-only switch that the constructor never accepted, and
   the process-global registry makes it a pollution hazard if it
   appears in any config parsed before another config is constructed.

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

# Keys that must never appear in any OBB config's DEIMTransformer /
# DEIMCriterion section.  ``decouple_angle`` was a stale YAML-only
# switch whose only occurrence (jyz/sp_ft_rep0.yml) propagated via
# ``__include__``; the constructor never accepted it, and the
# process-global registry makes it a pollution hazard.  Once removed
# it must never return.
FORBIDDEN_KEYS = {
    "decouple_angle",
}

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
    constructor kwarg."""
    problems = []
    for section, resolved in _resolved_sections(path).items():
        accepted = ACCEPTED_KWARGS[section]
        for key in _public_keys(resolved):
            if key not in accepted:
                problems.append(f"{section}.{key}")
    assert not problems, (
        f"{path.relative_to(REPO)}: unknown constructor keys -> {problems}"
    )


# ---------------------------------------------------------------------------
# Cleanup completeness (hard guards — cleanup complete)
# ---------------------------------------------------------------------------


def _collect_violations():
    key_violations = []
    rep_violations = []
    forbidden_violations = []
    for path in _PATHS:
        rel = path.relative_to(REPO)
        sections = _resolved_sections(path)
        for section, resolved in sections.items():
            for key in _public_keys(resolved):
                if key in REMOVED_ABLATION_KEYS:
                    key_violations.append(f"{rel}:{section}.{key}")
                if key in FORBIDDEN_KEYS:
                    forbidden_violations.append(f"{rel}:{section}.{key}")
        dt = sections.get("DEIMTransformer", {})
        if "angle_rep" in dt:
            val = dt["angle_rep"]
            if val not in (0, 3) or isinstance(val, bool):
                rep_violations.append(f"{rel}:DEIMTransformer.angle_rep={val!r}")
    return key_violations, rep_violations, forbidden_violations


def test_no_removed_ablation_keys_in_any_obb_config():
    key_violations, _, _ = _collect_violations()
    assert not key_violations, (
        "OBB configs still reference removed ablation keys:\n  "
        + "\n  ".join(sorted(set(key_violations)))
    )


def test_no_forbidden_keys_in_any_obb_config():
    _, _, forbidden_violations = _collect_violations()
    assert not forbidden_violations, (
        "OBB configs contain permanently-forbidden keys:\n  "
        + "\n  ".join(sorted(set(forbidden_violations)))
    )


def test_angle_rep_is_only_0_or_3_in_any_obb_config():
    _, rep_violations, _ = _collect_violations()
    assert not rep_violations, (
        "OBB configs still use angle_rep outside {0, 3}:\n  "
        + "\n  ".join(sorted(set(rep_violations)))
    )


# ---------------------------------------------------------------------------
# decouple_angle registry-pollution contract (plan Task 2)
# ---------------------------------------------------------------------------

_SP_FT_REP0 = REPO / "configs" / "custom_obb" / "jyz" / "sp_ft_rep0.yml"
_ABL_REP3 = REPO / "configs" / "custom_obb" / "dlzdt" / "ablation" / "abl_rep3.yml"


def test_sp_ft_rep0_has_no_decouple_angle():
    """The former sole source of ``decouple_angle`` must be clean."""
    assert _SP_FT_REP0.exists(), f"missing config: {_SP_FT_REP0}"
    sections = _resolved_sections(_SP_FT_REP0)
    dt = sections.get("DEIMTransformer", {})
    assert "decouple_angle" not in dt, (
        f"sp_ft_rep0.yml still carries the stale decouple_angle key: {dt}"
    )


def test_parsing_sp_ft_rep0_does_not_pollute_abl_rep3_kwargs():
    """Parsing sp_ft_rep0.yml before abl_rep3.yml must not inject
    ``decouple_angle`` (or any other unexpected kwarg) into abl_rep3's
    DEIMTransformer section via the process-global registry."""
    assert _ABL_REP3.exists(), f"missing config: {_ABL_REP3}"

    # Parse the former source of the stale key first.
    YAMLConfig(str(_SP_FT_REP0)).global_cfg

    # Then inspect abl_rep3's resolved DEIMTransformer section.
    sections = _resolved_sections(_ABL_REP3)
    dt = sections.get("DEIMTransformer", {})
    assert "decouple_angle" not in dt, (
        f"decouple_angle leaked into abl_rep3 via global registry: {dt}"
    )
    # angle_rep must be exactly int 3 (not float, not bool).
    assert dt.get("angle_rep") == 3
    assert isinstance(dt.get("angle_rep"), int)
    assert not isinstance(dt.get("angle_rep"), bool)
