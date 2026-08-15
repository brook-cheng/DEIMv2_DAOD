"""Runnable OBB diagnostic scripts must reference existing configs (plan Task 12).

The OBB ablation cleanup deleted the rep1/rep2/fused/AFP/mangle/offset-post
ablation configs and the old naming-era configs (``sp_ft_rep1.yml``,
``deimv2_obb_sp_dlzdt_anglerep0_p*.yml``, plain ``synthetic_exp_020.yml``).
Every ``configs/...yml`` reference in the runnable diagnostic scripts must
therefore resolve to a file on disk. Checkpoint / output paths under
``outputs/`` are historical run data and are intentionally out of scope.
``train.py`` keeps its upstream DEIM argparse default, so only the deleted
OBB names are contract-checked there.

This is a text/OS-level contract: no GPU or model code is executed.

Run:
    pytest test/test_obb_runnable_config_references.py -v
"""

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

#: Runnable OBB diagnostic scripts scanned for config references.
RUNNABLE_DIAG_SCRIPTS = (
    "test/tool_deimv2_obb_infer.py",
    "test/tool_debug_decoder.py",
    "test/bisect_eval.py",
    "test/model_health_diag.py",
    "test/diagnose_hungarian_matching.py",
    "test/test_cdn_inspect.py",
    "test/test_infer_diag.py",
    # shell runners reference the same config tree
    "tools/dataset/gen_synthetic_dataset/run_synthetic_training.sh",
    "tools/dataset/gen_synthetic_dataset/run_synthetic_A.sh",
    "tools/dataset/gen_synthetic_dataset/run_synthetic_B.sh",
    "scripts/single_gpu_train.sh",
    "scripts/single_gpu_val.sh",
    "scripts/multi_gpu_train.sh",
)

#: train entry point default is an upstream DEIM path outside this cleanup;
#: only the deleted OBB config names are contract-checked there.
TRAIN_PY = "tools/train/train.py"

#: Tokens of configs deleted by the OBB ablation cleanup (plan Tasks 5-11).
_DELETED_OBB_TOKENS = (
    "syn_ablation_afp",
    "syn_ablation_fused",
    "syn_ablation_mangle",
    "abl_shifted.yml",
    "abl_mangle.yml",
    "abl_offset_post.yml",
    "abl_rep1.yml",
    "abl_rep2.yml",
    "abl_rep2_fused.yml",
    "abl_rep3_afp.yml",
    "abl_rep3_fused.yml",
    "sp_ft_rep1.yml",
    "deimv2_obb_sp_dlzdt_anglerep0_p",
    "synthetic_exp_020.yml",
)

_CONFIG_REF_RE = re.compile(r"configs/[^\s'\"`]+\.yml")


def _read(rel_path: str) -> str:
    with open(os.path.join(ROOT, rel_path), encoding="utf-8") as fh:
        return fh.read()


def test_diagnostic_scripts_reference_existing_configs():
    for script in RUNNABLE_DIAG_SCRIPTS:
        text = _read(script)
        refs = sorted(set(_CONFIG_REF_RE.findall(text)))
        assert refs, f"{script}: expected at least one config reference"
        for ref in refs:
            if "{" in ref:
                # f-string template (e.g. anrep{ANGLE_REP}); resolved at
                # runtime from a retained module constant, not statically
                # checkable here.
                continue
            assert os.path.isfile(os.path.join(ROOT, ref)), (
                f"{script}: config reference {ref!r} does not exist on disk"
            )


def test_train_py_has_no_deleted_config_hint():
    text = _read(TRAIN_PY)
    for token in _DELETED_OBB_TOKENS:
        assert token not in text, (
            f"{TRAIN_PY}: stale reference to deleted config {token!r}"
        )
