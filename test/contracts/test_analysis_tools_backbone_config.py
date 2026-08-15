"""Analysis tools must build the DINOv3STAs backbone from a retained YAML.

``tools/analysis/grad_cam.py`` and ``tools/analysis/feature_similarity.py``
historically read ``DEIMV2_X_CFG["DINOv3STAs"]`` from the retired
``deim_wapper`` package — a key that no longer existed in that dict
(``KeyError`` at runtime). They now derive the backbone kwargs from the
``DINOv3STAs`` section of a retained training YAML. This contract locks both
halves: the tools stay off ``deim_wapper``, and the YAML section stays
constructible.

Run:
    pytest test/test_analysis_tools_backbone_config.py -v
"""

import inspect
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from engine.backbone.dinov3_adapter import DINOv3STAs  # noqa: E402
from engine.core.yaml_config import YAMLConfig  # noqa: E402

ANALYSIS_TOOLS = (
    "tools/analysis/grad_cam.py",
    "tools/analysis/feature_similarity.py",
)

BACKBONE_CONFIG = "configs/custom/deimv2_dinov3_vits16p_freeze_test_eiou.yml"


def test_analysis_tools_do_not_reference_deim_wapper():
    for tool in ANALYSIS_TOOLS:
        with open(os.path.join(ROOT, tool), encoding="utf-8") as fh:
            text = fh.read()
        assert "deim_wapper" not in text, f"{tool} still references deim_wapper"
        assert "DEIMV2_X_CFG" not in text, f"{tool} still uses the retired config dict"
        assert "deimv2_dinov3_vits16p_freeze_test_eiou.yml" in text, (
            f"{tool} must take backbone kwargs from the retained training YAML"
        )


def test_retained_yaml_backbone_section_is_constructible():
    section = YAMLConfig(os.path.join(ROOT, BACKBONE_CONFIG)).global_cfg["DINOv3STAs"]
    accepted = set(inspect.signature(DINOv3STAs.__init__).parameters) - {"self"}
    public = {k for k in section if not k.startswith("_") and k != "type"}
    unknown = public - accepted
    assert not unknown, (
        f"{BACKBONE_CONFIG}: DINOv3STAs section has keys the constructor "
        f"does not accept: {sorted(unknown)}"
    )
    assert "name" in public and "weights_path" in public, (
        "the section must keep name/weights_path so tools can override the "
        "checkpoint path"
    )


def test_tools_apply_section_via_kwarg_strip():
    pattern = re.compile(r"k != \"type\"")
    for tool in ANALYSIS_TOOLS:
        with open(os.path.join(ROOT, tool), encoding="utf-8") as fh:
            text = fh.read()
        assert pattern.search(text), (
            f"{tool} must strip the registry 'type' key before DINOv3STAs(**kwargs)"
        )
