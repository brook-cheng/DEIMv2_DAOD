"""App preset scheduler sanity (acceptance finding: dead-LR warmup trap).

This fork's ``FlatCosineLRScheduler`` multiplies the YAML ``warmup_iter`` by
iters-per-epoch — the config value is EPOCH-scale. Upstream HBB presets
carried ``warmup_iter: 2000`` (upstream's ITERATION semantics), which made
warmup 30x longer than the entire run: lr stayed on the squared warmup ramp
at ~1e-10 for every epoch and mAP never left zero (T01 acceptance finding).

Every app preset chain that selects the flatcosine scheduler must keep
``warmup_iter`` epoch-scale (strictly below ``epoches``).

Run:
    pytest test/contracts/test_app_preset_scheduler_sanity.py -v
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from engine.core import yaml_utils  # noqa: E402
from engine.core.yaml_utils import load_config  # noqa: E402

PRESET_CHAINS = (
    "configs/app/presets/deimv2_dinov3_vits16_freeze_hbb.yml",
    "configs/app/presets/deimv2_dinov3_vits16_freeze_obb.yml",
    "configs/app/presets/deimv2_dinov3_sp_obb.yml",
)


def test_flatcosine_warmup_is_epoch_scale():
    for preset in PRESET_CHAINS:
        yaml_utils.load_config.__defaults__ = ({},)  # isolate accumulator
        merged = load_config(os.path.join(ROOT, preset))
        if merged.get("lrsheduler") != "flatcosine":
            continue
        warmup = merged.get("warmup_iter")
        epoches = merged.get("epoches")
        assert isinstance(warmup, int) and isinstance(epoches, int), (
            f"{preset}: flatcosine needs int warmup_iter/epoches, "
            f"got {warmup!r}/{epoches!r}"
        )
        assert warmup < epoches, (
            f"{preset}: warmup_iter={warmup} must stay EPOCH-scale (<{epoches}); "
            "this fork multiplies it by iters/epoch — upstream's iteration-scale "
            "2000 pins lr at ~1e-10 for the whole run (T01 acceptance finding)"
        )
