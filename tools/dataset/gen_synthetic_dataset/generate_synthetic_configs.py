#!/usr/bin/env python3
"""Generate 7 standalone YAML training configs for synthetic ellipse density experiments.

Resolves all __include__ references from deimv2_obb_sp.yml, applies density-specific
overrides, and writes self-contained config files (no __include__ blocks).
"""

import copy
import sys
from pathlib import Path

import yaml


CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"
OUTPUT_DIR = CONFIG_DIR / "custom_obb"

DENSITIES = [1, 2, 5, 10, 20, 50, 100]

# Files included by deimv2_obb_sp.yml (merge order: earlier files are base)
INCLUDED_FILES = [
    CONFIG_DIR / "runtime.yml",
    CONFIG_DIR / "base" / "dataloader.yml",
    CONFIG_DIR / "base" / "optimizer.yml",
    CONFIG_DIR / "dataset" / "dota_detection.yml",
    CONFIG_DIR / "custom_obb" / "dataset_common.yml",
    CONFIG_DIR / "custom_obb" / "deimv2_obb_common.yml",
]

TEMPLATE_PATH = CONFIG_DIR / "custom_obb" / "deimv2_obb_sp.yml"


def load_yaml(path: Path) -> dict:
    with open(path, "r") as fh:
        data = yaml.safe_load(fh)
    return data if data is not None else {}


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base in place. Lists are replaced, not merged."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            deep_merge(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def apply_density_overrides(config: dict, density: int) -> dict:
    """Apply density-specific parameter changes in place."""
    d = f"{density:03d}"
    root = "/mnt/d/project_data/model_test/deimv2_obb_train_data/synthetic_ellipse"

    config["output_dir"] = f"./outputs/synthetic_exp_{d}"
    config["comet_project_name"] = "deimv2_obb_synthetic"
    config["eval_spatial_size"] = [256, 256]

    # Training schedule
    config["epoches"] = 30
    config["flat_epoch"] = 18
    config["no_aug_epoch"] = 3

    # ── Train dataloader ──────────────────────────────────────────────
    tdl = config.setdefault("train_dataloader", {})

    # collate_fn: multiscale base_size adapted to 256
    tdl.setdefault("collate_fn", {})["base_size"] = 256
    tdl["total_batch_size"] = 6

    tds = tdl.setdefault("dataset", {})
    tds["img_folder"] = f"{root}/density_{d}/train"
    tds["ann_folder"] = f"{root}/density_{d}/train"
    tds["classes_file"] = f"{root}/classes.txt"

    # Patch transforms: OBBResize & OBBConvertBoxes → 256×256
    transforms_ops = (
        tds.get("transforms", {}).get("ops", [])
        if isinstance(tds.get("transforms"), dict)
        else []
    )
    for op in transforms_ops:
        t = op.get("type", "")
        if t == "OBBResize":
            op["size"] = [256, 256]
        elif t == "OBBConvertBoxes":
            op["img_size"] = [256, 256]

    # ── Val dataloader ────────────────────────────────────────────────
    vdl = config.setdefault("val_dataloader", {})
    vdl["total_batch_size"] = 4

    vds = vdl.setdefault("dataset", {})
    vds["img_folder"] = f"{root}/density_{d}/val"
    vds["ann_folder"] = f"{root}/density_{d}/val"
    vds["classes_file"] = f"{root}/classes.txt"

    val_ops = (
        vds.get("transforms", {}).get("ops", [])
        if isinstance(vds.get("transforms"), dict)
        else []
    )
    for op in val_ops:
        t = op.get("type", "")
        if t == "OBBResize":
            op["size"] = [256, 256]
        elif t == "OBBConvertBoxes":
            op["img_size"] = [256, 256]

    return config


def main() -> None:
    # 1. Merge all included files
    merged = {}
    for path in INCLUDED_FILES:
        deep_merge(merged, load_yaml(path))

    # 2. Load template and strip __include__, then merge on top
    template = load_yaml(TEMPLATE_PATH)
    template.pop("__include__", None)
    deep_merge(merged, template)

    # 3. For each density, copy merged config, apply overrides, write
    generated = []
    for density in DENSITIES:
        config = copy.deepcopy(merged)
        apply_density_overrides(config, density)

        out_name = f"synthetic_exp_{density:03d}.yml"
        out_path = OUTPUT_DIR / out_name

        with open(out_path, "w") as fh:
            yaml.dump(config, fh, default_flow_style=False, allow_unicode=True, sort_keys=False)

        generated.append(out_name)

    print(f"Generated {len(generated)} configs:")
    for name in generated:
        print(f"  {name}")

    print("\nDone. Output directory:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
