#!/usr/bin/env python3
"""Generate 7 standalone YAML training configs for synthetic ellipse density experiments.

Builds every density variant from a single *retained standalone* synthetic
template (``synthetic_exp_001.yml``) that already has no ``__include__`` block.
The tool only applies density-specific overrides on top of that self-contained
template, so it never depends on a deleted base config and never re-introduces
removed aliases or ``__include__`` chains.
"""

import copy
from pathlib import Path
from typing import TypeAlias, cast

import yaml

# A synthetic config is a recursive JSON-like mapping. Every value is one of the
# scalar/leaf types, a homogeneous list of values, or a nested mapping.
ConfigValue: TypeAlias = (
    bool | int | float | str | None | list["ConfigValue"] | dict[str, "ConfigValue"]
)
Config: TypeAlias = dict[str, ConfigValue]

# This file lives at tools/dataset/gen_synthetic_dataset/generate_synthetic_configs.py
# -> parents[3] is the repository root.
REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = REPO_ROOT / "configs"
OUTPUT_DIR = CONFIG_DIR / "custom_obb" / "synthetic_configs"

DENSITIES: list[int] = [1, 2, 5, 10, 20, 50, 100]

# Canonical retained standalone template (density 001, fully self-contained,
# 256x256). Every density variant is derived from it via apply_density_overrides.
TEMPLATE_PATH = OUTPUT_DIR / "synthetic_exp_001.yml"

# Spatial size shared by eval_spatial_size and the OBBResize/OBBConvertBoxes ops.
_IMG_SIZE = 256


def _sub(config: Config, key: str) -> Config:
    """Return ``config[key]`` as a nested mapping, creating it empty if absent.

    Mirrors ``dict.setdefault(key, {})`` but returns a typed ``Config`` so nested
    subscript writes stay statically typed.
    """
    value = config.get(key)
    if isinstance(value, dict):
        return value
    child: Config = {}
    config[key] = child
    return child


def _ops(parent: Config, key: str) -> list[Config]:
    """Return the list of transform-op mappings stored under ``parent[key].ops``.

    Returns an empty list when the node or its ``ops`` field is missing or has an
    unexpected shape, so callers can iterate without further narrowing.
    """
    node = parent.get(key)
    if not isinstance(node, dict):
        return []
    raw_ops = node.get("ops")
    if not isinstance(raw_ops, list):
        return []
    ops: list[Config] = []
    for op in raw_ops:
        if isinstance(op, dict):
            ops.append(op)
    return ops


def load_yaml(path: Path) -> Config:
    """Load a YAML file and return it as a typed ``Config`` mapping.

    ``yaml.safe_load`` is typed as returning ``Any``; this function is the trust
    boundary that narrows it to ``object``, validates it is a mapping, and
    returns it as a typed ``Config``.
    """
    with open(path) as fh:
        loaded = cast(object, yaml.safe_load(fh))
    if not isinstance(loaded, dict):
        raise ValueError(f"{path}: expected a YAML mapping at the top level")
    return cast(Config, loaded)


def apply_density_overrides(config: Config, density: int) -> Config:
    """Apply density-specific parameter changes in place."""
    d = f"{density:03d}"
    root = "/mnt/d/project_data/model_test/deimv2_obb_train_data/synthetic_ellipse"
    img_size: list[ConfigValue] = [_IMG_SIZE, _IMG_SIZE]

    config["output_dir"] = f"./outputs/synthetic_exp_{d}"
    config["comet_project_name"] = "deimv2_obb_synthetic"
    config["eval_spatial_size"] = img_size

    # Training schedule
    config["epoches"] = 30
    config["flat_epoch"] = 18
    config["no_aug_epoch"] = 3

    # ── Train dataloader ──────────────────────────────────────────────
    tdl = _sub(config, "train_dataloader")

    # collate_fn: multiscale base_size adapted to 256
    _sub(tdl, "collate_fn")["base_size"] = _IMG_SIZE
    tdl["total_batch_size"] = 6

    tds = _sub(tdl, "dataset")
    tds["img_folder"] = f"{root}/density_{d}/train"
    tds["ann_folder"] = f"{root}/density_{d}/train"
    tds["classes_file"] = f"{root}/classes.txt"

    # Patch transforms: OBBResize & OBBConvertBoxes → 256×256
    for op in _ops(tds, "transforms"):
        op_type = op.get("type", "")
        if op_type == "OBBResize":
            op["size"] = list(img_size)
        elif op_type == "OBBConvertBoxes":
            op["img_size"] = list(img_size)

    # ── Val dataloader ────────────────────────────────────────────────
    vdl = _sub(config, "val_dataloader")
    vdl["total_batch_size"] = 4

    vds = _sub(vdl, "dataset")
    vds["img_folder"] = f"{root}/density_{d}/val"
    vds["ann_folder"] = f"{root}/density_{d}/val"
    vds["classes_file"] = f"{root}/classes.txt"

    for op in _ops(vds, "transforms"):
        op_type = op.get("type", "")
        if op_type == "OBBResize":
            op["size"] = list(img_size)
        elif op_type == "OBBConvertBoxes":
            op["img_size"] = list(img_size)

    return config


def main() -> None:
    # Load the standalone template (no __include__ to resolve) and derive every
    # density variant from it.
    template = load_yaml(TEMPLATE_PATH)

    generated: list[str] = []
    for density in DENSITIES:
        config = apply_density_overrides(copy.deepcopy(template), density)

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
