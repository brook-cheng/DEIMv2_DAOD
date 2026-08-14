"""Synthetic config generator contract tests (plan Task 3 stale-consumer cleanup).

The generator historically resolved ``__include__`` chains rooted at the now
deleted ``deimv2_obb_sp.yml``. After the cleanup it must build every density
config from a single *retained standalone* synthetic template (one that already
has no ``__include__`` block), so the tool never depends on a deleted file and
never re-introduces removed aliases.

Run::

    pytest test/test_synthetic_config_generator.py -v
"""

import copy
import importlib.util
import sys
import types
from pathlib import Path
from typing import Callable, Protocol, TypeAlias, runtime_checkable

import yaml

ROOT = Path(__file__).resolve().parent.parent
GEN_PATH = ROOT / "tools" / "dataset" / "gen_synthetic_dataset" / "generate_synthetic_configs.py"
CONFIG_DIR = ROOT / "configs"
CUSTOM_OBB = CONFIG_DIR / "custom_obb"
SYNTH_DIR = CUSTOM_OBB / "synthetic_configs"

DELETED_TEMPLATE = CUSTOM_OBB / "deimv2_obb_sp.yml"

# Recursive type of a synthetic config (mirrors the generator's Config alias).
ConfigValue: TypeAlias = (
    bool | int | float | str | None | list["ConfigValue"] | dict[str, "ConfigValue"]
)
Config: TypeAlias = dict[str, ConfigValue]


@runtime_checkable
class _Generator(Protocol):
    """Static interface of the dynamically-loaded generator module.

    The generator is loaded by file path (it is not an importable package), so
    attribute access on the runtime module is ``Any``. This Protocol declares the
    module's real public surface so the test body stays statically typed.
    """

    TEMPLATE_PATH: Path
    DENSITIES: list[int]
    load_yaml: Callable[[Path], Config]
    apply_density_overrides: Callable[[Config, int], Config]


def _load_generator() -> types.ModuleType:
    """Load the generator module by file path (it is not an importable package)."""
    spec = importlib.util.spec_from_file_location("generate_synthetic_configs", GEN_PATH)
    if spec is None:
        raise RuntimeError(f"could not build module spec for {GEN_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["generate_synthetic_configs"] = module
    if spec.loader is None:
        raise RuntimeError(f"module spec for {GEN_PATH} has no loader")
    spec.loader.exec_module(module)
    return module


def _child(config: Config, key: str) -> Config:
    """Assert and return a nested mapping stored under ``config[key]``."""
    value = config.get(key)
    assert isinstance(value, dict), f"{key!r} missing or not a mapping"
    return value


# Narrow the runtime-loaded module to its declared Protocol surface with an
# assertion (no cast, no ignore). The annotation makes _gen typed inside every
# test body, not just at module level.
_module = _load_generator()
assert isinstance(_module, _Generator), "generator module is missing required members"
_gen: _Generator = _module

TEMPLATE_PATH: Path = _gen.TEMPLATE_PATH
DENSITIES: list[int] = _gen.DENSITIES


# --------------------------------------------------------------------------- #
# Template contract
# --------------------------------------------------------------------------- #
def test_template_path_resolves_to_an_existing_file():
    """RED guard: the generator's template must point at a file that still exists."""
    assert TEMPLATE_PATH.exists(), f"TEMPLATE_PATH does not exist: {TEMPLATE_PATH}"


def test_template_path_is_not_the_deleted_deimv2_obb_sp():
    """The deleted ``deimv2_obb_sp.yml`` must never be restored as the template."""
    assert TEMPLATE_PATH != DELETED_TEMPLATE
    assert "deimv2_obb_sp.yml" not in TEMPLATE_PATH.name


def test_template_is_a_standalone_retained_synthetic_config():
    """The template must be a retained standalone config (no ``__include__``)."""
    assert SYNTH_DIR in TEMPLATE_PATH.parents, (
        f"template must live under synthetic_configs/, got {TEMPLATE_PATH}"
    )
    data = _gen.load_yaml(TEMPLATE_PATH)
    assert "__include__" not in data, "template must be standalone (no __include__)"


# --------------------------------------------------------------------------- #
# Density-override behavior (no file IO on retained configs)
# --------------------------------------------------------------------------- #
def test_density_overrides_mutate_density_specific_fields():
    """Overrides must rewrite output dir, paths, and sizes per density."""
    base = _gen.load_yaml(TEMPLATE_PATH)

    density = 50
    cfg = _gen.apply_density_overrides(copy.deepcopy(base), density)
    d = f"{density:03d}"

    # output dir + eval size
    assert cfg["output_dir"] == f"./outputs/synthetic_exp_{d}"
    assert cfg["eval_spatial_size"] == [256, 256]

    # train dataloader points at the requested density
    tds = _child(_child(cfg, "train_dataloader"), "dataset")
    train_img = tds.get("img_folder")
    assert isinstance(train_img, str) and train_img.endswith(f"density_{d}/train")
    train_ann = tds.get("ann_folder")
    assert isinstance(train_ann, str) and train_ann.endswith(f"density_{d}/train")

    # val dataloader points at the requested density
    vds = _child(_child(cfg, "val_dataloader"), "dataset")
    val_img = vds.get("img_folder")
    assert isinstance(val_img, str) and val_img.endswith(f"density_{d}/val")

    # transforms are pinned to 256x256 for both splits
    transforms = tds.get("transforms")
    assert isinstance(transforms, dict)
    ops = transforms.get("ops")
    assert isinstance(ops, list)
    for op in ops:
        assert isinstance(op, dict)
        op_type = op.get("type", "")
        if op_type == "OBBResize":
            assert op.get("size") == [256, 256]
        if op_type == "OBBConvertBoxes":
            assert op.get("img_size") == [256, 256]


def test_generation_to_temp_dir_produces_all_densities(tmp_path: Path):
    """End-to-end (output-safe): write all densities to a temp dir, no real IO."""
    template = _gen.load_yaml(TEMPLATE_PATH)

    written: list[Path] = []
    for density in DENSITIES:
        cfg = _gen.apply_density_overrides(copy.deepcopy(template), density)
        out = tmp_path / f"synthetic_exp_{density:03d}.yml"
        with open(out, "w") as fh:
            yaml.dump(cfg, fh)
        written.append(out)

    assert len(written) == len(DENSITIES)
    for out in written:
        assert out.exists()
        data = _gen.load_yaml(out)
        assert "__include__" not in data, "generated config must be standalone"
