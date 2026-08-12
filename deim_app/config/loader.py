"""Two-stage typed application YAML loader.

Stage 1 (trust boundary): read the user YAML raw with ``yaml.safe_load`` and
reject any key outside the public whitelist BEFORE touching the engine loader.
This prevents untrusted user input from injecting arbitrary algorithm sections.

Stage 2 (resolution): delegate to the engine loader (via the adapter) to
resolve ``__include__`` and merge the trusted application base, then extract
ONLY the six public sections into ``AppConfig``. CLI overrides are merged on
top and re-validated against the same whitelist.

The full engine dict (algorithm sections included) is returned as
``LoadedAppConfig.engine_base`` for downstream mapping (Task 3+). It is never
stored inside ``AppConfig``.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from deim_app.adapters._engine_yaml import load_engine_config, merge_dict
from deim_app.config.schema import AppConfig
from deim_app.errors import AppConfigError

__all__ = ["LoadedAppConfig", "load_app_config"]


# ---------------------------------------------------------------------------
# Public-key whitelist
# ---------------------------------------------------------------------------

#: The six public top-level sections a user may override.
PUBLIC_SECTIONS: tuple[str, ...] = (
    "project",
    "runtime",
    "data",
    "train",
    "evaluation",
    "inference",
)

#: Allowed leaf keys per top-level section.
_SECTION_KEYS: dict[str, frozenset[str]] = {
    "project": frozenset({"name", "output_dir"}),
    "runtime": frozenset({"input_size", "seed"}),
    "data": frozenset({
        "format", "train_images", "train_annotations", "val_images",
        "val_annotations", "classes_file", "num_workers", "cache_images",
    }),
    "train": frozenset({
        "epochs", "batch_size", "learning_rate", "device", "amp",
        "pretrained", "resume", "early_stopping",
    }),
    "evaluation": frozenset({"batch_size", "device"}),
    "inference": frozenset({
        "checkpoint", "device", "batch_size", "score_threshold",
        "top_k", "class_filter", "output_formats",
    }),
}

#: Allowed keys inside nested sub-mappings, keyed by (section, subkey).
_SUBSECTION_KEYS: dict[tuple[str, str], frozenset[str]] = {
    ("train", "early_stopping"): frozenset({"enabled", "patience"}),
}


def validate_public_keys(data: Mapping[str, object], location: str) -> None:
    """Reject any key outside the public whitelist, with its full dotted path.

    ``__include__`` is allowed as a directive at the top level. Every offending
    key raises ``AppConfigError`` with a message embedding the dotted path so
    users can locate it (e.g. ``train.early_stopping.bogus``).
    """
    for key in data:
        if key == "__include__":
            continue
        if key not in _SECTION_KEYS:
            raise AppConfigError(
                f"unknown public key '{key}' in {location}; "
                f"allowed top-level keys: __include__, {sorted(_SECTION_KEYS)}"
            )
    for section, allowed in _SECTION_KEYS.items():
        if section not in data:
            continue
        section_val = data[section]
        if section_val is None:
            continue
        if not isinstance(section_val, Mapping):
            raise AppConfigError(
                f"public section '{section}' in {location} must be a mapping, "
                f"got {type(section_val).__name__}"
            )
        for subkey in section_val:
            if subkey not in allowed:
                raise AppConfigError(
                    f"unknown key '{section}.{subkey}' in {location}; "
                    f"allowed: {sorted(allowed)}"
                )
        for (sec, sub), sub_allowed in _SUBSECTION_KEYS.items():
            if sec != section or sub not in section_val:
                continue
            subval = section_val[sub]
            if subval is None:
                continue
            if not isinstance(subval, Mapping):
                raise AppConfigError(
                    f"'{sec}.{sub}' in {location} must be a mapping, "
                    f"got {type(subval).__name__}"
                )
            for subsub in subval:
                if subsub not in sub_allowed:
                    raise AppConfigError(
                        f"unknown key '{sec}.{sub}.{subsub}' in {location}; "
                        f"allowed: {sorted(sub_allowed)}"
                    )


# ---------------------------------------------------------------------------
# __include__ validation
# ---------------------------------------------------------------------------

#: Approved application-base path marker (used to reject direct algorithm-preset includes).
_APPROVED_BASE_DIR = "configs/app/base/"


def validate_single_application_base(source: Path, includes: object) -> Path:
    """Verify ``__include__`` resolves to exactly one approved application base.

    Failure modes (all raise ``AppConfigError`` with "application base" in the
    message):
      * ``__include__`` absent, empty, or not length-1
      * the single entry is not a string
      * the entry traverses parent directories (``..``)
      * the entry references ``configs/`` outside ``configs/app/base/``
      * the resolved file does not exist
    """
    if not isinstance(includes, list) or len(includes) != 1:
        count: object = len(includes) if isinstance(includes, list) else "missing"
        raise AppConfigError(
            f"application base: user YAML must declare exactly one __include__ "
            f"target (got {count})"
        )
    raw = includes[0]
    if not isinstance(raw, str):
        raise AppConfigError(
            f"application base: __include__ entry must be a string, "
            f"got {type(raw).__name__} ({raw!r})"
        )
    parts = raw.replace("\\", "/").split("/")
    if ".." in parts:
        raise AppConfigError(
            f"application base: __include__ '{raw}' must not traverse parent "
            f"directories"
        )
    if "configs/" in raw and _APPROVED_BASE_DIR not in raw:
        raise AppConfigError(
            f"application base: __include__ '{raw}' must resolve to "
            f"{_APPROVED_BASE_DIR}hbb_app.yml or {_APPROVED_BASE_DIR}obb_app.yml"
        )
    resolved = (source.parent / raw).resolve()
    if not resolved.exists():
        raise AppConfigError(
            f"application base: '{raw}' resolved to {resolved} but file does "
            f"not exist"
        )
    return resolved


# ---------------------------------------------------------------------------
# Public-section extraction
# ---------------------------------------------------------------------------

def extract_public_sections(engine_base: Mapping[str, object]) -> dict[str, object]:
    """Deep-copy only the six public sections out of the merged engine dict.

    The deepcopy insulates ``engine_base`` from the subsequent CLI-override
    merge (which mutates ``public_merged`` in place).
    """
    result: dict[str, object] = {}
    for section in PUBLIC_SECTIONS:
        if section in engine_base:
            value = engine_base[section]
            result[section] = copy.deepcopy(value)
    return result


# ---------------------------------------------------------------------------
# Result container + entry point
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class LoadedAppConfig:
    """Result of ``load_app_config``.

    Attributes:
        app: frozen, validated application config (six public sections only).
        engine_base: full merged engine dict (algorithm content included) for
            downstream mapping. Must not be mutated by callers.
        source: resolved path to the user YAML.
        app_base: resolved path to the approved application-base YAML.
    """

    app: AppConfig
    engine_base: dict[str, object]
    source: Path
    app_base: Path


def load_app_config(
    path: str | Path,
    cli_overrides: Mapping[str, object] | None = None,
) -> LoadedAppConfig:
    """Load and validate a typed application config from a user YAML.

    Ordering (authoritative, from the task brief):
      1. ``yaml.safe_load`` the user file raw (trusted boundary).
      2. ``validate_public_keys`` on the user dict.
      3. ``validate_public_keys`` on CLI overrides (same whitelist).
      4. ``validate_single_application_base`` on ``__include__``.
      5. ``load_engine_config`` (engine loader, fresh ``cfg={}``).
      6. ``extract_public_sections`` (deep-copy the six sections).
      7. ``merge_dict`` CLI overrides on top of the public sections.
      8. ``AppConfig.from_mapping`` (type/enum/range validation).

    Raises:
        AppConfigError: on any whitelist, include, or value violation.
    """
    source = Path(path).resolve()
    if not source.exists():
        raise AppConfigError(f"application config '{source}' does not exist")

    user_raw: object = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if not isinstance(user_raw, dict):
        raise AppConfigError(
            f"application config root must be a mapping, "
            f"got {type(user_raw).__name__}"
        )

    validate_public_keys(user_raw, "user YAML")

    overrides: dict[str, object] = dict(cli_overrides) if cli_overrides else {}
    if overrides:
        validate_public_keys(overrides, "CLI overrides")

    app_base = validate_single_application_base(source, user_raw.get("__include__"))

    engine_base: dict[str, Any] = load_engine_config(source)

    public_merged = extract_public_sections(engine_base)
    if overrides:
        public_merged = merge_dict(public_merged, overrides)

    app = AppConfig.from_mapping(public_merged)

    return LoadedAppConfig(
        app=app,
        engine_base=engine_base,
        source=source,
        app_base=app_base,
    )
