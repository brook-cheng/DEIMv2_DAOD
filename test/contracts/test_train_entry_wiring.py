"""Training entry-point wiring and credential hygiene contracts.

Two concerns:

1. **No hardcoded credentials** (RED first): the Comet ML API key must come
   from the environment only. A literal key in scripts is a leaked secret
   (CWE-798) — it is already in git history, so the key itself must be
   rotated on the Comet dashboard; this contract keeps it from returning.

2. **Entry-point location**: the training entry lives at
   ``tools/train/train.py`` (all runnable entry points live under ``tools/``);
   the repo root stays package-only. Shell scripts and README must reference
   the new path.

Run:
    pytest test/test_train_entry_wiring.py -v
"""

import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CREDENTIAL_SCOPES = (
    "tools/train/train.py",
    "train.py",
    "tools/dataset/gen_synthetic_dataset/run_synthetic_training.sh",
    "tools/dataset/gen_synthetic_dataset/run_synthetic_A.sh",
    "tools/dataset/gen_synthetic_dataset/run_synthetic_B.sh",
)

ENTRY_POINT = "tools/train/train.py"

ENTRY_REFERENCING_FILES = (
    "scripts/single_gpu_train.sh",
    "scripts/multi_gpu_train.sh",
    "scripts/single_gpu_val.sh",
    "README.md",
)

# A hardcoded credential: COMET_API_KEY= followed by a quoted non-empty value.
_KEY_LITERAL = re.compile(r'COMET_API_KEY\s*=\s*["\'][A-Za-z0-9]+["\']')


def test_no_hardcoded_comet_credentials():
    for rel in CREDENTIAL_SCOPES:
        path = os.path.join(ROOT, rel)
        if not os.path.isfile(path):
            continue
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
        assert not _KEY_LITERAL.search(text), (
            f"{rel}: hardcoded COMET_API_KEY literal — credentials must come "
            "from the environment only"
        )


def test_training_entry_lives_under_tools():
    assert os.path.isfile(os.path.join(ROOT, ENTRY_POINT)), (
        "training entry point must exist at tools/train/train.py"
    )
    assert not os.path.isfile(os.path.join(ROOT, "train.py")), (
        "root train.py must not coexist with the relocated entry point"
    )


def test_scripts_and_readme_reference_relocated_entry():
    for rel in ENTRY_REFERENCING_FILES:
        with open(os.path.join(ROOT, rel), encoding="utf-8") as fh:
            text = fh.read()
        assert "tools/train/train.py" in text, (
            f"{rel} must reference the relocated entry point"
        )
        assert not re.search(r"(?<![\w/])train\.py", text), (
            f"{rel} still references a bare root-level train.py"
        )


def test_entry_imports_engine_from_foreign_cwd(tmp_path):
    """The entry must bootstrap sys.path itself, independent of the cwd.

    ``python3 tools/train/train.py`` puts the SCRIPT directory (not the cwd)
    at ``sys.path[0]``; torchrun likewise runs from arbitrary working
    directories. The only reliable import root is the one derived from
    ``__file__`` — three dirnames up from ``<repo>/tools/train/train.py``.
    Regression: a two-dirname insert resolved to ``<repo>/tools`` and every
    engine import failed outside the repo root.
    """
    import subprocess
    import sys as _sys

    proc = subprocess.run(
        [
            _sys.executable,
            os.path.join(ROOT, "tools", "train", "train.py"),
            "--help",
        ],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, (
        f"entry failed from foreign cwd {tmp_path}:\n{proc.stderr[-2000:]}"
    )
    assert "usage:" in proc.stdout
