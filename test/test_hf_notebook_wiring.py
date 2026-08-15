"""HuggingFace example notebook hygiene contracts.

The upstream DEIMv2-HF publishing demo lives at ``docs/examples/hf_models.ipynb``
(root stays package/directory-only, mirroring the ``tools/train`` relocation).
The notebook must stay diff-reviewable and machine-portable:

1. **Location**: ``docs/examples/hf_models.ipynb``; no root-level copy.
2. **No embedded outputs**: execution results (base64 images, dumps) bloat the
   file to ~10x its source size and make git diffs useless.
3. **No machine-specific absolute paths** in source cells — examples use
   placeholders, not one developer's home directory.
4. **README** links to the relocated path.

Run:
    pytest test/test_hf_notebook_wiring.py -v
"""

import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

NOTEBOOK = "docs/examples/hf_models.ipynb"

_ABS_PATH = re.compile(r'["\']/(home|mnt|Users|data)/[^"\']+["\']')


def _cells():
    with open(os.path.join(ROOT, NOTEBOOK), encoding="utf-8") as fh:
        return json.load(fh)["cells"]


def test_notebook_lives_under_docs_examples():
    assert os.path.isfile(os.path.join(ROOT, NOTEBOOK)), (
        f"example notebook must exist at {NOTEBOOK}"
    )
    assert not os.path.isfile(os.path.join(ROOT, "hf_models.ipynb")), (
        "root-level hf_models.ipynb must not coexist with the relocated notebook"
    )


def test_notebook_has_no_embedded_outputs():
    offenders = [
        i for i, cell in enumerate(_cells()) if cell.get("outputs")
    ]
    assert not offenders, (
        f"cells {offenders} carry embedded outputs — clear outputs before "
        "committing (source-only notebooks stay reviewable)"
    )


def test_notebook_source_has_no_machine_specific_paths():
    for i, cell in enumerate(_cells()):
        src = "".join(cell["source"])
        match = _ABS_PATH.search(src)
        assert match is None, (
            f"cell {i} hardcodes a machine-specific path: {match.group(0)!r} — "
            "use a placeholder instead"
        )


def test_readme_links_relocated_notebook():
    with open(os.path.join(ROOT, "README.md"), encoding="utf-8") as fh:
        text = fh.read()
    assert "(./docs/examples/hf_models.ipynb)" in text, (
        "README must link the relocated notebook"
    )
    assert "(./hf_models.ipynb)" not in text, (
        "README must not link the removed root-level notebook"
    )
