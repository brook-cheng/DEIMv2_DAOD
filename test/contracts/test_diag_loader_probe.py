from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = (ROOT / "scripts/diag_2gpu.sh").read_text(encoding="utf-8")


def test_diag_script_contains_loader_probe_evidence_contract():
    for marker in (
        "loader_probe",
        "loader_probe_rank",
        "classes_file_set",
        "sampler_type",
        "loader_len",
        "FIRST_BATCH_OK",
        "LOADER_PROBE_RC",
    ):
        assert marker in SCRIPT


def test_loader_probe_bootstraps_repo_root_import_path():
    assert 'DEIM_DIAG_REPO_ROOT="$REPO_ROOT"' in SCRIPT
    assert 'sys.path.insert(0, os.environ["DEIM_DIAG_REPO_ROOT"])' in SCRIPT


def test_loader_probe_uses_same_config_and_gates_training():
    assert 'DEIM_DIAG_CONFIG="$CONFIG"' in SCRIPT
    probe_pos = SCRIPT.index("loader_probe")
    train_pos = SCRIPT.index("tools/train/train.py")
    assert probe_pos < train_pos
    assert "if [ \"$LOADER_PROBE_RC\" -ne 0 ]; then" in SCRIPT
