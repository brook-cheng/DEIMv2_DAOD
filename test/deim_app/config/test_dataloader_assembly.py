"""App→engine dataloader assembly must be ``create()``-ready (acceptance T01 regression).

The app layer stubs ``YAMLConfig`` in its unit tests, so nothing caught that
``_map_data`` assembled ``train_dataloader``/``val_dataloader`` sections
WITHOUT ``type: DataLoader``. The engine include chains used by app presets
(base/dataloader.yml, base/deimv2.yml, deimv2_obb_common.yml) do not carry
that key either — the first REAL ``solver.train()`` died with
``KeyError: '_pymodule'`` inside ``workspace.create``. The resolver owns the
dataloader assembly, so it must inject the type itself.

Run:
    pytest test/deim_app/config/test_dataloader_assembly.py -v
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deim_app.config.mapping import _map_data  # noqa: E402
from deim_app.config.schema import AppConfig, DataConfig  # noqa: E402


def _coco_app() -> AppConfig:
    return AppConfig(
        data=DataConfig(
            format="COCO",
            train_images="/synth/train",
            train_annotations="/synth/train.json",
            val_images="/synth/val",
            val_annotations="/synth/val.json",
        )
    )


def _dota_app() -> AppConfig:
    return AppConfig(
        data=DataConfig(
            format="DOTA",
            train_images="/synth/train",
            train_annotations="/synth/train",
            val_images="/synth/val",
            val_annotations="/synth/val",
            classes_file="/synth/classes.txt",
        )
    )


def test_map_data_injects_dataloader_type_coco():
    overrides: dict = {}
    _map_data(overrides, _coco_app(), box_mode="hbb")
    assert overrides["train_dataloader"]["type"] == "DataLoader"
    assert overrides["val_dataloader"]["type"] == "DataLoader"


def test_map_data_injects_dataloader_type_dota():
    overrides: dict = {}
    _map_data(overrides, _dota_app(), box_mode="obb")
    assert overrides["train_dataloader"]["type"] == "DataLoader"
    assert overrides["val_dataloader"]["type"] == "DataLoader"


def test_map_data_dataset_sections_still_typed():
    overrides: dict = {}
    _map_data(overrides, _coco_app(), box_mode="hbb")
    assert overrides["train_dataloader"]["dataset"]["type"] == "CocoDetection"
    _map_data(overrides, _dota_app(), box_mode="obb")
    assert overrides["val_dataloader"]["dataset"]["type"] == "DotaDataset"


def test_map_runtime_preserves_and_injects_transforms_compose_type():
    """``_map_runtime`` rewrites op sizes and must also make every transforms
    section inject-ready (``type: Compose``) — the app layer has no
    dataset-YAML sibling to carry that key."""
    from deim_app.config.mapping import _map_runtime
    from deim_app.config.schema import AppConfig, RuntimeConfig

    overrides: dict = {
        "train_dataloader": {
            "dataset": {
                "transforms": {
                    "ops": [{"type": "Resize", "size": [1280, 1280]}],
                }
            }
        },
        "val_dataloader": {"dataset": {"transforms": {"ops": []}}},
    }
    app = AppConfig(runtime=RuntimeConfig(input_size=(320, 640)))
    _map_runtime(overrides, app)
    for dl_key in ("train_dataloader", "val_dataloader"):
        tf = overrides[dl_key]["dataset"]["transforms"]
        assert tf["type"] == "Compose"
    assert overrides["train_dataloader"]["dataset"]["transforms"]["ops"][0]["size"] == [
        320,
        640,
    ]
