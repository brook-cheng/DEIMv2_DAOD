import os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
import numpy as np
import pytest
from PIL import Image
from engine.data.dataset.dota_dataset import DotaDataset


@pytest.fixture
def dataset_dir(tmp_path):
    img_dir = tmp_path / "images" / "train"
    ann_dir = tmp_path / "labels" / "train"
    img_dir.mkdir(parents=True)
    ann_dir.mkdir(parents=True)
    classes = tmp_path / "classes.txt"
    classes.write_text("target\n")
    for i in range(3):
        Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)).save(
            img_dir / f"{i:06d}.jpg"
        )
        (ann_dir / f"{i:06d}.txt").write_text("0 0.25 0.25 0.75 0.25 0.75 0.75 0.25 0.75\n")
    return str(img_dir), str(ann_dir), str(classes)


def test_img_ann_dict_built_once(dataset_dir):
    img_dir, ann_dir, classes = dataset_dir
    ds = DotaDataset(img_dir, ann_dir, classes, None, "YOLO-OBB")
    assert len(ds._img_ann_dict) == 3
    assert not hasattr(DotaDataset, "img_ann_dict")


def test_annotations_preloaded(dataset_dir):
    img_dir, ann_dir, classes = dataset_dir
    ds = DotaDataset(img_dir, ann_dir, classes, None, "YOLO-OBB")
    assert len(ds._ann_cache) == 3
    assert len(ds._ann_cache["000000"]) == 1  # 每图 1 行标注


def test_cat2id_built_once(dataset_dir):
    img_dir, ann_dir, classes = dataset_dir
    ds = DotaDataset(img_dir, ann_dir, classes, None, "YOLO-OBB")
    assert ds._cat2id == {"target": 0}
    assert ds._id2cat == {0: "target"}


def test_load_image_three_tiers(dataset_dir):
    img_dir, ann_dir, classes = dataset_dir
    ds = DotaDataset(img_dir, ann_dir, classes, None, "YOLO-OBB",
                     cache_images="disk", cache_ram=2)
    ds.precache_images(num_workers=2)
    npy_dir = ds._npy_dir()
    assert len(os.listdir(npy_dir)) == 3  # 3 张图全部生成 npy

    # L2 命中（不走 Image.open）
    stem = "000000"
    img_path = os.path.join(img_dir, f"{stem}.jpg")
    img = ds._load_image(stem, img_path)
    assert img.size == (64, 64)
    # L1 命中（第二次调用返回同一对象）
    img2 = ds._load_image(stem, img_path)
    assert img.size == img2.size  # L1 命中，内容一致


def test_lru_eviction(dataset_dir):
    img_dir, ann_dir, classes = dataset_dir
    ds = DotaDataset(img_dir, ann_dir, classes, None, "YOLO-OBB",
                     cache_images="disk", cache_ram=2)
    ds.precache_images(num_workers=2)
    for stem in ["000000", "000001", "000002"]:
        ds._load_image(stem, os.path.join(img_dir, f"{stem}.jpg"))
    assert "000000" not in ds._img_cache  # 最旧被淘汰
    assert len(ds._img_cache) == 2


def test_corrupt_npy_fallback(dataset_dir):
    img_dir, ann_dir, classes = dataset_dir
    ds = DotaDataset(img_dir, ann_dir, classes, None, "YOLO-OBB",
                     cache_images="disk", cache_ram=0)
    ds.precache_images(num_workers=2)
    npy_path = os.path.join(ds._npy_dir(), "000000.npy")
    with open(npy_path, "w") as f:
        f.write("corrupt")
    img = ds._load_image("000000", os.path.join(img_dir, "000000.jpg"))
    assert img.size == (64, 64)  # 回退 Image.open 成功
    assert not os.path.exists(npy_path)  # 损坏 npy 被删除


def test_precache_idempotent(dataset_dir):
    img_dir, ann_dir, classes = dataset_dir
    ds = DotaDataset(img_dir, ann_dir, classes, None, "YOLO-OBB",
                     cache_images="disk", cache_ram=0)
    ds.precache_images(num_workers=2)
    npy_path = os.path.join(ds._npy_dir(), "000000.npy")
    mtime1 = os.path.getmtime(npy_path)

    import time
    time.sleep(0.1)
    ds.precache_images(num_workers=2)
    mtime2 = os.path.getmtime(npy_path)
    assert mtime1 == mtime2  # npy 未被重新生成


def test_load_item_uses_cache(dataset_dir, monkeypatch):
    img_dir, ann_dir, classes = dataset_dir
    ds = DotaDataset(img_dir, ann_dir, classes, None, "YOLO-OBB",
                     cache_images="disk", cache_ram=10)
    ds.precache_images(num_workers=2)

    opened = []
    orig_open = Image.open
    monkeypatch.setattr(
        "PIL.Image.open",
        lambda p, *a, **kw: (opened.append(str(p)), orig_open(p, *a, **kw))[1],
    )

    img, target = ds.load_item(0)
    assert img.size == (64, 64)
    assert target["boxes"].shape[1] == 5  # xywhr
    assert target["labels"].shape[0] == 1
    assert len(opened) == 0  # 全程未调用 Image.open（全走 npy）

    # 再次调用，L1 命中（返回 copy，内容一致）
    img2, _ = ds.load_item(0)
    assert img.size == img2.size


def test_load_item_default_behavior(dataset_dir):
    """默认配置（none/0）行为与当前一致。"""
    img_dir, ann_dir, classes = dataset_dir
    ds = DotaDataset(img_dir, ann_dir, classes, None, "YOLO-OBB")
    img, target = ds.load_item(0)
    assert img.size == (64, 64)
    assert target["boxes"].shape[1] == 5
