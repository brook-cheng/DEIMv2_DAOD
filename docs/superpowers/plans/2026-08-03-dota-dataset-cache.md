# DOTA Dataset 分层缓存实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 `DotaDataset` 实现 L1 内存 LRU + L2 磁盘 npy 双层图片缓存，消除文件索引/类别映射/标注解析的重复构建，提升训练数据加载速度。

**Architecture:** 三个递进任务：(1) 一次性构建文件索引/类别映射/标注缓存；(2) 三级图片加载路径 + 预生成接口；(3) 重构 `load_item` 接入缓存 + 训练入口预生成调用。每任务独立可测试。

**Tech Stack:** Python 3.12, PyTorch, numpy, PIL, multiprocessing.pool.ThreadPool

## Global Constraints

- 默认配置（`cache_images="none"`, `cache_ram=0`）行为与当前完全一致。
- 仅修改 `engine/data/dataset/dota_dataset.py`，不碰 Mosaic、DataLoader、`engine/core/`。
- 不引入新第三方依赖（numpy/PIL 已有）。
- npy 缓存目录为 `<img_folder>_npy/`，与图片文件夹平级。
- YOLO-OBB 标注反归一化依赖图片尺寸 `(w,h)`，标注缓存必须存原始行而非 tensor。
- 配置经 `__inject__` 自动注入构造器——YAML 添加 `cache_images`/`cache_ram` 键即可。
- **⚠️ 禁止任何 git 操作**：不执行 `git add`、`git commit`、`git push`、`git stash` 等。git 管理工作完全由用户自行完成。所有 Task 末尾的「提交」步骤已移除，实现完成后仅报告改动文件清单。

---

### Task 1: 一次性构建文件索引、类别映射、标注缓存

**Files:**
- Modify: `engine/data/dataset/dota_dataset.py`（`__init__`、新增 `_build_img_ann_dict`、`_preload_annotations`、`_read_annotation_lines`）
- Test: `test/test_dota_dataset_cache.py`

**Interfaces:**
- Consumes: 现有 `__init__` 参数（`img_folder, ann_folder, classes_file, transforms, format`）
- Produces:
  - `self._img_ann_dict: dict[str, str]`（一次性构建，替代 property）
  - `self._cat2id: dict[str, int]` / `self._id2cat: dict[int, str]`
  - `self._ann_cache: dict[str, list[str]]`（stem → 原始行）
  - 构造参数 `cache_images: str = "none"`、`cache_ram: int = 0`

- [ ] **Step 1: 写失败测试**

创建 `test/test_dota_dataset_cache.py`：

```python
import os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
    # property 不再存在，直接访问缓存属性
    assert not hasattr(DotaDataset, "img_ann_dict") or isinstance(
        ds._img_ann_dict, dict
    )


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
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest test/test_dota_dataset_cache.py -v`
Expected: FAIL — `DotaDataset` 无 `_img_ann_dict` 等属性（AttributeError）

- [ ] **Step 3: 重构 `__init__`**

修改 `engine/data/dataset/dota_dataset.py`：

```python
    def __init__(
        self,
        img_folder,
        ann_folder,
        classes_file,
        transforms,
        format,
        cache_images: str = "none",
        cache_ram: int = 0,
    ):
        super(DotaDataset, self).__init__()
        from ...deim.obb_geometry import xyxyxyxy_to_xywhr

        self.xyxyxyxy_to_xywhr = xyxyxyxy_to_xywhr
        self.transforms = transforms
        self.img_folder = img_folder
        self.ann_folder = ann_folder
        self.classes_file = classes_file
        self.img_extensions = [".jpg", ".jpeg", ".png", ".bmp"]
        with open(classes_file, "r") as f:
            label_names = [line.strip() for line in f.readlines()]
        self.label_names = label_names
        self.format = format
        if self.format != "DOTA" and self.format != "YOLO-OBB":
            raise ValueError(
                f"Unsupported format: {self.format}, must be DOTA or YOLO-OBB"
            )
        self.cache_images = cache_images
        self.cache_ram = cache_ram
        # 一次性构建，替代原 property
        self._img_ann_dict = self._build_img_ann_dict()
        self._cat2id = {name: id for id, name in enumerate(self.label_names)}
        self._id2cat = {id: name for id, name in enumerate(self.label_names)}
        # 一次性解析标注（存原始行）
        self._ann_cache: dict[str, list[str]] = {}
        self._preload_annotations()
        # 图片缓存
        self._img_cache = {}
        self._npy_dir_path = os.path.join(img_folder + "_npy")
```

- [ ] **Step 4: 添加辅助方法**

在 `__init__` 后添加：

```python
    def _build_img_ann_dict(self) -> dict:
        image_files = [
            f
            for f in os.listdir(self.img_folder)
            if str.lower(os.path.splitext(f)[-1]) in self.img_extensions
        ]
        ann_files = set(os.listdir(self.ann_folder))
        result = {
            img_file: img_file.replace(os.path.splitext(img_file)[-1], ".txt")
            for img_file in image_files
            if img_file.replace(os.path.splitext(img_file)[-1], ".txt") in ann_files
        }
        if len(result) == 0:
            raise ValueError(
                f"Valid annotation is empty, in {self.img_folder} and {self.ann_folder}"
            )
        return result

    def _preload_annotations(self) -> None:
        for img_name, ann_name in self._img_ann_dict.items():
            stem = os.path.splitext(img_name)[0]
            ann_path = os.path.join(self.ann_folder, ann_name)
            with open(ann_path, "r") as f:
                self._ann_cache[stem] = f.readlines()
```

删除原 `img_ann_dict` property（line 40-58）和 `cat2id`/`id2cat` property（line 60-66），改由缓存属性提供。

- [ ] **Step 5: 更新 `__len__`**

```python
    def __len__(self):
        return len(self._img_ann_dict)
```

- [ ] **Step 6: 运行测试确认通过**

Run: `pytest test/test_dota_dataset_cache.py -v`
Expected: 3 个测试全部 PASS

---

### Task 2: 三级图片加载路径 + 预生成接口

**Files:**
- Modify: `engine/data/dataset/dota_dataset.py`（新增 `_npy_dir`、`_load_image`、`precache_images`、`_convert_one`）
- Test: `test/test_dota_dataset_cache.py`（追加测试）

**Interfaces:**
- Consumes: Task 1 的 `_img_ann_dict`、`cache_images`、`cache_ram`
- Produces:
  - `_load_image(stem, image_path) -> PIL.Image`（L1→L2→回退）
  - `precache_images(num_workers=8) -> None`（训练前全量生成 npy）
  - `_npy_dir() -> str`

- [ ] **Step 1: 写失败测试（追加到测试文件）**

```python
def test_load_image_three_tiers(dataset_dir, tmp_path):
    img_dir, ann_dir, classes = dataset_dir
    ds = DotaDataset(img_dir, ann_dir, classes, None, "YOLO-OBB",
                     cache_images="disk", cache_ram=2)
    # 预生成
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
    assert img is img2


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
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest test/test_dota_dataset_cache.py -k "load_image or lru or corrupt" -v`
Expected: FAIL — `DotaDataset` 无 `_load_image`/`precache_images`（AttributeError）

- [ ] **Step 3: 实现缓存方法**

在 `_preload_annotations` 后添加：

```python
    def _npy_dir(self) -> str:
        return self._npy_dir_path

    def _convert_one(self, task) -> None:
        stem, img_path, npy_path = task
        if os.path.exists(npy_path):
            return
        try:
            img = Image.open(img_path).convert("RGB")
            np.save(npy_path, np.array(img), allow_pickle=False)
        except Exception as e:
            print(f"[DotaDataset] Failed to cache {img_path}: {e}")

    def precache_images(self, num_workers: int = 8) -> None:
        from multiprocessing.pool import ThreadPool

        os.makedirs(self._npy_dir(), exist_ok=True)
        tasks = [
            (
                os.path.splitext(img_name)[0],
                os.path.join(self.img_folder, img_name),
                os.path.join(self._npy_dir(), f"{os.path.splitext(img_name)[0]}.npy"),
            )
            for img_name in self._img_ann_dict.keys()
        ]
        with ThreadPool(num_workers) as pool:
            pool.map(self._convert_one, tasks)
        print(f"[DotaDataset] Cached {len(tasks)} images to {self._npy_dir()}")

    def _load_image(self, stem, image_path):
        if self.cache_ram > 0 and stem in self._img_cache:
            self._img_cache.move_to_end(stem)
            return self._img_cache[stem]

        if self.cache_images == "disk":
            npy_path = os.path.join(self._npy_dir(), f"{stem}.npy")
            try:
                img = Image.fromarray(np.load(npy_path))
            except Exception:
                if os.path.exists(npy_path):
                    os.remove(npy_path)
                img = Image.open(image_path).convert("RGB")
        else:
            img = Image.open(image_path).convert("RGB")

        if self.cache_ram > 0:
            self._img_cache[stem] = img
            if len(self._img_cache) > self.cache_ram:
                self._img_cache.popitem(last=False)
        return img
```

需要在文件头部添加 `import numpy as np`（当前只有 `import torch`, `import torchvision`, `from PIL import Image`）。

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest test/test_dota_dataset_cache.py -v`
Expected: 全部测试 PASS（含 Task 1 的 3 个 + Task 2 的 3 个）

---

### Task 3: 重构 `load_item` 接入缓存 + 训练入口预生成调用

**Files:**
- Modify: `engine/data/dataset/dota_dataset.py`（`load_item`、`_parse_from_lines`）
- Modify: `engine/solver/_solver.py`（dataloader 创建后触发 precache）
- Test: `test/test_dota_dataset_cache.py`（追加测试）

**Interfaces:**
- Consumes: Task 1 的 `_ann_cache`、Task 2 的 `_load_image`
- Produces:
  - `_parse_from_lines(ann_lines, w, h) -> (boxes_tensor, labels_tensor)`
  - `load_item(index) -> (img, target)`（使用缓存）

- [ ] **Step 1: 写失败测试（追加）**

```python
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

    # 再次调用，L1 命中
    img2, _ = ds.load_item(0)
    assert img is img2


def test_load_item_default_behavior(dataset_dir):
    """默认配置（none/0）行为与当前一致。"""
    img_dir, ann_dir, classes = dataset_dir
    ds = DotaDataset(img_dir, ann_dir, classes, None, "YOLO-OBB")
    img, target = ds.load_item(0)
    assert img.size == (64, 64)
    assert target["boxes"].shape[1] == 5
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest test/test_dota_dataset_cache.py -k "load_item" -v`
Expected: 当前 `load_item` 不调用 `_parse_from_lines`（AttributeError）或行为不符合断言

- [ ] **Step 3: 重构 `load_item`**

替换现有 `load_item`（line 68-127）：

```python
    def _parse_from_lines(self, ann_lines, w, h):
        cat_id_dict = self._cat2id
        boxes, labels = [], []
        if self.format == "YOLO-OBB":
            for line in ann_lines:
                parts = line.strip().split()
                if len(parts) < 9:
                    continue
                cls_id = (
                    int(parts[0])
                    if parts[0].lstrip("-").isdigit()
                    else cat_id_dict.get(parts[0], 0)
                )
                pts = [float(p) for p in parts[1:9]]
                xyxyxyxy = torch.tensor(pts).reshape(4, 2)
                xyxyxyxy[:, 0] *= w
                xyxyxyxy[:, 1] *= h
                xywhr = self.xyxyxyxy_to_xywhr(xyxyxyxy)
                boxes.append(xywhr)
                labels.append(cls_id)
        elif self.format == "DOTA":
            for line in ann_lines:
                parts = line.strip().split()
                if len(parts) < 9:
                    continue
                pts = [float(p) for p in parts[:8]]
                xyxyxyxy = torch.tensor(pts).reshape(4, 2)
                xywhr = self.xyxyxyxy_to_xywhr(xyxyxyxy)
                boxes.append(xywhr)
                cat_parts = parts[8 : len(parts) - 1]
                cat = " ".join(cat_parts)
                labels.append(cat_id_dict[cat])
        else:
            raise ValueError(
                f"Unsupported format: {self.format}, must be DOTA or YOLO-OBB"
            )
        return (
            torch.stack(boxes) if boxes else torch.zeros(0, 5),
            torch.tensor(labels) if labels else torch.zeros(0, dtype=torch.long),
        )

    def load_item(self, index):
        image_names = list(self._img_ann_dict.keys())
        image_absolute_path = os.path.join(self.img_folder, image_names[index])
        stem = os.path.splitext(image_names[index])[0]

        img = self._load_image(stem, image_absolute_path)
        w, h = img.size

        ann_lines = self._ann_cache[stem]
        boxes, labels = self._parse_from_lines(ann_lines, w, h)

        target = {
            "boxes": boxes,
            "labels": labels,
            "orig_size": torch.tensor([w, h]),
        }
        return img, target
```

- [ ] **Step 4: 在训练入口触发预生成**

修改 `engine/solver/_solver.py`，在 dataloader 创建后（line 164-165 附近）添加：

```python
        # DOTA dataset disk cache pre-generation (spec 2026-08-03)
        train_ds = self.train_dataloader.dataset
        if getattr(train_ds, "cache_images", "none") == "disk":
            train_ds.precache_images(num_workers=8)
```

- [ ] **Step 5: 运行全部测试确认通过**

Run: `pytest test/test_dota_dataset_cache.py -v`
Expected: 全部 8 个测试 PASS

Run: `python -c "import py_compile; py_compile.compile('engine/data/dataset/dota_dataset.py', doraise=True); py_compile.compile('engine/solver/_solver.py', doraise=True); print('OK')"`
Expected: OK

---

## Final Verification Wave

- [ ] `pytest test/test_dota_dataset_cache.py -v` 全部 8 个测试通过
- [ ] `python -m pytest tests/ -k "obb" --collect-only -q` 不报错（现有测试不受影响）
- [ ] 语法检查：4 个修改文件 `py_compile` 通过
- [ ] 默认配置（`cache_images="none"`, `cache_ram=0`）下 `load_item` 行为与修改前一致（`test_load_item_default_behavior` 覆盖）

## Commit Strategy

- **⚠️ 无 git 操作**。本计划不创建任何 commit。
- 用户自行管理 git，包括 add/commit/push。
- 实现完成后，向用户报告完整改动文件清单（`engine/data/dataset/dota_dataset.py`、`engine/solver/_solver.py`、`test/test_dota_dataset_cache.py`），由用户决定何时提交及提交信息。

## Success Criteria

- `_img_ann_dict`/`_cat2id`/`_id2cat`/`_ann_cache` 一次性构建，无重复 listdir/dict 构建/标注解析。
- `cache_images="disk"` 时训练启动自动预生成全部 npy；`load_item` 全程 `np.load` 不触发 `Image.open`。
- `cache_ram=N` 时 LRU 正确淘汰，Mosaic 同窗口重复采样内存命中。
- 损坏 npy 删除后回退 `Image.open`，不中断训练。
- 默认配置零行为变化。
