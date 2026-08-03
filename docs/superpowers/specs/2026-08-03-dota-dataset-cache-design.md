# DOTA Dataset 分层缓存设计

Date: 2026-08-03

## 1. Purpose

优化 DEIMv2-OBB 训练数据加载：消除 `DotaDataset` 中的重复磁盘读取，借鉴 Ultralytics 的工业级缓存设计，实现**磁盘预生成缓存（L2）+ 内存 LRU 缓存（L1）** 双层加速。

## 2. 背景与现状分析

### 当前实现的问题（`engine/data/dataset/dota_dataset.py`）

1. **`img_ann_dict` 反复扫描磁盘**（line 40-58）：property 每次访问都执行 `os.listdir` 两次（图片 + 标注）。`__len__`（line 36-38）和 `load_item`（line 69）都会触发。
2. **`cat2id` / `id2cat` 每次构建**（line 60-66）：每次访问重建字典。
3. **标注文件每次读盘**（line 80）：每 epoch 每张图重新 `open` + `readlines` + 解析。
4. **图片每次读盘**（line 76）：每 epoch 每张图重新 `Image.open` + 解码。

### Ultralytics 借鉴点

| 特性 | Ultralytics 实现 |
|---|---|
| 文件索引缓存 | `im_files` 列表 `__init__` 构建一次 |
| RAM 缓存 | `cache='ram'`：全部图像进内存 |
| Disk 缓存 | `cache='disk'`：图像转 `.npy`，`np.load` 快 2-3 倍 |
| 损坏处理 | `base.py:231-233`：npy 损坏删除后回退 `imread` |

## 3. 设计目标

1. 消除 `img_ann_dict` / `cat2id` / `id2cat` 的重复构建。
2. 标注解析只做一次（缓存原始行）。
3. 图片加载走**两级缓存**：L1 内存 LRU（热点图）+ L2 磁盘 npy（全量预生成）。
4. 保持向后兼容：默认行为与当前完全一致。

## 4. 非目标

- 不修改 Mosaic（已有 `use_cache` 机制）。
- 不修改 DataLoader 构建逻辑。
- 不引入新的第三方依赖（`numpy`、`PIL` 已有）。

## 5. 架构

```
┌────────────────────────────────────────────────────────────┐
│  DotaDataset                                               │
│                                                            │
│  初始化（一次性）                                           │
│  ├─ _img_ann_dict: 文件索引 dict（替代 property）          │
│  ├─ _cat2id / _id2cat: 类别映射 dict                       │
│  ├─ _ann_cache: stem → 原始标注行列表                       │
│  ├─ _npy_dir: 磁盘缓存目录（<img_folder>_npy/）             │
│  ├─ _img_cache: OrderedDict LRU（L1，上限可配）             │
│  └─ precache_images(): 训练前全量生成 .npy（L2）            │
│                                                            │
│  load_item(index)                                          │
│  ├─ _load_image(stem):                                     │
│  │   ① L1 命中 → 返回内存图                                │
│  │   ② L2 命中 → np.load → 入 L1 → 返回                   │
│  │   ③ 回退 → Image.open → 返回（防御性）                  │
│  └─ _ann_cache[stem] + 图片尺寸 → boxes/labels             │
└────────────────────────────────────────────────────────────┘
```

## 6. 接口设计

### 6.1 构造参数

```python
class DotaDataset(DetDataset):
    def __init__(
        self,
        img_folder,
        ann_folder,
        classes_file,
        transforms,
        format,
        cache_images: str = "none",  # "none" | "disk"
        cache_ram: int = 0,          # 0=禁用L1, N=LRU上限
    ):
```

### 6.2 预生成接口

```python
def precache_images(self, num_workers: int = 8) -> None:
    """训练前预生成所有图片的 .npy 缓存（L2）。多进程并行。"""
```

- 存储目录：`<img_folder>_npy/`（与图片文件夹平级）
- npy 格式：`np.save(path, np.array(img))`（RGB uint8）
- 已存在的 npy 跳过（断点续生成）
- 多进程：`ThreadPool(num_workers)`（I/O 密集用线程池即可）

### 6.3 读取路径

```python
def _load_image(self, stem, image_path):
    if self._cache_ram > 0 and stem in self._img_cache:
        self._img_cache.move_to_end(stem)
        return self._img_cache[stem]

    if self.cache_images == "disk":
        npy_path = os.path.join(self._npy_dir(), f"{stem}.npy")
        try:
            img = Image.fromarray(np.load(npy_path))
        except Exception:
            if os.path.exists(npy_path):
                os.remove(npy_path)          # 损坏则删除回退
            img = Image.open(image_path).convert("RGB")
    else:
        img = Image.open(image_path).convert("RGB")

    if self._cache_ram > 0:
        self._img_cache[stem] = img
        if len(self._img_cache) > self._cache_ram:
            self._img_cache.popitem(last=False)
    return img
```

### 6.4 标注解析重构

```python
def _preload_annotations(self) -> None:
    """一次性读取所有标注原始行，存入 _ann_cache。"""
    for img_name, ann_name in self._img_ann_dict.items():
        stem = os.path.splitext(img_name)[0]
        ann_path = os.path.join(self.ann_folder, ann_name)
        with open(ann_path, "r") as f:
            self._ann_cache[stem] = f.readlines()

def _parse_from_lines(self, ann_lines, w, h):
    """从原始标注行解析 boxes/labels。

    YOLO-OBB 分支（format == "YOLO-OBB"）：
        每行: cls_id x1 y1 x2 y2 x3 y3 x4 y4（归一化）
        → xyxyxyxy[:, 0] *= w; xyxyxyxy[:, 1] *= h
        → xywhr = self.xyxyxyxy_to_xywhr(xyxyxyxy)
        → boxes.append(xywhr); labels.append(cls_id)

    DOTA 分支（format == "DOTA"）：
        每行: x1 y1 x2 y2 x3 y3 x4 y4 label_name difficulty（像素）
        → xywhr = self.xyxyxyxy_to_xywhr(xyxyxyxy)
        → cat = " ".join(parts[8:-1]); labels.append(cat_id_dict[cat])

    返回 (boxes_tensor, labels_tensor)，空标注返回 (zeros(0,5), zeros(0,long))
    """
```

**关键点：** YOLO-OBB 标注是归一化坐标，反归一化依赖图片尺寸 `(w,h)`。因此 `_ann_cache` 存**原始行**（不预先转 tensor），在 `load_item` 读图拿到 `w,h` 后再转换。

## 7. 配置项

```yaml
train_dataloader:
  dataset:
    cache_images: disk    # none | disk
    cache_ram: 200        # 0=禁用, N=LRU上限（推荐 200）
```

| cache_images | cache_ram | 行为 |
|---|---|---|
| none | 0 | 原始路径（默认，零改动） |
| disk | 0 | 仅磁盘预生成 |
| disk | N | **磁盘 + 内存 LRU（推荐）** |
| none | N | 仅内存 LRU |

## 8. 调用流程

### 训练前预生成（`train.py`）

```python
# 在构建 dataloader 之前
if dataset.cache_images == "disk":
    dataset.precache_images(num_workers=8)
```

### 训练中

```python
# 无需改动，load_item 内部自动走缓存路径
```

## 9. 错误处理

| 场景 | 处理 |
|---|---|
| npy 损坏（np.load 抛异常） | 删除 npy，回退 Image.open，不中断 |
| npy 不存在（预生成未跑） | 回退 Image.open（防御性） |
| 图片文件丢失 | 保持现有 FileNotFoundError 行为 |
| LRU 缓存满 | `popitem(last=False)` 淘汰最久未用 |

## 10. 测试计划

### 10.1 单元测试

- `img_ann_dict` 只构建一次（无 listdir 重复调用）
- `cat2id`/`id2cat` 构建一次
- `_preload_annotations` 正确缓存所有标注行
- `_load_image` 三级路径：L1 命中 / L2 命中 / 回退
- LRU 淘汰逻辑（cache_ram=2 时插入 3 张，第 1 张被淘汰）
- npy 损坏回退

### 10.2 集成测试

- `cache_images="disk"` 下 `precache_images()` 生成全部 npy
- 重新实例化 dataset，`load_item` 全部走 L2 命中
- Mosaic 场景下 L1 命中（同窗口重复采样）

## 11. 验收标准

1. 默认配置（none/0）行为与当前完全一致，现有测试全通过。
2. `cache_images="disk"` 预生成后，`load_item` 不触发 `Image.open`（全部 np.load）。
3. `cache_ram=N` 时，Mosaic 重复采样同一窗口图片内存命中。
4. 损坏 npy 不中断训练。
5. 标注解析只做一次（可观测：`_parse_from_lines` 调用计数）。
