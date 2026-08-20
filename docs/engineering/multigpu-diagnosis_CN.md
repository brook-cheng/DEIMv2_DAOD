# 多卡训练进度漂移/超时诊断指南

> 症状：双卡 torchrun 训练若干 epoch 后两卡进度不一致，慢卡超出预设时间触发异常中断。
> 目标：通过完整现场记录复现并定位根因。

## 一、快速开始（训练服务器）

```bash
bash scripts/diag_2gpu.sh                 # 双卡跑 diag_2gpu_vits16.yml（vits16 冻结小模型）
EPOCHS=30 bash scripts/diag_2gpu.sh       # 指定轮次（覆盖配置）
GPUS=0,1 MASTER_PORT=7777 bash scripts/diag_2gpu.sh
```

诊断配置 `configs/custom_obb/dlzdt/diag_2gpu_vits16.yml`：
模型 ← `app/presets/deimv2_dinov3_vits16_freeze_obb.yml`（ResAtten+vits16 冻结，224 维小模型，迭代快、复现快）；
数据/训练参数 ← `sp_fz_common.yml` 全套继承（dlzdt 电缆终端头 OBB，含 cache_images=disk 与完整增强管线）。

## 二、采集的现场信息

| 产物 | 内容 | 定位作用 |
|---|---|---|
| `env_snapshot.txt` | GPU/驱动/拓扑/P2P//dev/shm/torch/nccl 版本 | 排除环境差异 |
| `train_full_<时间戳>.log` | torchrun 全量输出（含 NCCL_DEBUG=INFO） | 看通信初始化、coll 选择、告警 |
| `loader_probe_<时间戳>.log` | 正式训练前构造 train DataLoader 并读取首 batch 的双 rank 输出 | 区分配置/构造/首 batch 问题与训练期问题 |
| `loader_probe_rank*_<时间戳>.log` | 按 rank 分割的 loader probe 输出 | 对比各 rank 的 dataset、sampler、loader 和首 batch 状态 |
| `heartbeat_rank{0,1}.jsonl` | **每 rank 每 iteration 时间戳** | **定位分歧点**（见下） |
| `nccl_flight_dump/` | 超时自动 dump 的 c10d flight recorder + 各 rank 调用栈 | 看超时瞬间各 rank 卡在哪个集合通信 |
| 人工栈 dump | 挂住时另开终端 `kill -USR1 <训练pid>` | 立即抓全线程栈入日志 |

## 三、心跳对齐分析法（核心）

```bash
python3 - <<'PY'
import json
def load(p):
    return {r["global_step"]: r["ts"] for r in map(json.loads, open(p))}
r0 = load("outputs/diag_2gpu_vits16/diag/heartbeat_rank0.jsonl")
r1 = load("outputs/diag_2gpu_vits16/diag/heartbeat_rank1.jsonl")
gaps = [(s, r0[s]-r1[s]) for s in sorted(set(r0) & set(r1))]
first_bad = next(((s,g) for s,g in gaps if abs(g) > 5.0), None)
print("首个分歧 iteration（两卡时间差>5s）:", first_bad)
# 仅一方有记录的最后一个 step = 该 rank 停摆点
only0 = sorted(set(r0)-set(r1)); only1 = sorted(set(r1)-set(r0))
print("rank0 独有最后 step:", only0[-1] if only0 else None)
print("rank1 独有最后 step:", only1[-1] if only1 else None)
PY
```

- 分歧点前后的 `train_full.log`（该 iteration 附近）即第一现场
- 若停摆 step 后该 rank 心跳**完全消失**：进程/worker 挂死（查 USR1 栈）
- 若心跳**持续但变慢**：数据加载/IO/缓存问题（见观察清单①②）

## 四、DataLoader 探针解读（训练前第一道闸）

脚本按以下顺序推进，前一步失败时不会进入下一步：

```text
NCCL smoke -> loader probe -> first batch -> first iteration -> epoch/policy switch -> collective timeout
```

`loader_probe_<时间戳>.log` 中的关键记录：

| 记录 | 含义 | 正常预期 |
|---|---|---|
| `PROBE` | rank、local rank、world size、代码版本 | 两个 rank 的 world size 相同，local rank 分别为 0/1 |
| `DATASET_CONFIG` | 配置类型、classes 文件和数据目录存在性 | `type=DotaDataset`、`classes_file_set=True`、目录存在 |
| `DATASET_INSTANCE` | 实际构造出的 dataset 类型和长度 | 两个 rank 类型和长度相同 |
| `SAMPLER` | sampler 类型、shuffle、drop_last、每 rank 样本数 | 分布式时为 `DistributedSampler`，两个 rank 的 `num_samples` 相同 |
| `LOADER` | batch、shuffle、drop_last、worker 数和 loader 长度 | 显式 sampler 时 DataLoader 的 `shuffle` 不应为 True |
| `FIRST_BATCH_OK` | 首次 `next(iter(loader))` 是否成功及耗时 | 两个 rank 都出现，耗时差距可接受 |
| `PROBE_OK` | 首 batch 后的 rank 间 barrier 是否通过 | 两个 rank 都出现 |

分布式 DataLoader 的正确关系是：

```text
DistributedSampler(shuffle=True)
DataLoader(shuffle=False, sampler=DistributedSampler(...))
```

`shuffle=True` 应由 `DistributedSampler` 控制；PyTorch 不允许 DataLoader 同时使用显式 sampler 和 `shuffle=True`。每个 epoch 创建 iterator 前还必须调用 `sampler.set_epoch(epoch)`，否则各 epoch 会重复相同的 shuffle 顺序。

按 probe 停止位置判断范围：

- 没有 `DATASET_INSTANCE`：dataset/workspace 配置或外部路径问题。
- 有 `DATASET_INSTANCE`，没有 `LOADER`：DataLoader 参数冲突或 inject 问题。
- 有 `LOADER`，没有 `FIRST_BATCH_OK`：worker、transform、cache、collate 或首样本问题。
- 两个 `FIRST_BATCH_OK` 都有但没有 `PROBE_OK`：rank 间 probe barrier 或进程组问题。
- `PROBE_OK` 完整但正式训练失败：问题已经越过 loader 构造边界，继续看 heartbeat、flight recorder 和模型 collective 顺序。

## 五、观察清单（按嫌疑排序）

> **2026-08-20 已结案**：根因为观察清单 ②（缺 DistributedSampler），证据与修复见
> 提交 `f08cfb7`。flight recorder 显示两卡卡在不同 iteration 的不同 collective
> （SyncBN allgather vs 损失 allreduce，SeqNum 差 4）——不是慢卡漂移而是
> collective 流错位死锁；触发窗口恰为 epoch10 增强策略切换点。

### ① disk 缓存双卡竞态（高嫌疑）
`DotaDataset.precache_images` 在两 rank 各自并发执行：`np.save` 直接写同一路径
**无锁、无原子写**；`_load_image` 读到半写文件时 `np.load` 失败→**删除该文件**→可能删掉
另一 rank 正在写的缓存→反复竞态。慢卡读缓存重试/回退原图导致迭代变慢，快卡在
all_reduce 处等待直至 watchdog 超时。**与症状（若干轮后漂移）吻合**。
验证：症状出现时看 `train_full.log` 是否有 `[DotaDataset] Failed to cache`；尝试
`cache_images: none` 复跑对比是否复现。

### ② 无 DistributedSampler（高嫌疑）
`engine/core/yaml_config.build_dataloader` 仅按 world_size 均分 batch_size，
**未挂 DistributedSampler**——两卡各自 shuffle 全量数据：样本重复（等效 batch 语义错）、
两卡首个 epoch 的迭代耗时就可能不同（不同样本的 Mosaic/CopyBlend 代价差异大），
与 ① 叠加放大漂移。验证：心跳对比两卡同 iteration 耗时分布。

### ③ /dev/shm 不足
多 worker DataLoader 共享内存不足时 worker 通信阻塞（Docker 常见默认 64M）。
已在 `env_snapshot.txt` 记录；<1G 时优先怀疑。

### ④ NCCL 通信本身
P2P/IB 配置、NVLink 拓扑（`nvidia-smi topo -m`）；`train_full.log` 中 NCCL INIT
段的告警。flight recorder dump 会直接给出超时瞬间卡住的集合通信调用。

## 六、诊断开关速查

| 环境变量 | 作用 | 默认 |
|---|---|---|
| `DEEP=1` | 升级 NCCL 日志至 INIT+COLL+TUNING（逐 coll 明细，量极大仅深潜用） | 关 |
| `DEIM_DIAG_HEARTBEAT=1` | 每 rank 每 iteration 心跳 JSONL | 关（零开销） |
| `DEIM_DIAG_USR1=1` | 启用 `kill -USR1` 栈 dump | 关 |
| `TORCH_NCCL_TRACE_BUFFER_SIZE` | c10d flight recorder 环形缓冲 | 脚本设 2048 |
| `TORCH_NCCL_DUMP_ON_TIMEOUT=1` | 超时自动 dump（字符串开关仅 1/0，设 3 会炸 init） | 脚本设 1 |
| `TORCH_NCCL_HEARTBEAT_TIMEOUT` | watchdog 超时（毫秒） | 脚本 600000；**复现期保持与出问题时一致**，仅加速复现时下调 |
| `EPOCHS` / `GPUS` / `MASTER_PORT` | 脚本参数 | 30 / 0,1 / 7777 |

## 七、定位后的修复方向预告（按观察清单对应）

- ①：缓存写入改原子（`tmp + os.replace`），预缓存仅 rank0 执行 + barrier
- ②：`build_dataloader` 挂 `DistributedSampler(dataset, shuffle=..., drop_last=...)`
- ③：`docker run --shm-size` 或减小 num_workers
- ④：按 NCCL dump 针对性设置（如 `NCCL_P2P_DISABLE=1` 对照实验）

## 八、多卡训练学习路径

建议先按下面顺序阅读：

1. [PyTorch Distributed Overview](https://pytorch.org/tutorials/beginner/dist_overview.html)：理解 rank、world size、process group 和 collective。
2. [DistributedDataParallel notes](https://pytorch.org/docs/stable/notes/ddp.html)：理解每个 rank 为什么必须执行一致的模型与 collective 顺序。
3. [DistributedSampler API](https://pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler)：理解数据如何分片、`shuffle`、`drop_last` 以及 `set_epoch`。
4. [DataLoader API](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader)：理解 `sampler`、`shuffle`、`batch_sampler` 的互斥关系，以及 `num_workers` 的进程模型。
5. [torchrun / Distributed Elastic](https://pytorch.org/docs/stable/elastic/run.html)：理解 `LOCAL_RANK`、`RANK`、`WORLD_SIZE` 和 torchrun 进程启动方式。
6. [NVIDIA NCCL User Guide](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/)：在 DataLoader 和模型逻辑确认后，用于分析 collective、拓扑和 NCCL 调试开关。

当前问题最推荐先看第 1、2、3 篇，再结合本节的 DataLoader 探针输出理解实际运行状态。
