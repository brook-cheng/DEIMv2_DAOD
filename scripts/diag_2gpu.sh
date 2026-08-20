#!/bin/bash
# ============================================================================
# 双卡训练诊断启动器（多卡进度漂移 / watchdog 超时中断 复现与定位）
# ----------------------------------------------------------------------------
# 用法（训练服务器上）:
#   bash scripts/diag_2gpu.sh                # 默认双卡，配置 diag_2gpu_vits16.yml
#   GPUS=0,1 EPOCHS=30 bash scripts/diag_2gpu.sh
#
# 产物（自动打包到 outputs/diag_2gpu_vits16/diag/bundle_<时间戳>.tar.gz）:
#   env_snapshot.txt   — GPU/驱动/NCCL/torch//dev/shm/拓扑 检查清单
#   train_full.log     — torchrun 全量 stdout+stderr（NCCL_DEBUG=INFO）
#   heartbeat_rank*.jsonl — 每 rank 每 iteration 时间戳（定位分歧点）
#   nccl_flight_dump/  — 超时自动 dump 的 c10d flight recorder + 各 rank 栈
#
# 人工干预手段（训练挂住时，另开终端）:
#   kill -USR1 <训练pid>   — 立即 dump 该 rank 全线程栈到 train_full.log
# ============================================================================
set -u -o pipefail   # pipefail: torchrun|tee 管道失败必须传播，否则冒烟拦截失效
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

GPUS="${GPUS:-0,1}"
NPROC="${NPROC:-2}"
CONFIG="${CONFIG:-configs/custom_obb/dlzdt/diag_2gpu_vits16.yml}"
EPOCHS="${EPOCHS:-}"
OUT_DIR="./outputs/diag_2gpu_vits16"
DIAG_DIR="$OUT_DIR/diag"
STAMP="$(date +%Y%m%d_%H%M%S)"
mkdir -p "$DIAG_DIR/nccl_flight_dump"

UPDATE_ARGS=()
[ -n "$EPOCHS" ] && UPDATE_ARGS+=(-u "epoches=$EPOCHS")

# ── 1. 环境快照（复现所需完整场景信息）──────────────────────────────────
{
  echo "===== diag run $STAMP ====="
  echo "--- date/host ---"; date; hostname; uname -a
  echo "--- GPUs ---"; nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv
  echo "--- GPU 拓扑 ---"; nvidia-smi topo -m 2>/dev/null || echo "(nvidia-smi topo 不可用)"
  echo "--- P2P ---"; nvidia-smi -q 2>/dev/null | grep -iA2 "peer" | head -20 || true
  echo "--- /dev/shm（多卡 DataLoader 命名经典坑） ---"; df -h /dev/shm
  echo "--- python/torch/nccl ---"
  python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'nccl', torch.cuda.nccl.version())"
  echo "--- 关键环境变量 ---"; env | grep -E "NCCL|CUDA_VISIBLE|TORCH" | sort
  echo "--- 配置 ---"; echo "GPUS=$GPUS NPROC=$NPROC CONFIG=$CONFIG EPOCHS=$EPOCHS UPDATE=${UPDATE_ARGS[*]:-}"
} > "$DIAG_DIR/env_snapshot.txt" 2>&1

# ── 2. 诊断开关 ─────────────────────────────────────────────────────────
# NCCL_DEBUG 默认 WARN（正常跑静默）；DEEP=1 升 INIT+COLL+TUNING 逐 coll 明细（仅深潜漂移时用）
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-ALL}"
if [ "${DEEP:-0}" = "1" ]; then
  export NCCL_DEBUG=INFO
  export NCCL_DEBUG_SUBSYS=INIT,COLL,TUNING
fi
export TORCH_NCCL_TRACE_BUFFER_SIZE=2048  # 开启 c10d flight recorder（环形缓冲）
# 注意：此变量是字符串开关（仅 1/0）；写成等级数字会让 init_process_group
# 直接抛 "Invalid value for environment variable"（服务器 20260818_153144 实证）
export TORCH_NCCL_DUMP_ON_TIMEOUT=1       # 超时时自动 dump flight recorder + 栈
export TORCH_NCCL_DEBUG_INFO_TEMP_FILE="$DIAG_DIR/nccl_flight_dump/timeout_info"
# watchdog 超时（毫秒）：保持与出问题时一致以免改变复现条件；需要缩短复现周期时下调
export TORCH_NCCL_HEARTBEAT_TIMEOUT="${TORCH_NCCL_HEARTBEAT_TIMEOUT:-600000}"
export DEIM_DIAG_HEARTBEAT=1              # 每 rank 每 iteration 心跳 JSONL
export DEIM_DIAG_USR1=1                   # kill -USR1 <pid> 人工 dump 栈
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

echo "[diag] 心跳与 NCCL flight recorder 已开启；日志 → $DIAG_DIR/"

# ── 2.5 分布式最小冒烟（30s 暴露 init 层问题，与训练代码解耦）──────────
# 独立日志文件：训练段 tee 会重建 train_full log，混写会丢失冒烟现场
# 注意：torchrun 的入口必须是脚本文件；`torchrun ... python3 -c "code"` 会把
# python3 当脚本路径（can't open file '.../python3'），故先落临时脚本。
echo "[diag] 分布式冒烟: torchrun ${NPROC} 进程 init + allreduce..."
SMOKE_SCRIPT="$(mktemp /tmp/diag_smoke_XXXXXX.py)"
cat > "$SMOKE_SCRIPT" <<'PYEOF'
import os

import torch
import torch.distributed as d

# NCCL requires one GPU per rank: bind via LOCAL_RANK or every rank lands on
# cuda:0 → "Duplicate GPU detected" (server 20260818_1602 evidence).
torch.cuda.set_device(int(os.getenv("LOCAL_RANK", 0)))
d.init_process_group("nccl")
t = torch.ones(1, device="cuda")
d.all_reduce(t)
name = torch.cuda.get_device_name(torch.cuda.current_device())
print(
    f"SMOKE_OK rank={d.get_rank()} ws={d.get_world_size()} "
    f"dev={torch.cuda.current_device()}({name}) sum={t.item():.0f}"
)
d.destroy_process_group()
PYEOF
SMOKE_RC=0
CUDA_VISIBLE_DEVICES="$GPUS" \
torchrun --master_port="${SMOKE_PORT:-7778}" --nproc_per_node="$NPROC" \
  "$SMOKE_SCRIPT" 2>&1 | tee "$DIAG_DIR/smoke_$STAMP.log" || SMOKE_RC=$?
rm -f "$SMOKE_SCRIPT"
if [ "$SMOKE_RC" -ne 0 ]; then
  echo "[diag] FATAL: 分布式初始化冒烟失败（见上方输出）——NCCL/torchrun 环境问题，训练必然无法多卡，先修环境再跑。"
  exit "$SMOKE_RC"
fi
echo "[diag] 冒烟通过。"

# ── 3. torchrun 双卡启动（tee 全量日志）─────────────────────────────────
rm -f "$DIAG_DIR"/heartbeat_rank*.jsonl
CUDA_VISIBLE_DEVICES="$GPUS" \
torchrun --master_port="${MASTER_PORT:-7777}" --nproc_per_node="$NPROC" \
  tools/train/train.py -c "$CONFIG" "${UPDATE_ARGS[@]}" --seed=0 \
  2>&1 | tee "$DIAG_DIR/train_full_$STAMP.log"
RUN_RC="${PIPESTATUS[0]}"
echo "[diag] 训练退出码: $RUN_RC"

# ── 4. 产物打包（无论成败）──────────────────────────────────────────────
BUNDLE="$DIAG_DIR/bundle_${STAMP}.tar.gz"
tar czf "$BUNDLE" -C "$DIAG_DIR" \
  env_snapshot.txt "train_full_$STAMP.log" "smoke_$STAMP.log" \
  $(cd "$DIAG_DIR" && ls heartbeat_rank*.jsonl 2>/dev/null) \
  nccl_flight_dump 2>/dev/null
echo "[diag] 现场已打包: $BUNDLE"
echo "[diag] 复现与定位指南: docs/engineering/multigpu-diagnosis_CN.md"
exit "$RUN_RC"
