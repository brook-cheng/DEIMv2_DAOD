#!/bin/bash
set -e
cd /mnt/d/cx/thired/deimv2_daod
# COMET_API_KEY 必须来自环境（密钥已轮换，禁止写回脚本）
# export COMET_API_KEY="..."
# 020 密度族已在 OBB 消融清理中删除（synthetic_exp_020.yml 不存在）
for d in 010; do
  echo "==== [A] density_${d} $(date) ===="
  python tools/train/train.py --config "configs/custom_obb/synthetic_configs/synthetic_exp_${d}.yml"
  echo "==== [A] density_${d} DONE $(date) ===="
done
echo "=== GROUP A DONE ==="
