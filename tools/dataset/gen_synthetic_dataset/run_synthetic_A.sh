#!/bin/bash
set -e
cd /mnt/d/cx/thired/deimv2_daod
# COMET_API_KEY 必须来自环境（密钥已轮换，禁止写回脚本）
# export COMET_API_KEY="..."
for d in 010 020; do
  echo "==== [A] density_${d} $(date) ===="
  python train.py --config "configs/custom_obb/synthetic_exp_${d}.yml"
  echo "==== [A] density_${d} DONE $(date) ===="
done
echo "=== GROUP A DONE ==="
