#!/bin/bash
set -e
cd /mnt/d/cx/thired/deimv2_daod
# COMET_API_KEY 必须来自环境（密钥已轮换，禁止写回脚本）
# export COMET_API_KEY="..."
echo "==== [B] density_100 $(date) ===="
python tools/train/train.py --config "configs/custom_obb/synthetic_configs/synthetic_exp_100.yml"
echo "==== [B] density_100 DONE $(date) ===="
echo "=== GROUP B DONE ==="
