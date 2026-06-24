#!/bin/bash
set -e
cd /mnt/d/cx/thired/deimv2_daod
export COMET_API_KEY="EoSgIYtwa6a5rKElgh9KD59xS"
echo "==== [B] density_100 $(date) ===="
python train.py --config "configs/custom_obb/synthetic_exp_100.yml"
echo "==== [B] density_100 DONE $(date) ===="
echo "=== GROUP B DONE ==="
