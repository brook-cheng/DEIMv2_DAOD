#!/bin/bash
set -e
cd /mnt/d/cx/thired/deimv2_daod
export COMET_API_KEY="EoSgIYtwa6a5rKElgh9KD59xS"
for d in 010 020; do
  echo "==== [A] density_${d} $(date) ===="
  python train.py --config "configs/custom_obb/synthetic_exp_${d}.yml"
  echo "==== [A] density_${d} DONE $(date) ===="
done
echo "=== GROUP A DONE ==="
