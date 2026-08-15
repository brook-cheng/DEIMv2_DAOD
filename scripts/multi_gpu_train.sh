# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL
CUDA_VISIBLE_DEVICES=0,1 torchrun --master_port=7777 --nproc_per_node=2 tools/train/train.py -c ./configs/custom/deimv2_dinov3_vith16p_freeze.yml  --seed=0
