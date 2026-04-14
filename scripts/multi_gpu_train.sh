# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL
CUDA_VISIBLE_DEVICES=0,1 torchrun --master_port=7777 --nproc_per_node=2 train.py -c ./configs/custom/deimv2_dinov3_vith16p_freeze_test_eiou_attenRes.yml  --use-amp --seed=0
