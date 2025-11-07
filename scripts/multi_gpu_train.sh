# DEVICE='0,1,2,3'
DEVICE='0'
MDOEL='x'
# export CUDA_VISIBLE_DEVICES=$DEVICE
# echo $CUDA_VISIBLE_DEVICES
CUDA_VISIBLE_DEVICES=${DEVICE} torchrun --master_port=7777 --nproc_per_node=4 train.py -c configs/deimv2/deimv2_dinov3_${MDOEL}_custom.yml --use-amp --seed=0
