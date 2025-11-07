DEVICE='0'
MDOEL='x'
export CUDA_VISIBLE_DEVICES=$DEVICE
echo $CUDA_VISIBLE_DEVICES
python3 train.py -c configs/deimv2/deimv2_dinov3_${MDOEL}_custom.yml --use-amp --seed=0