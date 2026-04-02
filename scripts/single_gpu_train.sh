# export CUDA_VISIBLE_DEVICES=$DEVICE
# echo $CUDA_VISIBLE_DEVICES
python3 train.py --config ./configs/custom/deimv2_dinov3_x_freeze_test_iou.yml --device "cuda:0" --use-amp  --seed=0