# export CUDA_VISIBLE_DEVICES=$DEVICE
# echo $CUDA_VISIBLE_DEVICES
python3 train.py --config ./configs/custom/deimv2_dinov3_vitl16_freeze_test_eiou_attenRes.yml --device "cuda:1" --use-amp  --seed=0