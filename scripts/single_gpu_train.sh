# export CUDA_VISIBLE_DEVICES=$DEVICE
# echo $CUDA_VISIBLE_DEVICES
python3 train.py --config ./configs/custom_obb/deimv2_obb_sp_jyz.yml --device "cuda:0"  --seed=0