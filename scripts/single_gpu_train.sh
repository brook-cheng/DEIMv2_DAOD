# export CUDA_VISIBLE_DEVICES=$DEVICE
# echo $CUDA_VISIBLE_DEVICES
python3 tools/train/train.py --config ./configs/custom_obb/jyz/sp_ft_rep0.yml --device "cuda:0"  --seed=0