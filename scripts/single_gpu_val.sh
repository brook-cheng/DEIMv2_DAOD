DEVICE='cuda:1'
MODEL_TYPE='x'
MODEL_PATH='outputs/deimv2_dinov3_x_custom/best_stg2__freeze_1109_e186_mAP67.pth'

python3 tools/train/train.py -c configs/deimv2/deimv2_dinov3_${MODEL_TYPE}_custom_val.yml --test-only -r ${MODEL_PATH}