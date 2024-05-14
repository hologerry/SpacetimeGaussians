python test.py \
    --eval \
    --skip_train \
    --config configs/hyfluid/scalar_real_simple_color.json \
    --source /dev/shm/ScalarReal \
    --model_path log/hyfluid_scalar_real/simple_pillar_r018_ymin-005_ymax07_grey_color0 \
    --val_loader hyfluid_valid
