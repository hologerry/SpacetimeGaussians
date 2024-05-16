python test.py \
    --eval \
    --skip_train \
    --source /dev/shm/ScalarReal \
    --config configs/hyfluid/scalar_real_simple_xyz_linear_color.json \
    --model_path log/hyfluid_scalar_real/simple_pillar_r018_ymin-005_ymax07_grey_xyz_linear_color \
    --val_loader hyfluid_valid
