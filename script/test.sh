python test.py \
    --eval \
    --skip_train \
    --config configs/hyfluid/scalar_real_simple_xyz_linear.json \
    --source /dev/shm/ScalarReal \
    --model_path log/hyfluid_scalar_real/simple_pillar_r018_ymin-005_ymax07_grey_xyz_linear \
    --val_loader hyfluid_valid
