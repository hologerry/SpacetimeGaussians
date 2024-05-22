python test.py \
    --eval \
    --skip_train \
    --source_path /dev/shm/ScalarReal \
    --config configs/hyfluid/scalar_real_simple_xyz_linear_color.json \
    --model_path log/hyfluid_scalar_real/simple_pillar_large_imgoffset_grey_xyz_linear_color \
    --val_loader hyfluid_valid
