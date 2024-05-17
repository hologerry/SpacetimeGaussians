python test.py \
    --eval \
    --skip_train \
    --source /dev/shm/ScalarReal \
    --config configs/hyfluid/scalar_real_simple_xyz_linear_color_source.json \
    --model_path log/hyfluid_scalar_real/simple_pillar_source_imgoffset_grey_xyz_linear_color_source_s0d1 \
    --val_loader hyfluid_valid
