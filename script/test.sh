python test.py \
    --eval \
    --skip_train \
    --source /dev/shm/ScalarReal \
    --config configs/hyfluid/scalar_real_simple_xyz_quadric_trbf_c.json \
    --model_path log/hyfluid_scalar_real/simple_pillar_imgoffset_grey_xyz_quadric_trbf_center \
    --val_loader hyfluid_valid
