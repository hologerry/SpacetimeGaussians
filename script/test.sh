python test.py \
    --eval \
    --skip_train \
    --source_path /dev/shm/ScalarReal \
    --config configs/hyfluid/scalar_real_simple_xyz_linear_color_trbf_c_act.json \
    --model_path log/hyfluid_scalar_real/simple_pillar_small_imgoffset_grey_xyz_linear_color_trbf_c_act_bs1 \
    --val_loader hyfluid_valid
