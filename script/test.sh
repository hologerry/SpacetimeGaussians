python test.py \
    --eval \
    --skip_train \
    --config configs/hyfluid/scalar_real_lite_zero123_vis.json \
    --source /dev/shm/ScalarReal103 \
    --model_path log/hyfluid_scalar_real/sim103_colmap_0_120_1_bbox_lite_zero123_grey_view0_fake134_l1_loss \
    --val_loader hyfluid_valid
