python test.py \
    --eval \
    --skip_train \
    --config configs/hyfluid/scalar_real_lite_velocity.json \
    --source /data/Dynamics/ScalarReal \
    --model_path log/hyfluid_scalar_real/colmap_0_120_1_bbox_lite_fixCam_fixInit_pillar_raw-op-reset \
    --val_loader hyfluid_valid
