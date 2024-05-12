python test.py \
    --eval \
    --skip_train \
    --config configs/hyfluid/scalar_real_lite.json \
    --source /dev/shm/ScalarReal \
    --model_path log/hyfluid_scalar_real/colmap_0_120_1_bbox_lite_fixCam_fixInit_pillar_r003_ymin-005_ymax015_raw-op-reset_grey \
    --val_loader hyfluid_valid
