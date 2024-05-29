#!/bin/sh
{
python test.py \
    --eval \
    --skip_train \
    --source_path /dev/shm/ScalarReal \
    --config configs/hyfluid/scalar_real_simple_color_scale_rotation.json \
    --model_path log/hyfluid_scalar_real/simple_pillar_small_imgoffset_grey_color_scale_rotation_trbfcInit0_no_prune \
    --val_loader hyfluid_valid
exit
}
