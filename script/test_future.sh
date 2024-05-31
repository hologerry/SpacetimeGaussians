#!/bin/sh
{
python test_future.py \
    --source_path /dev/shm/ScalarReal \
    --config configs/hyfluid/scalar_real_simple_color_scale_rotation_act.json \
    --model_path log/hyfluid_scalar_real/simple_pillar_small_imgoffset_grey_color_scale_rotation_act_trbfcInit0 \
    --val_loader hyfluid_valid
exit
}
