#!/bin/sh
{
python train.py \
    --eval \
    --source_path /dev/shm/ScalarReal \
    --config configs/hyfluid/scalar_real_simple_color_scale_rotation_act.json \
    --model_path log/hyfluid_scalar_real/simple_pillar_small_imgoffset_grey_color_scale_rotation_act_trbfcInit0_no_prune_N120x5e4 \
    --loader hyfluid
exit
}
