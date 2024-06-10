#!/bin/sh
{
python train.py \
    --source_path /dev/shm/ScalarReal \
    --config configs/hyfluid/scalar_real_simple_color_scale_rotation_act.json \
    --model_path /data/Dynamics/SpacetimeGaussiansLog/hyfluid_scalar_real/simple_pillar_small_imgoffset_grey_color_scale_rotation_act_trbfcInit0_desify3e4 \
    --loader hyfluid
exit
}
