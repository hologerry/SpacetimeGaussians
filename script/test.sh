#!/bin/sh
{
python test.py \
    --test \
    --source_path /dev/shm/ScalarReal \
    --config configs/hyfluid/scalar_real_simple_color_scale_rotation_act.json \
    --model_path /data/Dynamics/SpacetimeGaussiansLog/hyfluid_scalar_real/simple_pillar_small_imgoffset_grey_color_scale_rotation_act_trbfcInit0_N120x10_densify5000_postpruneall \
    --val_loader hyfluid_valid
exit
}
