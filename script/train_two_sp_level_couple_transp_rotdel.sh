#!/bin/sh
{
python train_two_sp_level.py \
    --source_path /dev/shm/ScalarReal \
    --config configs/gaussian_fluid/scalar_real_simple_color_scale_rotation_act_two_sp_level_couple_transp_rotdel.json \
    --model_path /data/Dynamics/SpacetimeGaussiansLog/gaussian_fluid_scalar_real/two_sp_level_couple_transp_rotdel_L1_default_L2_default-delrotrs1vrand_lrr0025_lrv002 \
    --loader hyfluid
exit
}

