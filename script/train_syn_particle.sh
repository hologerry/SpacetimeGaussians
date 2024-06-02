#!/bin/sh
{
python train.py \
    --source_path /home/yuegao/Dynamics/synthetic_particle \
    --config configs/synthetic_particle/synthetic_particle_simple_color_scale_rotation_opacity_act.json \
    --model_path /data/Dynamics/SpacetimeGaussiansLog/synthetic_particle/simple_1_point_grey_color_scale-4_rotation_opacity_act_trbfcInit0_small_N10_simple_trajectory_no_clone_split \
    --loader synthetic_particle
exit
}
