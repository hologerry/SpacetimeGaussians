#!/bin/sh
{
python train_two_sp_level.py \
    --source_path /dev/shm/ScalarReal \
    --config configs/gaussian_fluid/scalar_real_simple_color_scale_rotation_act_two_sp_level_couple_transp_sindel_trbfs_color.json \
    --model_path /data/Dynamics/SpacetimeGaussiansLog/gaussian_fluid_scalar_real/two_sp_level_couple_transp_sindel_trbfs1_color_L1_default_L2_default_sina1e-5o5p0 \
    --loader hyfluid
exit
}

