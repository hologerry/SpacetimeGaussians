#!/bin/sh
{
python test_two_sp_level.py \
    --test \
    --source_path /dev/shm/ScalarReal \
    --config configs/gaussian_fluid/scalar_real_simple_color_scale_rotation_act_two_sp_level_couple_transp_rotdel_trbfs_color.json \
    --model_path /data/Dynamics/SpacetimeGaussiansLog/gaussian_fluid_scalar_real/two_sp_level_couple_transp_rotdel_trbfs1_color_L1_default_L2_default_rot-xyz \
    --val_loader hyfluid_valid
exit
}

