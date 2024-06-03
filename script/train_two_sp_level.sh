#!/bin/sh
{
python train_two_sp_level.py \
    --source_path /dev/shm/ScalarReal \
    --config configs/gaussian_fluid/scalar_real_simple_color_scale_rotation_two_sp_level_act.json \
    --model_path /data/Dynamics/SpacetimeGaussiansLog/gaussian_fluid_scalar_real/two_sp_level_L1_co-sc-ro-tci0_N120x10_csp_densi0-5e3_pprune_L2_N120x100-op01-co06-sc--5_csp_densi5e3-1e4_noact \
    --loader hyfluid
exit
}
# --model_path /data/Dynamics/SpacetimeGaussiansLog/gaussian_fluid_scalar_real/two_sp_level_L1_co-sc-ro-tci0_N120x10_csp_densi0-9e3_pprune_L2_N120x100-op06-co06-xyzlar_csp_densi0-8e3_no-act \
