#!/bin/sh
{
python train.py \
    --source_path /dev/shm/ScalarReal \
    --config configs/gaussian_fluid/scalar_real_lite_two_level.json \
    --model_path /data/Dynamics/SpacetimeGaussiansLog/gaussian_fluid_scalar_real/two_level_lite_L1_15e3_N120x5_nocsp_L2_30e3_densi15e3-14e3_csp-nosplitp \
    --loader hyfluid
exit
}
