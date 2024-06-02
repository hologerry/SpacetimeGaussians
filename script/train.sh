#!/bin/sh
{
python train.py \
    --source_path /dev/shm/ScalarReal \
    --config configs/gaussian_fluid/scalar_real_lite_two_level.json \
    --model_path /data/Dynamics/SpacetimeGaussiansLog/gaussian_fluid_scalar_real/baseline_pillar_small_imgoffset_grey_L0-N120x5-no-csp_L1-csp-nosplitp_zero-grad-0 \
    --loader hyfluid
exit
}
