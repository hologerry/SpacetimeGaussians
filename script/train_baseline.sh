#!/bin/sh
{
python train.py \
    --source_path /dev/shm/ScalarReal \
    --config configs/hyfluid/scalar_real_lite.json \
    --model_path /data/Dynamics/SpacetimeGaussiansLog/hyfluid_scalar_real/baseline_pillar_small_imgoffset_grey_N120_no-clone-split \
    --loader hyfluid
exit
}
