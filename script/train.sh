#!/bin/sh
{
python train.py \
    --source_path /dev/shm/ScalarReal \
    --config configs/hyfluid/scalar_real_lite_act.json \
    --model_path log/hyfluid_scalar_real/baseline_pillar_small_imgoffset_grey_act_trbfcInit0_N120x5000_no-clone-split-prune \
    --loader hyfluid
exit
}
