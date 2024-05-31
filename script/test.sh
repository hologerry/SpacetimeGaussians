#!/bin/sh
{
python test.py \
    --source_path /dev/shm/ScalarReal \
    --config configs/hyfluid/scalar_real_lite_act.json \
    --model_path log/hyfluid_scalar_real/baseline_pillar_small_imgoffset_grey_act_trbfcInit0_N120x10000_no-clone-split-prune \
    --val_loader hyfluid_valid
exit
}
