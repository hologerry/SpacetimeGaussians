#!/bin/sh
{
python test.py \
    --eval \
    --skip_train \
    --source_path /home/yuegao/Dynamics/synthetic_particle \
    --config configs/synthetic_particle/synthetic_particle_simple_color_scale_rotation_opacity_act.json \
    --model_path log/synthetic_particle/simple_1_point_grey_color_scale-5_rotation_opacity_act_trbfcInit0_small_N1_simple_trajectory_no_clone_split \
    --val_loader synthetic_particle_valid
exit
}
