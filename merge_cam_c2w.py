import os

import numpy as np


cam_names = ["train00", "train01", "train03", "train04"]

all_poses = []
for cam_name in cam_names:
    cam_pose_npy = f"cam_vis/cam_{cam_name}_pose.npy"
    cam_pose = np.load(cam_pose_npy)
    all_poses.append(cam_pose)

all_pose = np.stack(all_poses, axis=0)

np.save("cam_vis/all_c2w_poses.npy", all_pose)
