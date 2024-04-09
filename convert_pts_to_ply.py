import numpy as np
import open3d as o3d


pts_path = "cam_vis/input_xyz.npy"
points = np.load(pts_path)
points = points[:, :3]
print(points.shape)

# Convert the NumPy array to a PointCloud
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

# Save the PointCloud to a .ply file
o3d.io.write_point_cloud("cam_vis/input_points.ply", pcd)
