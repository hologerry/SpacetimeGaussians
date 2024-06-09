import numpy as np
import torch

from mmcv.ops import knn

from gaussian_splatting.gaussian.gaussian_model import GaussianModel
from gaussian_splatting.utils.graphics_utils import BasicPointCloud


def get_model(model="full") -> GaussianModel:
    if model == "full":
        from gaussian_splatting.gaussian.gm_full import GaussianModel

    elif model == "lite":
        from gaussian_splatting.gaussian.gm_lite import GaussianModel

    elif model == "lite_act":
        from gaussian_splatting.gaussian.gm_lite_act import GaussianModel

    elif model == "simple_scale":
        from gaussian_splatting.gaussian.gm_simple_scale import GaussianModel

    elif model == "simple_rotation":
        from gaussian_splatting.gaussian.gm_simple_rotation import GaussianModel

    elif model == "simple_color":
        from gaussian_splatting.gaussian.gm_simple_color import GaussianModel

    elif model == "simple_color_scale_rotation":
        from gaussian_splatting.gaussian.gm_simple_color_scale_rotation import (
            GaussianModel,
        )

    elif model == "simple_color_scale_rotation_act":
        from gaussian_splatting.gaussian.gm_simple_color_scale_rotation_act import (
            GaussianModel,
        )

    elif model == "simple_color_scale_rotation_act_two_sp_level":
        from gaussian_splatting.gaussian.gm_simple_color_scale_rotation_act_two_sp_level import (
            GaussianModel,
        )

    elif model == "simple_color_scale_rotation_act_two_sp_level_couple":
        from gaussian_splatting.gaussian.gm_simple_color_scale_rotation_act_two_sp_level_couple import (
            GaussianModel,
        )

    elif model == "simple_color_scale_rotation_act_two_sp_level_couple_transp":
        from gaussian_splatting.gaussian.gm_simple_color_scale_rotation_act_two_sp_level_couple_transp import (
            GaussianModel,
        )

    elif model == "simple_color_scale_rotation_act_two_sp_level_couple_transp_zerodel":
        from gaussian_splatting.gaussian.gm_simple_color_scale_rotation_act_two_sp_level_couple_transp_zerodel import (
            GaussianModel,
        )

    elif model == "simple_color_scale_rotation_act_two_sp_level_couple_transp_zerodel_trbfs":
        from gaussian_splatting.gaussian.gm_simple_color_scale_rotation_act_two_sp_level_couple_transp_zerodel_trbfs import (
            GaussianModel,
        )

    elif model == "simple_color_scale_rotation_act_two_sp_level_couple_transp_rotdel":
        from gaussian_splatting.gaussian.gm_simple_color_scale_rotation_act_two_sp_level_couple_transp_rotdel import (
            GaussianModel,
        )

    elif model == "simple_color_scale_rotation_act_two_sp_level_couple_transp_rotdel_trbfs":
        from gaussian_splatting.gaussian.gm_simple_color_scale_rotation_act_two_sp_level_couple_transp_rotdel_trbfs import (
            GaussianModel,
        )

    elif model == "simple_color_scale_rotation_act_two_sp_level_couple_transp_trotdel_trbfs":
        from gaussian_splatting.gaussian.gm_simple_color_scale_rotation_act_two_sp_level_couple_transp_trotdel_trbfs import (
            GaussianModel,
        )

    elif model == "simple_color_scale_rotation_act_two_sp_level_couple_transp_sindel_trbfs":
        from gaussian_splatting.gaussian.gm_simple_color_scale_rotation_act_two_sp_level_couple_transp_sindel_trbfs import (
            GaussianModel,
        )

    elif model == "simple_color_scale_rotation_act_two_sp_level_couple_transp_linsindel_trbfs":
        from gaussian_splatting.gaussian.gm_simple_color_scale_rotation_act_two_sp_level_couple_transp_linsindel_trbfs import (
            GaussianModel,
        )

    elif model == "simple_color_scale_rotation_act_two_sp_level_couple_transp_rigsurdel_trbfs":
        from gaussian_splatting.gaussian.gm_simple_color_scale_rotation_act_two_sp_level_couple_transp_rigsurdel_trbfs import (
            GaussianModel,
        )

    elif model == "simple_opacity_no_t":
        from gaussian_splatting.gaussian.gm_simple_opacity_no_t import GaussianModel

    elif model == "simple_opacity_linear":
        from gaussian_splatting.gaussian.gm_simple_opacity_linear import GaussianModel

    elif model == "simple_opacity_w_t":
        from gaussian_splatting.gaussian.gm_simple_opacity_w_t import GaussianModel

    elif model == "simple_xyz_quadric":
        from gaussian_splatting.gaussian.gm_simple_xyz_quadric import GaussianModel

    elif model == "simple_xyz_linear":
        from gaussian_splatting.gaussian.gm_simple_xyz_linear import GaussianModel

    elif model == "simple_xyz_linear_color":
        from gaussian_splatting.gaussian.gm_simple_xyz_linear_color import GaussianModel

    elif model == "simple_xyz_linear_color_trbf_c_act":
        from gaussian_splatting.gaussian.gm_simple_xyz_linear_color_trbf_c_act import (
            GaussianModel,
        )

    elif model == "simple_fix_xyz_linear_color_trbf_c_act":
        from gaussian_splatting.gaussian.gm_simple_fix_xyz_linear_color_trbf_c_act import (
            GaussianModel,
        )

    elif model == "simple_xyz_linear_color_source":
        from gaussian_splatting.gaussian.gm_simple_xyz_linear_color_source import (
            GaussianModel,
        )

    elif model == "simple_xyz_linear_color_trbf_center":
        from gaussian_splatting.gaussian.gm_simple_xyz_linear_color_trbf_center import (
            GaussianModel,
        )

    elif model == "simple_trbf_center":
        from gaussian_splatting.gaussian.gm_simple_trbf_center import GaussianModel

    elif model == "simple_xyz_quadric_trbf_center":
        from gaussian_splatting.gaussian.gm_simple_xyz_quadric_trbf_center import (
            GaussianModel,
        )

    elif model == "simple_all":
        from gaussian_splatting.gaussian.gm_simple_all import GaussianModel

    elif model == "lite_two_level":
        from gaussian_splatting.gaussian.gm_lite_two_level import GaussianModel

    else:
        raise ValueError(f"Model {model} not found")

    return GaussianModel


def interpolate_point(pcd, N=4):

    old_xyz = pcd.points
    old_color = pcd.colors
    old_normal = pcd.normals
    old_time = pcd.times

    time_stamps = np.unique(old_time)

    new_xyz = []
    new_color = []
    new_normal = []
    new_time = []
    for time_idx, time in enumerate(time_stamps):
        selected_mask = old_time == time
        selected_mask = selected_mask.squeeze(1)

        if time_idx == 0:
            new_xyz.append(old_xyz[selected_mask])
            new_color.append(old_color[selected_mask])
            new_normal.append(old_normal[selected_mask])
            new_time.append(old_time[selected_mask])
        else:
            xyz_input = old_xyz[selected_mask]
            xyz_input = torch.from_numpy(xyz_input).float().cuda()
            xyz_input = xyz_input.unsqueeze(0).contiguous()  # 1 x N x 3
            xyz_nn_points = knn(2, xyz_input, xyz_input, False)

            nearest_neighbour_idx = xyz_nn_points[0, 1].long()  # N x 1
            spatial_distance = torch.norm(xyz_input - xyz_input[:, nearest_neighbour_idx, :], dim=2)  #  1 x N
            spatial_distance = spatial_distance.squeeze(0)

            diff_sorted, _ = torch.sort(spatial_distance)
            N = spatial_distance.shape[0]
            num_take = int(N * 0.25)
            masks = spatial_distance > diff_sorted[-num_take]
            masks_numpy = masks.cpu().numpy()

            new_xyz.append(old_xyz[selected_mask][masks_numpy])
            new_color.append(old_color[selected_mask][masks_numpy])
            new_normal.append(old_normal[selected_mask][masks_numpy])
            new_time.append(old_time[selected_mask][masks_numpy])

    new_xyz = np.concatenate(new_xyz, axis=0)
    new_color = np.concatenate(new_color, axis=0)
    new_time = np.concatenate(new_time, axis=0)
    assert new_xyz.shape[0] == new_color.shape[0]

    new_pcd = BasicPointCloud(points=new_xyz, colors=new_color, normals=None, times=new_time)

    return new_pcd


def interpolate_point_v3(pcd, N=4, m=0.25):

    old_xyz = pcd.points
    old_color = pcd.colors
    old_normal = pcd.normals
    old_time = pcd.times

    time_stamps = np.unique(old_time)

    new_xyz = []
    new_color = []
    new_normal = []
    new_time = []
    for time_idx, time in enumerate(time_stamps):
        selected_mask = old_time == time
        selected_mask = selected_mask.squeeze(1)

        if time_idx % N == 0:
            new_xyz.append(old_xyz[selected_mask])
            new_color.append(old_color[selected_mask])
            new_normal.append(old_normal[selected_mask])
            new_time.append(old_time[selected_mask])

        else:
            xyz_input = old_xyz[selected_mask]
            xyz_input = torch.from_numpy(xyz_input).float().cuda()
            xyz_input = xyz_input.unsqueeze(0).contiguous()  # 1 x N x 3
            xyz_nn_points = knn(2, xyz_input, xyz_input, False)

            nearest_neighbour_idx = xyz_nn_points[
                0, 1
            ].long()  # N x 1  skip the first one, we select the second closest one
            spatial_distance = torch.norm(xyz_input - xyz_input[:, nearest_neighbour_idx, :], dim=2)  #  1 x N
            spatial_distance = spatial_distance.squeeze(0)

            diff_sorted, _ = torch.sort(spatial_distance)
            M = spatial_distance.shape[0]
            num_take = int(M * m)
            masks = spatial_distance > diff_sorted[-num_take]
            masks_numpy = masks.cpu().numpy()

            new_xyz.append(old_xyz[selected_mask][masks_numpy])
            new_color.append(old_color[selected_mask][masks_numpy])
            new_normal.append(old_normal[selected_mask][masks_numpy])
            new_time.append(old_time[selected_mask][masks_numpy])
            #
    new_xyz = np.concatenate(new_xyz, axis=0)
    new_color = np.concatenate(new_color, axis=0)
    new_time = np.concatenate(new_time, axis=0)
    assert new_xyz.shape[0] == new_color.shape[0]

    new_pcd = BasicPointCloud(points=new_xyz, colors=new_color, normals=None, times=new_time)

    return new_pcd


def interpolate_part_use(pcd, N=4):
    # used in ablation study
    old_xyz = pcd.points
    old_color = pcd.colors
    old_normal = pcd.normals
    old_time = pcd.times

    time_stamps = np.unique(old_time)

    new_xyz = []
    new_color = []
    new_normal = []
    new_time = []
    for time_idx, time in enumerate(time_stamps):
        selected_mask = old_time == time
        selected_mask = selected_mask.squeeze(1)

        if time_idx % N == 0:
            new_xyz.append(old_xyz[selected_mask])
            new_color.append(old_color[selected_mask])
            new_normal.append(old_normal[selected_mask])
            new_time.append(old_time[selected_mask])

        else:
            pass
            #
    new_xyz = np.concatenate(new_xyz, axis=0)
    new_color = np.concatenate(new_color, axis=0)
    new_time = np.concatenate(new_time, axis=0)
    assert new_xyz.shape[0] == new_color.shape[0]

    new_pcd = BasicPointCloud(points=new_xyz, colors=new_color, normals=None, times=new_time)

    return new_pcd


def padding_point(pcd, N=4):

    old_xyz = pcd.points
    old_color = pcd.colors
    old_normal = pcd.normals
    old_time = pcd.times

    time_stamps = np.unique(old_time)
    total_length = len(time_stamps)

    new_xyz = []
    new_color = []
    new_normal = []
    new_time = []
    for time_idx, time in enumerate(time_stamps):
        selected_mask = old_time == time
        selected_mask = selected_mask.squeeze(1)

        if time_idx != 0 and time_idx != len(time_stamps) - 1:
            new_xyz.append(old_xyz[selected_mask])
            new_color.append(old_color[selected_mask])
            new_normal.append(old_normal[selected_mask])
            new_time.append(old_time[selected_mask])

        else:
            new_xyz.append(old_xyz[selected_mask])
            new_color.append(old_color[selected_mask])
            new_normal.append(old_normal[selected_mask])
            new_time.append(old_time[selected_mask])

            xyz_input = old_xyz[selected_mask]
            xyz_input = torch.from_numpy(xyz_input).float().cuda()
            xyz_input = xyz_input.unsqueeze(0).contiguous()  # 1 x N x 3

            xyz_nn_points = knn(2, xyz_input, xyz_input, False)

            # N x 1  skip the first one, we select the second closest one
            nearest_neighbour_idx = xyz_nn_points[0, 1].long()
            spatial_distance = torch.norm(xyz_input - xyz_input[:, nearest_neighbour_idx, :], dim=2)  #  1 x N
            spatial_distance = spatial_distance.squeeze(0)

            diff_sorted, _ = torch.sort(spatial_distance)
            N = spatial_distance.shape[0]
            num_take = int(N * 0.125)
            masks = spatial_distance > diff_sorted[-num_take]
            masks_numpy = masks.cpu().numpy()

            new_xyz.append(old_xyz[selected_mask][masks_numpy])
            new_color.append(old_color[selected_mask][masks_numpy])
            new_normal.append(old_normal[selected_mask][masks_numpy])

            if time_idx == 0:
                new_time.append(old_time[selected_mask][masks_numpy] - (1 / total_length))
            else:
                new_time.append(old_time[selected_mask][masks_numpy] + (1 / total_length))
    new_xyz = np.concatenate(new_xyz, axis=0)
    new_color = np.concatenate(new_color, axis=0)
    new_time = np.concatenate(new_time, axis=0)
    assert new_xyz.shape[0] == new_color.shape[0]

    new_pcd = BasicPointCloud(points=new_xyz, colors=new_color, normals=None, times=new_time)

    return new_pcd
