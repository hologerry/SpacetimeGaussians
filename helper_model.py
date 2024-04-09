# MIT License

# Copyright (c) 2023 OPPO

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
import torch
import torch.nn as nn

from mmcv.ops import knn
from simple_knn._C import distCUDA2

from thirdparty.gaussian_splatting.utils.graphics_utils import BasicPointCloud


class Sandwich(nn.Module):
    def __init__(self, dim, outdim=3, bias=False):
        super().__init__()

        self.mlp1 = nn.Conv2d(12, 6, kernel_size=1, bias=bias)

        self.mlp2 = nn.Conv2d(6, 3, kernel_size=1, bias=bias)
        self.relu = nn.ReLU()

        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, input, rays, time=None):
        albedo, spec, time_feature = input.chunk(3, dim=1)
        specular = torch.cat([spec, time_feature, rays], dim=1)  # 3+3 + 5
        specular = self.mlp1(specular)
        specular = self.relu(specular)
        specular = self.mlp2(specular)

        result = albedo + specular
        result = self.sigmoid(result)
        return result


class Sandwichnoact(nn.Module):
    def __init__(self, dim, outdim=3, bias=False):
        super().__init__()

        self.mlp1 = nn.Conv2d(12, 6, kernel_size=1, bias=bias)
        self.mlp2 = nn.Conv2d(6, 3, kernel_size=1, bias=bias)
        self.relu = nn.ReLU()

    def forward(self, input, rays, time=None):
        albedo, spec, time_feature = input.chunk(3, dim=1)
        specular = torch.cat([spec, time_feature, rays], dim=1)  # 3+3 + 5
        specular = self.mlp1(specular)
        specular = self.relu(specular)
        specular = self.mlp2(specular)

        result = albedo + specular
        result = torch.clamp(result, min=0.0, max=1.0)
        return result


class Sandwichnoactss(nn.Module):
    def __init__(self, dim, outdim=3, bias=False):
        super().__init__()

        self.mlp2 = nn.Conv2d(12, 6, kernel_size=1, bias=bias)
        self.mlp3 = nn.Conv2d(6, 3, kernel_size=1, bias=bias)

        self.relu = nn.ReLU()

    def forward(self, input, rays, time=None):
        albedo, spec, time_feature = input.chunk(3, dim=1)
        specular = torch.cat([spec, time_feature, rays], dim=1)  # 3+3 + 5
        specular = self.mlp2(specular)
        specular = self.relu(specular)
        specular = self.mlp3(specular)

        result = albedo + specular
        return result


####### following are also good rgb model but not used in the paper, slower than sandwich, inspired by color shift in hyperreel
# remove sigmoid for immersive dataset
class RGBDecoderVRayShift(nn.Module):
    def __init__(self, dim, outdim=3, bias=False):
        super().__init__()

        self.mlp1 = nn.Conv2d(dim, outdim, kernel_size=1, bias=bias)
        self.mlp2 = nn.Conv2d(15, outdim, kernel_size=1, bias=bias)
        self.mlp3 = nn.Conv2d(6, outdim, kernel_size=1, bias=bias)
        self.sigmoid = torch.nn.Sigmoid()

        self.dw_conv = nn.Conv2d(9, 9, kernel_size=1, bias=bias)

    def forward(self, input, rays, t=None):
        x = self.dw_conv(input) + input
        albedo = self.mlp1(x)
        specular = torch.cat([x, rays], dim=1)
        specular = self.mlp2(specular)

        final_feature = torch.cat([albedo, specular], dim=1)
        result = self.mlp3(final_feature)
        result = self.sigmoid(result)
        return result


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

            nearest_neighbour_idx = xyz_nn_points[
                0, 1
            ].long()  # N x 1  skip the first one, we select the second closest one
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


def get_color_model(rgb_function):
    if rgb_function == "sandwich":
        rgb_decoder = Sandwich(9, 3)

    elif rgb_function == "sandwichnoact":
        rgb_decoder = Sandwichnoact(9, 3)

    elif rgb_function == "sandwichnoactss":
        rgb_decoder = Sandwichnoactss(9, 3)

    else:
        return None
    return rgb_decoder


def pix2ndc(v, S):
    return (v * 2.0 + 1.0) / S - 1.0


def ndc2pix(v, S):
    return ((v + 1.0) * S - 1.0) * 0.5
