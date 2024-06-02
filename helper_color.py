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

import torch
import torch.nn as nn


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
