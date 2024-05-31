import json
import os
import random
import shutil
import sys
import time
import uuid

from argparse import ArgumentParser, Namespace
from random import randint

import cv2
import numpy as np
import torch

from tqdm import tqdm

from gaussian_splatting.arguments import (
    ModelParams,
    OptimizationParams,
    PipelineParams,
    get_combined_args,
)
from gaussian_splatting.utils.general_utils import safe_state


def get_parser():
    parser = ArgumentParser(description="Training script parameters")
    mp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6029)
    parser.add_argument("--debug_from", type=int, default=-2)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)

    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 15_000, 30_000])
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 15_000, 30_000])
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])

    parser.add_argument("--quiet", action="store_true")

    parser.add_argument("--start_checkpoint", type=str, default=None)

    parser.add_argument("--config_path", type=str, default="None")

    args = parser.parse_args(sys.argv[1:])

    if args.iterations not in args.save_iterations:
        args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    # incase we provide config file not directly pass to the file
    if os.path.exists(args.config_path) and args.config_path != "None":
        print("overload config from " + args.config_path)
        config = json.load(open(args.config_path))
        for k in config.keys():
            try:
                value = getattr(args, k)
                newvalue = config[k]
                setattr(args, k, newvalue)
            except:
                print("failed set config: " + k)
        print("finish load config from " + args.config_path)
    else:
        raise ValueError("config file not exist or not provided")

    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    return args, mp.extract(args), op.extract(args), pp.extract(args)


def get_test_parser():
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)

    parser.add_argument("--test_iteration", default=-1, type=int)

    parser.add_argument("--val_loader", type=str, default="colmap")
    parser.add_argument("--config_path", type=str, default="1")

    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    safe_state(args.quiet)

    if os.path.exists(args.config_path) and args.config_path != "None":
        print("overload config from " + args.config_path)
        config = json.load(open(args.config_path))
        for k in config.keys():
            try:
                value = getattr(args, k)
                newvalue = config[k]
                setattr(args, k, newvalue)
            except:
                print("failed set config: " + k)
        print("finish load config from " + args.config_path)
        print("args: " + str(args))

        return args, model.extract(args), pipeline.extract(args)
