# -*- CODING: UTF-8 -*-
# @time 2023/10/7 16:02
# @Author tyqqj
# @File main.py
# @
# @Aim 

import argparse
import os

from mlflow.entities import Experiment
# from functools import partial
import utils.BOX as box

from monai.losses import DiceCELoss, DiceLoss
from monai.metrics import DiceMetric
from monai.transforms import Activations, AsDiscrete, Compose
from monai.utils.enums import MetricReduction
from monai.networks.nets import UNet
from monai import __version__
import mlflow
import mlflow.pytorch

import numpy as np
import torch
from torch.utils import data
from torch import nn

import numpy as np
import torch
from torch.utils import data
from torch import nn

# import os
# import matplotlib.pyplot as plt
# import pandas as pd
# import torchvision

# 设定运行处理器
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('using ', device)

# 定义参数解析器
parser = argparse.ArgumentParser(description="monai training arguments")
# parser.add_argument("--test", action="store_true", help="test mode")

# mlflow 基础参数
mlflow_parser = parser.add_argument_group('mlflow')
parser.add_argument("-n", "--exp_name", type=str, default=None, help="experiment name, ***must be set***")
parser.add_argument("-i", "--run_id", type=str, default=None,
                    help="run id, ***must be set when test or is_continue, only for continue training***")
parser.add_argument("-c", "--is_continue", action="store_true", help="continue training")
parser.add_argument("--train", action="store_true", help="train mode")
parser.add_argument("--test", action="store_true", help="test mode")
parser.add_argument("--new_run_name", type=str, default=None, help="new run name")
parser.add_argument("--log_dir", type=str, default="./runs", help="log dir")
# parser.add_argument("--artifact_dir", type=str, default="./artifacts", help="artifact dir")
parser.add_argument("--tag_id", type=str, default=None, help="tag id, ***commanded to set***")
parser.add_argument("--log_frq", type=int, default=10, help="log frequency")
parser.add_argument("--save_frq", type=int, default=10, help="save frequency, disabled in test and val")

parser.add_argument("--user_name", type=str, default="tyqqj", help="user name")

parser.add_argument("--vis_3d", action="store_true", help="visualize 3d images")
parser.add_argument("--vis_2d", action="store_true", help="visualize 2d images")
# 覆盖保存显示, 控制可视化是保留最新还是保留每个epoch
parser.add_argument("--vis_2d_cover", action="store_true", help="visualize 2d images")
parser.add_argument("--vis_3d_cover", action="store_true", help="visualize 3d images")

# 运行使用参数
run_parser = parser.add_argument_group('run')
parser.add_argument("--max_epochs", type=int, default=100, help="number of maximum epochs")
parser.add_argument("--batch_size", type=int, default=2, help="batch size")

# 解析出输入参数
args = parser.parse_args()


def main_worker(args):
    with run_box as run:

        for i in range(args.max_epochs):
            loader = []
            run_box.start_epoch(loader, 'train', i)
            for j in range(len(loader)):
                out = None
                target = None
                run_box.update_in_epoch(out, target)
                # 保存artifacts
                # mlflow.log_artifact("artifacts.txt")
                # 保存模型
            # if i % args.save_frq == 0:
            # model = UNet()
            # mlflow.pytorch.log_model(model, "models")
            # 保存模型参数
            model = None
            last_batch = None
            run_box.end_epoch(model, last_batch, i)
        mlflow.log_param("epochs", args.max_epochs)
        mlflow.log_param("batch_size", args.batch_size)


# mlflow.log_param("epochs", args.epochs)

if __name__ == "__main__":
    # 解析参数
    args = parser.parse_args()
    run_box = BOX.box(args)
    # mlflow 实验运行
    main_worker(args)
