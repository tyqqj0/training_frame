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
parser.add_argument("--save_frq", type=int, default=10, help="save frequency")

parser.add_argument("--user_name", type=str, default="tyqqj", help="user name")

# 运行使用参数
parser.add_argument("--max_epochs", type=int, default=100, help="number of maximum epochs")
parser.add_argument("--batch_size", type=int, default=2, help="batch size")

# 解析出输入参数
args = parser.parse_args()


def main_worker(args):
    # 出示服务器
    print("mlflow server: ", mlflow.get_tracking_uri())
    # 模拟参数
    run_name = None
    if args.new_run_name != None:
        run_name = args.exp_name + '-' + args.new_run_name
    with mlflow.start_run(run_id=args.run_id, run_name=run_name) as run:  # as的作用是将run的值赋给run_id
        mlflow.set_tag("mlflow.user", args.user_name)
        mlflow.set_tag("mlflow.note.content", "test" if args.test else "train")
        mlflow.set_tag("mlflow.note.run_id", args.run_id if args.run_id != None else run.info.run_id)
        # mlflow 参数
        mlflow.log_param("monai_version", __version__)
        for k, v in vars(args).items():
            mlflow.log_param(k, v)

        for i in range(args.max_epochs):
            if i % args.log_frq == 0:
                mlflow.log_metric("loss", i * 0.1, step=i)
                mlflow.log_metric("acc", i * 0.2, step=i)
                mlflow.log_metric("auc", i * 0.3, step=i)
                mlflow.log_metric("f1", i * 0.4, step=i)
                mlflow.log_metric("recall", i * 0.5, step=i)
                mlflow.log_metric("precision", i * 0.6, step=i)
                mlflow.log_metric("specificity", i * 0.7, step=i)
                mlflow.log_metric("sensitivity", i * 0.8, step=i)
                mlflow.log_metric("dice", i * 0.9, step=i)
                mlflow.log_metric("iou", i * 1.0, step=i)
                # 保存artifacts
                # mlflow.log_artifact("artifacts.txt")
                # 保存模型
            # if i % args.save_frq == 0:
            # model = UNet()
            # mlflow.pytorch.log_model(model, "models")
        # 保存模型参数
        mlflow.log_param("epochs", args.max_epochs)
        mlflow.log_param("batch_size", args.batch_size)

    # 记录最后运行run_id
    mlflow.log_param("last_run_id", run.info.run_id)
    # 出示服务器
    print("run {} finished".format(run.info.run_name))
    print("mlflow server: ", mlflow.get_tracking_uri())


# mlflow.log_param("epochs", args.epochs)

if __name__ == "__main__":
    # 解析参数
    args = parser.parse_args()

    # 实验模式检查
    if args.train and args.test:
        raise ValueError("Cannot set both train and test to True")

    if args.is_continue and not args.train:
        raise ValueError("Cannot set is_continue to True when train is False")

    # mlflow 实验设定
    mlflow.set_tracking_uri("http://localhost:5000")
    experiment = mlflow.get_experiment_by_name(args.exp_name)
    if experiment is None:
        print("create experiment: ", args.exp_name)
        mlflow.create_experiment(name=args.exp_name, tags={"mlflow.user": args.user_name, "type": "run_test"})
    else:
        print("use experiment: ", args.exp_name)
        # print(experiment)
        print("experiment id: ", experiment.experiment_id)
        print("artifact location: ", experiment.artifact_location)

    mlflow.set_experiment(args.exp_name)
    # 检查实验的状况
    # experiment = mlflow.get_experiment_by_name(args.exp_name)
    # print(experiment)
    # mlflow 运行设置参数
    # 默认继续的运行
    if args.is_continue and args.run_id is None:
        runs = mlflow.search_runs(order_by=["attribute.start_time DESC"], max_results=1)
        if runs.empty:
            raise ValueError("no run exists, please check if the experiment name is correct or muti-user conflict on",
                             mlflow.get_tracking_uri())  # 遇到了开多个不同源的mlflow server的问题，会在不同地方创建同名的实验
        # print(runs)
        last_run_id = runs.loc[0, "run_id"]
        # print(runs)
        print("using last run id: {}, name: {}".format(last_run_id, runs.loc[0, "tags.mlflow.runName"]))
        if last_run_id is None:
            # raise ValueError("Cannot set is_continue to True when name is None")
            raise ValueError("Cannot find last run id")
        args.run_id = last_run_id

    if args.test and args.run_id is None:
        raise ValueError("Cannot set test to True when name is None")

    # mlflow 实验运行
    main_worker(args)
