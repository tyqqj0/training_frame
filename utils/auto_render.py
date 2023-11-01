# -*- CODING: UTF-8 -*-
# @time 2023/10/31 19:49
# @Author tyqqj
# @File auto_render.py
# @
# @Aim 

import numpy as np
import torch
from torch.utils import data
from torch import nn
import mlflow

# import os
# import matplotlib.pyplot as plt
# import pandas as pd
# import torchvision


from .BOX.render import render


def generate_path(path, key=None):
    '''
    生成mha路径列表，若key不为空，则筛选名称包含key的路径
    :param path:
    :param key:
    :return: path_list
    '''
    path_list = []
    # 遍历文件夹, walk返回三元组, root是当前目录, dirs是当前目录下的文件夹, files是当前目录下的文件
    for root, dirs, files in os.walk(path):
        for file in files:
            if key is None:
                path_list.append(os.path.join(root, file))
            else:
                if key in file:
                    path_list.append(os.path.join(root, file))

        for dir in dirs:
            path_list.extend(generate_path(os.path.join(root, dir), key))

    return path_list


def get_artifact(run_id=None):
    '''
    从mlflow中获取artifact
    :param run_id:
    :param key:
    :return:
    '''
    if run_id is None:
        # 尝试获取最后一个运行的id
        runs = mlflow.search_runs(order_by=["attribute.start_time DESC"], max_results=1)
        if runs.empty:
            raise ValueError(
                "no run exists, please check if the experiment name is correct or muti-user conflict on",
                mlflow.get_tracking_uri())  # 遇到了开多个不同源的mlflow server的问题，会在不同地方创建同名的实验
        # print(runs)
        last_run_id = runs.loc[0, "run_id"]
        # print(runs)
        print("using last run id: {}, name: {}".format(last_run_id, runs.loc[0, "tags.mlflow.runName"]))
        if last_run_id is None:
            # raise ValueError("Cannot set is_continue to True when name is None")
            raise ValueError("Cannot find last run id")
        run_id = last_run_id
        # self.args.run_id = last_run_id
        # 默认使用最后一个运行的id
        run_id = last_run_id
    else:
        # 检查run_id是否存在
        run = mlflow.get_run(run_id=str(args.run_id))
        if run is None:
            raise ValueError("run_id {} not exists".format(args.run_id))
        # print(runs)
        print("using run id: {}, name: {}".format(args.run_id, run.data.tags["mlflow.runName"]))
        # 默认使用最后一个运行的id
        run_id = run_id

    # 获取artifact的路径
    mlflow.start_run(run_id=run_id)
    artifact_uri = mlflow.get_artifact_uri()
    mlflow.end_run()
    return artifact_uri


if __name__ == '__main__':
    # 尝试从mlflow中获取路径
    run_id = None
    path = None
    if path is None:
        path = get_artifact(run_id)
    print(path)
    # 生成路径列表
    path_list = generate_path(path, key='mha')
