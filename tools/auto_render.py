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

import os
# import matplotlib.pyplot as plt
# import pandas as pd
# import torchvision
from urllib.parse import urlparse
import json

os.environ['MLFLOW_TRACKING_URI'] = '../mlruns'

import utils.BOX as box

import utils.BOX.render as render


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
        # print(root, dirs, files)
        for file in files:
            if key is None:
                path_list.append(os.path.join(root, file))
            else:
                if key in file and 'mha' in file:
                    path_list.append(os.path.join(root, file))

        for dir in dirs:
            path_list.extend(generate_path(os.path.join(root, dir), key))

    return path_list


def generate_path_from_data_json(path, set=['train', 'val'], key=['label']):
    '''
    从data.json中读取路径
    :param path:
    :param key:
    :return:
    '''
    path_list = []
    data_file = json.load(open(path, 'r'))
    for set_name in set:
        for file in data_file[set_name]:
            # print("file: ", file)
            for f in file:
                # print("f: ", f)
                if key is None:
                    path_list.append(file[f])
                elif any(k in f for k in key):
                    path_list.append(file[f])
    return path_list


def get_artifact(exp_name, run_id=None):
    '''
    从mlflow中获取artifact
    :param run_id:
    :param key:
    :return:
    '''
    mlflow.set_experiment(exp_name)
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
        run = mlflow.get_run(run_id=str(run_id))
        if run is None:
            raise ValueError("run_id {} not exists".format(run_id))
        # print(runs)
        print("using run id: {}, name: {}".format(run_id, run.data.tags["mlflow.runName"]))
        # 默认使用最后一个运行的id
        run_id = run_id

    # 获取artifact的路径
    mlflow.start_run(run_id=run_id)
    artifact_uri = mlflow.get_artifact_uri()
    run_name = mlflow.get_run(run_id).data.tags['mlflow.runName']
    mlflow.end_run()
    tracking_uri = '../mlartifacts'
    parsed_tracking_uri = urlparse(tracking_uri)
    parsed_artifact_uri = urlparse(artifact_uri)

    # 将 MLFLOW_TRACKING_URI 和 artifact_uri 拼接起来，以获取 artifact 的绝对路径
    local_path = parsed_tracking_uri.path.lstrip('/') + '/' + parsed_artifact_uri.path.lstrip('/')
    local_path = local_path.replace('/', '\\')
    return local_path, run_name


def batch_render(path_list, loader, mesher, renderer, save_path=None):
    '''
    批量渲染
    :param path_list:
    :param loader:
    :param mesher:
    :param renderer:
    :return:
    '''
    for path in path_list:
        print('rendering ', os.path.basename(path))
        # 读取数据
        vtk_image, file_name = loader(path)
        # 生成网格
        mesh = mesher(vtk_image)
        # 渲染
        renderer.save(mesh, file_name, path=save_path)


if __name__ == '__main__':
    run_name = 'vessel'
    # 尝试从mlflow中获取路径
    # run_id = 'b0fa3574f13244ba8ad9fc433226aef9'
    # path = None
    # if path is None:
    #     path, run_name = get_artifact('train', run_id)
    # print(path)
    # # 生成路径列表
    # path_list = generate_path(path + "/vis_3d", key='output')

    # 从data.json中获取路径
    path = '../data/'
    json_path = path + run_name + '.json'
    path_list = generate_path_from_data_json(json_path, set=['train', 'val'], key=['label'])
    print(path_list)
    # 定义渲染工具
    loader = render.mhaReader()
    mesher = render.npMesher()
    renderer = render.meshRenderer(opacity=0.85, save_path=path + '/' + run_name + '_3d')
    batch_render(path_list, loader, mesher, renderer)
