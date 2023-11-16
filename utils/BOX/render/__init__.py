# -*- CODING: UTF-8 -*-
# @time 2023/10/30 20:14
# @Author tyqqj
# @File render.py
# @
# @Aim


from .mha_loader import *
from .render import *
from .vtk_render import *

# import argparse
# import os
# from functools imp ort partial

# from monai.losses import DiceCELoss, DiceLoss
# from monai.metrics import DiceMetric
# from monai.transforms import Activations, AsDiscrete, Compose
# from monai.utils.enums import MetricReduction
# from monai import __version__


# import torch
# from torch.utils import data
# from torch import nn
#
# import numpy as np
# import torch
# from torch.utils import data
# from torch import nn


# import pandas as pd
# import torchvision


# from mayavi import mlab

# # 假设 `voxels` 是你的体素数据
# mlab.contour3d(voxels)
# mlab.show()


# camera_pos = [0, -1, 0]


# def checker(vtk_image, n=64):
#     # 转化成numpy
#     arr = numpy_support.vtk_to_numpy(vtk_image.GetPointData().GetScalars())
#     # 抽取其中的一个层
#     # 获取 VTK 图像的维度
#     dims = vtk_image.GetDimensions()
#
#     # 将一维数组重塑为原始的三维形状
#     arr_reshaped = arr.reshape(dims[2], dims[1], dims[0])
#
#     # 选择一个层进行可视化
#     slice_index = dims[2] // 2  # 选择中间层
#     slice_arr = arr_reshaped[n, :, :]
#
#     # 使用 matplotlib 进行可视化
#     plt.imshow(slice_arr, cmap='gray', interpolation='nearest')
#     plt.show()


# def read_mha_to_vtk(


if 0:
    # 读取 .mha 文件
    path = "D:\\Data\\test\\249_label.mha"
    vtkrdr = vtkReader()
    Mesher = vtkMesher()
    Renderer = meshRenderer(opacity=0.85)
    print("loading", path)
    vtk_image, file_name = vtkrdr(path)
    print("meshing")
    mesh_3d = Mesher(vtk_image)
    print("Rendering")
    vtk_img = Renderer.np(mesh_3d)
    Renderer.save(mesh_3d, file_name)
    plt.imshow(vtk_img)
    plt.show()
