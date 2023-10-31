# -*- CODING: UTF-8 -*-
# @time 2023/10/30 20:14
# @Author tyqqj
# @File render.py
# @
# @Aim 

# import argparse
# import os
# from functools import partial

# from monai.losses import DiceCELoss, DiceLoss
# from monai.metrics import DiceMetric
# from monai.transforms import Activations, AsDiscrete, Compose
# from monai.utils.enums import MetricReduction
# from monai import __version__

import numpy as np
# import torch
# from torch.utils import data
# from torch import nn
#
# import numpy as np
# import torch
# from torch.utils import data
# from torch import nn

# import os
import matplotlib.pyplot as plt
# import pandas as pd
# import torchvision

import vtk
from vtk.util import numpy_support
import itk
# from mayavi import mlab

# # 假设 `voxels` 是你的体素数据
# mlab.contour3d(voxels)
# mlab.show()

from sklearn.preprocessing import MinMaxScaler

camera_pos = [0, -1, 0]


def checker(vtk_image, n=64):
    # 转化成numpy
    arr = numpy_support.vtk_to_numpy(vtk_image.GetPointData().GetScalars())
    # 抽取其中的一个层
    # 获取 VTK 图像的维度
    dims = vtk_image.GetDimensions()

    # 将一维数组重塑为原始的三维形状
    arr_reshaped = arr.reshape(dims[2], dims[1], dims[0])

    # 选择一个层进行可视化
    slice_index = dims[2] // 2  # 选择中间层
    slice_arr = arr_reshaped[n, :, :]

    # 使用 matplotlib 进行可视化
    plt.imshow(slice_arr, cmap='gray', interpolation='nearest')
    plt.show()


def read_mha_to_vtk(path):
    print("loading", path)
    # 读取 .mha 文件
    image = itk.imread(path)

    # 将 ITK image 转换为 VTK image
    vtk_image = vtk.vtkImageData()
    vtk_image.SetDimensions(image.GetLargestPossibleRegion().GetSize())
    vtk_image.SetSpacing(image.GetSpacing())

    arr = itk.GetArrayViewFromImage(image)

    # 获取原始形状并将其变形为一维的
    original_shape = arr.shape
    arr_flattened = arr.flatten().reshape(-1, 1)

    # 缩放数据到0-255
    scaler = MinMaxScaler(feature_range=(0, 255))
    arr_scaled = scaler.fit_transform(arr_flattened)

    # 将数据重新变形为原来的形状
    arr_reshaped = arr_scaled.reshape(original_shape)

    vtk_image.GetPointData().SetScalars(
        numpy_support.numpy_to_vtk(arr_reshaped.ravel(), deep=True, array_type=vtk.VTK_FLOAT))

    return vtk_image


def render_vtk_image(vtk_image, camera_position=(0, -10, 0), focal_point=(0, 0, 0), view_up=(0, 0, 1)):
    print("rendering")
    # 创建渲染器和渲染窗口
    renderer = vtk.vtkRenderer()
    renderWindow = vtk.vtkRenderWindow()
    renderWindow.AddRenderer(renderer)

    # 设置相机的位置和朝向
    camera = renderer.GetActiveCamera()
    camera.SetPosition(camera_position)
    camera.SetFocalPoint(focal_point)
    camera.SetViewUp(view_up)

    # 创建体渲染映射器并设置其输入为 VTK image
    volumeMapper = vtk.vtkGPUVolumeRayCastMapper()
    volumeMapper.SetInputData(vtk_image)

    # 创建体渲染属性和体渲染
    volumeColor = vtk.vtkColorTransferFunction()
    volumeColor.AddRGBPoint(0, 0.0, 0.0, 0.0)  # black
    volumeColor.AddRGBPoint(255, 1.0, 0.5, 0.0)  # orange

    volumeScalarOpacity = vtk.vtkPiecewiseFunction()
    volumeScalarOpacity.AddPoint(0, 0.0)
    volumeScalarOpacity.AddPoint(255, 1.0)

    volumeGradientOpacity = vtk.vtkPiecewiseFunction()

    volumeProperty = vtk.vtkVolumeProperty()
    volumeProperty.SetColor(volumeColor)
    volumeProperty.SetScalarOpacity(volumeScalarOpacity)
    volumeProperty.SetGradientOpacity(volumeGradientOpacity)
    volumeProperty.SetInterpolationTypeToLinear()  # 线性插值

    volume = vtk.vtkVolume()
    volume.SetMapper(volumeMapper)
    volume.SetProperty(volumeProperty)

    # mlab.contour3d(voume)
    # mlab.show()

    # plt.imshow(volume[50, :, :])  # 显示第 50 层
    # # 显示第 50 层
    # plt.show()

    # 添加体渲染到渲染器
    renderer.AddVolume(volume)

    # 开始渲染
    print("rendering")
    renderWindow.Render()

    # 获取渲染的图像
    windowToImageFilter = vtk.vtkWindowToImageFilter()
    windowToImageFilter.SetInput(renderWindow)
    windowToImageFilter.Update()

    return windowToImageFilter.GetOutput()


def vtk_image_to_numpy_3d(vtk_image):
    # 获取vtkImageData的宽度和高度
    dims = vtk_image.GetDimensions()

    # 将vtkImageData转换为NumPy数组
    vtk_array = vtk_image.GetPointData().GetScalars()
    numpy_array = numpy_support.vtk_to_numpy(vtk_array)

    # 将NumPy数组的形状调整为图像的形状
    numpy_array = numpy_array.reshape(dims[2], dims[1], dims[0])

    return numpy_array


def show_image_with_plt(vtk_image):
    # 将vtkImageData转换为NumPy数组
    numpy_array = vtk_to_numpy(vtk_image)

    # 如果图像是RGB图像，则需要将颜色通道放在最后
    if numpy_array.shape[2] == 3:
        numpy_array = np.flip(numpy_array, axis=2)

    # 使用matplotlib来显示图像
    plt.imshow(numpy_array)
    plt.show()

def vtk_to_numpy(vtk_image):
    # 获取vtkImageData中的点数据
    point_data = vtk_image.GetPointData()

    # 获取vtkImageData中的标量数据
    scalar_data = point_data.GetScalars()

    # 将vtkArray转换为NumPy数组
    numpy_array = numpy_support.vtk_to_numpy(scalar_data)

    # 获取图像的维度
    dims = vtk_image.GetDimensions()

    # 重塑NumPy数组以匹配图像的维度
    numpy_array = numpy_array.reshape(dims[1], dims[0], -1)

    return numpy_array


def show_vtk_image(vtk_image):
    # 创建一个 ImageViewer
    viewer = vtk.vtkImageViewer2()
    viewer.SetInputData(vtk_image)

    # 创建一个渲染窗口交互器
    iren = vtk.vtkRenderWindowInteractor()
    viewer.SetupInteractor(iren)

    # 设置 viewer 的参数
    viewer.SetColorWindow(255)
    viewer.SetColorLevel(127.5)
    viewer.Render()

    # 开始交互
    iren.Start()


# def show_image_with_plt(vtk_image, n=64):
#     # 将vtkImageData转换为NumPy数组
#     numpy_array = vtk_image_to_numpy(vtk_image)
#
#     # 如果图像是RGB图像，则需要将颜色通道放在最后
#     if numpy_array.shape[2] == 3:
#         numpy_array = np.flip(numpy_array, axis=2)
#
#     # 选择一个层进行可视化
#     # if numpy_array.shape[0] <= n:
#     # img = numpy_array[n, :, :]
#
#     # 使用matplotlib来显示图像
#     plt.imshow(img)
#     plt.show()


if __name__ == '__main__':
    # 读取 .mha 文件
    vtk_image = read_mha_to_vtk("D:\\Data\\test\\249_label.mha")

    # 渲染并保存图像
    windowed_image = render_vtk_image(vtk_image, [0, -1, 0], [0, 1, 0], [0, 0, 1])
    checker(vtk_image)
    # 显示
    # show_vtk_image(windowed_image)
    show_image_with_plt(windowed_image)
    # writer = vtk.vtkPNGWriter()
    # writer.SetFileName("C:\\Users\\tyqqj\\Desktop\\test.png")
    # writer.SetInputData(windowed_image)
    # writer.Write()
