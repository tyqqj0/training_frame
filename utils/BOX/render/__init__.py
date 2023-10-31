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

import os
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
from vtk import vtkWindowToImageFilter, vtkPNGWriter


# camera_pos = [0, -1, 0]


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


class vtkReader:
    def __init__(self):
        pass

    def __call__(self, path):
        # print("loading", path)
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

        # 获取文件名称, 去掉后缀
        file_name = os.path.splitext(os.path.basename(path))[0]
        return vtk_image, file_name


# def read_mha_to_vtk(


class vtkMesher:
    def __init__(self, method='sinc', easy_factor=0.26, iterations=24, relaxation_factor=0.01,
                 feature_smoothing=False, boundary_smoothing=True, center=True):
        self.method = method
        self.easy_factor = easy_factor
        self.iterations = iterations
        self.relaxation_factor = relaxation_factor
        self.feature_smoothing = feature_smoothing
        self.boundary_smoothing = boundary_smoothing
        self.center = center

    def __call__(self, vtk_image):
        return vtk_to_3dmesh(vtk_image, method=self.method, easy_factor=self.easy_factor, iterations=self.iterations,
                             relaxation_factor=self.relaxation_factor, feature_smoothing=self.feature_smoothing,
                             boundary_smoothing=self.boundary_smoothing, center=self.center)


class meshRenderer:
    def __init__(self, window_size=(1080, 1080), color=(1, 166 / 255, 0), camera_position=(0, 0, 200),
                 camera_focal_point=(0, 0, 0), opacity=1.0, save_path="./"):
        self.window_size = window_size
        self.color = color
        self.camera_position = camera_position
        self.camera_focal_point = camera_focal_point
        self.opacity = opacity
        self.save_path = save_path

    def __call__(self, mesh, color=None, opacity=None):
        if color is None:
            color = self.color
        if opacity is None:
            opacity = self.opacity
        return render_mesh_to_image(mesh, window_size=self.window_size, color=color,
                                    camera_position=self.camera_position, camera_focal_point=self.camera_focal_point,
                                    opacity=opacity)

    def np(self, mesh, color=None, opacity=None):
        vtk_img = self(mesh, color=color, opacity=opacity)
        plt_img = vtk_image_to_numpy(vtk_img)
        return plt_img

    def show(self, mesh, color=None, opacity=None):
        vtk_img = self(mesh, color=None, opacity=None)
        display_image(vtk_img)

    def save(self, mesh, file_name, path=None, color=None, opacity=None):
        vtk_img = self(mesh, color=None, opacity=None)
        if path is None:
            if self.save_path is None:
                raise ValueError("save path is None")
            else:
                path = self.save_path

        # 如果path包括文件名，则去掉文件名
        path = os.path.dirname(path)
        # 保存到png
        # 设置保存的文件路径
        if 'png' not in file_name:
            file_name += '.png'
        file_path = os.path.join(path, file_name)

        # 创建一个窗口到图像的过滤器
        # window_to_image_filter = vtkWindowToImageFilter()
        # window_to_image_filter.SetInput(vtk_img)
        # window_to_image_filter.Update()

        # 创建一个 PNG 写入器
        writer = vtkPNGWriter()
        writer.SetFileName(file_path)
        writer.SetInputData(vtk_img)
        writer.Write()

        print(f"Image saved to {file_path}")
        return vtk_img


def smooth_mesh(mesh, iterations=15, relaxation_factor=0.01, feature_smoothing=False, boundary_smoothing=False):
    print("smoothing")
    smoothFilter = vtk.vtkSmoothPolyDataFilter()
    smoothFilter.SetInputData(mesh)
    smoothFilter.SetNumberOfIterations(iterations)  # 迭代的次数
    smoothFilter.SetRelaxationFactor(relaxation_factor)  # 点移动的步长
    smoothFilter.SetFeatureEdgeSmoothing(feature_smoothing)  # 是否保留特征边缘
    smoothFilter.SetBoundarySmoothing(boundary_smoothing)  # 是否平滑边界点
    smoothFilter.Update()

    return smoothFilter.GetOutput()


def sinc_smooth_mesh(mesh, iterations=20, pass_band=0.1, boundary_smoothing=False, feature_smoothing=False):
    sincFilter = vtk.vtkWindowedSincPolyDataFilter()
    sincFilter.SetInputData(mesh)
    sincFilter.SetNumberOfIterations(iterations)
    sincFilter.SetPassBand(pass_band)
    sincFilter.SetBoundarySmoothing(boundary_smoothing)
    sincFilter.SetFeatureEdgeSmoothing(feature_smoothing)
    sincFilter.NonManifoldSmoothingOff()
    sincFilter.NormalizeCoordinatesOn()
    sincFilter.Update()

    return sincFilter.GetOutput()


def vtk_to_3dmesh(vtk_image, method='sinc', easy_factor=0.26, iterations=24, relaxation_factor=0.01,
                  feature_smoothing=False, boundary_smoothing=True, center=True):
    # print("converting")
    contourFilter = vtk.vtkMarchingCubes()
    contourFilter.SetInputData(vtk_image)
    contourFilter.SetValue(0, 127.5)  # 设置表面提取的阈值
    contourFilter.Update()
    if method == 'default':
        smoothed_mesh = smooth_mesh(contourFilter.GetOutput(), iterations, relaxation_factor, feature_smoothing,
                                    boundary_smoothing)
    elif method == 'sinc':
        smoothed_mesh = sinc_smooth_mesh(contourFilter.GetOutput(), iterations, easy_factor, boundary_smoothing,
                                         feature_smoothing)
        # 这里的0.1-1.
    if center:
        smoothed_mesh = center_mesh(smoothed_mesh)
    return smoothed_mesh


def center_mesh(mesh):
    # 计算模型的几何中心
    centerOfMassFilter = vtk.vtkCenterOfMass()
    centerOfMassFilter.SetInputData(mesh)
    centerOfMassFilter.SetUseScalarsAsWeights(False)
    centerOfMassFilter.Update()
    centerOfMass = centerOfMassFilter.GetCenter()

    # 移动模型，使其原点在几何中心
    transform = vtk.vtkTransform()
    transform.Translate(-centerOfMass[0], -centerOfMass[1], -centerOfMass[2])
    transformFilter = vtk.vtkTransformPolyDataFilter()
    transformFilter.SetInputData(mesh)
    transformFilter.SetTransform(transform)
    transformFilter.Update()
    mesh = transformFilter.GetOutput()
    return mesh


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


def save_vtk_to_obj(path, file_name):
    # 如果path包含文件名，则去掉文件名
    path = os.path.dirname(path)
    # 如果path不存在，则创建
    if not os.path.exists(path):
        os.makedirs(path)
    # 保存到obj
    writer = vtk.vtkOBJWriter()
    writer.SetFileName(os.path.join(path, file_name + ".obj"))
    writer.SetInputData(mesh_3d)
    writer.Write()
    print("saved to", os.path.join(path, file_name + ".obj"))


def render_mesh_to_image(mesh, window_size=(800, 800), color=(1, 166 / 255, 0), camera_position=(0, 0, 200),
                         camera_focal_point=(0, 0, 0), opacity=1.0):
    # 为模型创建一个 mapper
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(mesh)
    mapper.ScalarVisibilityOff()  # 忽略模型的颜色信息

    # 创建一个 actor
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    # actor.GetProperty().SetColorModeToDefault()  # 设置颜色模式为默认
    actor.GetProperty().SetColor(*color)  # 设置颜色, *color表示将color中的元素作为参数传入
    # 设置透明度,默认为不透明
    actor.GetProperty().SetOpacity(opacity)

    # 创建渲染器和渲染窗口
    renderer = vtk.vtkRenderer()
    renderer.AddActor(actor)
    renderer.SetBackground(0, 0, 0)  # 设置背景色为黑色

    # 设置相机的位置和焦点
    camera = renderer.GetActiveCamera()
    camera.SetPosition(*camera_position)
    camera.SetFocalPoint(*camera_focal_point)
    # camera = renderer.GetActiveCamera()
    camera.SetViewAngle(45)  # 设置视角为45度

    renderWindow = vtk.vtkRenderWindow()
    renderWindow.AddRenderer(renderer)
    renderWindow.SetSize(*window_size)  # 设置窗口大小
    renderWindow.SetOffScreenRendering(1)
    renderWindow.Render()

    # 创建一个窗口到图像的过滤器
    windowToImageFilter = vtk.vtkWindowToImageFilter()
    windowToImageFilter.SetInput(renderWindow)
    windowToImageFilter.Update()

    return windowToImageFilter.GetOutput()


def vtk_image_to_numpy(image):
    # 获取图像的大小
    # 获取图像的大小
    width, height, _ = image.GetDimensions()
    # 获取图像的所有数据
    scalars = image.GetPointData().GetScalars()
    # 将vtk数据转换为numpy数组
    numpy_array = vtk.util.numpy_support.vtk_to_numpy(scalars)
    # 将一维数组转换为三维数组
    numpy_array = numpy_array.reshape(height, width, -1)
    return numpy_array


def display_image(image):
    # 将vtkImageData转换为numpy数组
    numpy_array = vtk_image_to_numpy(image)

    # 设置matplotlib的显示窗口大小
    plt.figure(figsize=(10, 10))

    # 使用matplotlib显示图像
    plt.imshow(numpy_array, cmap='gray')
    plt.show()
    return numpy_array


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
