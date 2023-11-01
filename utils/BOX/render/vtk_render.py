import os
import matplotlib.pyplot as pl
import numpy as np

import vtk
from vtk.util import numpy_support
from vtk import vtkWindowToImageFilter, vtkPNGWriter

__all__ = ["meshRenderer"]


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
        if os.path.isfile(path):
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
