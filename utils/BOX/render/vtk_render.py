import os

import numpy as np
import vtk
from matplotlib import pyplot as plt
from vtk import vtkPNGWriter
from vtk.util.numpy_support import vtk_to_numpy

__all__ = ["meshRenderer", "npMesher", "enshure_binary"]

from vtkmodules.util import numpy_support


class npMesher:
    def __init__(self, method='sinc', easy_factor=0.26, iterations=24, relaxation_factor=0.01,
                 feature_smoothing=False, boundary_smoothing=True, center=True):
        # 可以考虑加颜色
        self.method = method
        self.easy_factor = easy_factor
        self.iterations = iterations
        self.relaxation_factor = relaxation_factor
        self.feature_smoothing = feature_smoothing
        self.boundary_smoothing = boundary_smoothing
        self.center = center

    def __call__(self, np_image):
        vtk_image = npToVTK(np_image)
        return vtk_to_3dmesh(vtk_image, method=self.method, easy_factor=self.easy_factor, iterations=self.iterations,
                             relaxation_factor=self.relaxation_factor, feature_smoothing=self.feature_smoothing,
                             boundary_smoothing=self.boundary_smoothing, center=self.center)


def npToVTK(np_array):
    '''
    接受二值化的np数组
    :param np_array:
    :return: vtkImageData
    '''

    enshure_binary(np_array)
    # np_array = np.transpose(np_array, (1, 2, 0))
    # print("np_array shape:", np_array.shape)
    # Create a vtkImageData object
    vtk_image = vtk.vtkImageData()

    # Set the dimensions of the vtkImageData object
    vtk_image.SetDimensions(np_array.shape[::-1])

    # Convert the numpy array to a vtkArray and set it as the scalars of the vtkImageData object
    vtk_array = numpy_support.numpy_to_vtk(num_array=np_array.ravel(), deep=True, array_type=vtk.VTK_FLOAT)
    vtk_image.GetPointData().SetScalars(vtk_array)

    return vtk_image


def enshure_binary(np_array):
    # Ensure the numpy array is binary
    unique_values = np.unique(np_array)
    # if np.any(np.abs(unique_values - 0) > 1e-9) and np.any(np.abs(unique_values - 1) > 1e-9):
    #     raise ValueError(
    #         "The numpy array must be binary, all values must be either 0 or 1, got {}.".format(unique_values))
    if not np.all(np.isclose(unique_values, [0., 1.], atol=1e-8)):
        raise ValueError(
            "The numpy array must be binary, all values must be either 0 or 1, got {}.".format(unique_values))
        # Ensure the numpy array is 3D
    if len(np_array.shape) != 3:
        raise ValueError("The numpy array must be 3D.")


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
    # if vtk_image.GetPointData().GetScalars() is None:
    #     raise ValueError("Input vtk_image has no scalar data.")

    # Check if vtk_image has scalar data
    if vtk_image.GetPointData().GetScalars() is None:
        raise ValueError("Input vtk_image has no scalar data.")

    # Get the scalar range
    min_val, max_val = vtk_image.GetScalarRange()

    contourFilter = vtk.vtkMarchingCubes()
    contourFilter.SetInputData(vtk_image)

    # Set the threshold value to the middle of the scalar range
    threshold = (min_val + max_val) / 2
    threshold = 0.01
    # print("auto threshold", threshold)
    contourFilter.SetValue(0, threshold)
    contourFilter.Update()

    # Check if contourFilter output has scalar data
    output_scalars = contourFilter.GetOutput().GetPointData().GetScalars()
    if output_scalars is None:
        raise ValueError("No contours were generated. Check the threshold value.")

    # contourFilter.Update()
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


class meshRenderer:
    def __init__(self, window_size=(1080, 1080), color=(1, 166 / 255, 0), camera_position_scale=(-4, 0, 0),
                 camera_focal_point=(0, 0, 0), opacity=1.0, save_path="./"):
        self.window_size = window_size
        self.color = color
        self.camera_position_scale = camera_position_scale
        self.camera_focal_point = camera_focal_point
        self.opacity = opacity
        self.save_path = save_path

    def __call__(self, mesh, color=None, opacity=None, see_pos=True):
        if color is None:
            color = self.color
        if opacity is None:
            opacity = self.opacity

        bounds = mesh.GetBounds()

        # Compute the size of the mesh along each axis
        mesh_size = (bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4])

        # Compute the camera position by multiplying the mesh size by the scale
        camera_position = tuple(a * b for a, b in zip(self.camera_position_scale, mesh_size))
        if see_pos:
            print("camera_position:{} * {} = {}".format(self.camera_position_scale, mesh_size, camera_position))

        return render_mesh_to_image(mesh, window_size=self.window_size, color=color,
                                    camera_position=camera_position, camera_focal_point=self.camera_focal_point,
                                    opacity=opacity)

    def np(self, mesh, color=None, opacity=None, see_pos=False):
        vtk_img = self(mesh, color=color, opacity=opacity, see_pos=see_pos)
        plt_img = vtkTonp(vtk_img)
        return plt_img

    def show(self, mesh, color=None, opacity=None, see_pos=False):
        vtk_img = self(mesh, color=None, opacity=None, see_pos=see_pos)
        display_image(vtk_img)

    def save(self, mesh, file_name, path=None, color=None, opacity=None, see_pos=False):
        vtk_img = self(mesh, color=None, opacity=None, see_pos=see_pos)
        if path is None:
            if self.save_path is None:
                raise ValueError("save path is None")
            else:
                path = self.save_path

        # 如果path包括文件名，则去掉文件名
        if os.path.isfile(path):
            path = os.path.dirname(path)
        if not os.path.exists(path):
            os.mkdir(path)
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


def save_vtk_to_obj(path, mesh_3d, file_name):
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
    # camera.SetViewAngle(45)  # 设置视角为45度

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


def vtkTonp(image):
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
    numpy_array = vtkTonp(image)

    # 设置matplotlib的显示窗口大小
    plt.figure(figsize=(10, 10))

    # 使用matplotlib显示图像
    plt.imshow(numpy_array, cmap='gray')
    plt.show()
    return numpy_array


def threshold_image(img, threshold=0.5):
    # 检查img信息
    # print("img array names:", img.GetPointData().GetArrayName(0))
    # 检查输入范围
    # print("img min, max:", img.GetScalarRange())
    # 获取点数据
    # point_data = img.GetPointData()
    thresh_filter = vtk.vtkThreshold()
    thresh_filter.SetInputData(img)
    thresh_filter.ThresholdByLower(threshold)
    # thresh_filter.SetInputArrayToProcess(0, 0, 0, 1,
    #                                      "ImageFile")  # 设置输入数组，0,0,0,1的意思是表示,0表示第一个输入，0表示第一个输出，0表示第一个输入的第一个数组，1表示第一个输入的第一个数组的第一个分量

    # 更新过滤器并获取输出
    thresh_filter.Update()
    img_thresholded = thresh_filter.GetOutput()

    # 检查图像是否包含数据
    if img_thresholded.GetNumberOfPoints() == 0:
        raise ValueError("The thresholded image contains no data.")

    return img_thresholded


def check_vtk_slice(vtk_image, slice_index=64):
    # def check_vtk_slice(vtk_image, slice_index):
    # Check if vtk_image is a vtkImageData object
    if not isinstance(vtk_image, vtk.vtkImageData):
        raise ValueError("Input must be a vtkImageData object.")

    # Get the extent of the vtk_image
    extent = vtk_image.GetExtent()

    # Check if slice_index is within the range of slices
    if slice_index < extent[4] or slice_index > extent[5]:
        raise ValueError(
            f"Slice index {slice_index} is out of range. It should be between {extent[4]} and {extent[5]}.")

    # Create a vtkMatrix4x4 object to indicate the direction of the slice
    reslice_axes = vtk.vtkMatrix4x4()
    reslice_axes.DeepCopy((1, 0, 0, 0,
                           0, 1, 0, 0,
                           0, 0, 1, 0,
                           0, 0, slice_index, 1))

    # Create a vtkImageReslice object to get a slice of the vtk_image
    reslice = vtk.vtkImageReslice()
    reslice.SetInputData(vtk_image)
    reslice.SetOutputDimensionality(2)
    reslice.SetResliceAxes(reslice_axes)
    reslice.Update()
    # Get the dimensions of the slice
    slice_dims = reslice.GetOutput().GetDimensions()

    # Convert the output of the vtkImageReslice object to a numpy array
    # slice_array = vtk_to_numpy(reslice.GetOutput().GetPointData().GetScalars())
    # Convert the output of the vtkImageReslice object to a numpy array
    slice_array = vtk_to_numpy(reslice.GetOutput().GetPointData().GetScalars())

    # Reshape the array to 2D
    slice_array_2d = slice_array.reshape(slice_dims[1], slice_dims[0])

    return slice_array_2d
