import itk
import vtk
from vtk.util import numpy_support

from sklearn.preprocessing import MinMaxScaler

__all__ = ['vtkReader', 'vtkMesher', 'center_mesh']


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
    return meshdef
    smooth_mesh(mesh, iterations=15, relaxation_factor=0.01, feature_smoothing=False, boundary_smoothing=False):
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
