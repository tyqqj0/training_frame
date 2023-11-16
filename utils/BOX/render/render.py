# -*- CODING: UTF-8 -*-
# @time 2023/11/15 20:37
# @Author tyqqj
# @File render.py
# @
# @Aim 
import numpy as np

from .vtk_render import meshRenderer, npMesher, enshure_binary


def rotate_3d_matrix(matrix, degrees):
    x, y, z = degrees

    # 旋转次数是角度除以90
    # x = x // 90
    # y = y // 90
    # z = z // 90

    # 绕 x 轴旋转
    if x:
        matrix = np.rot90(matrix, k=x, axes=(1, 2))

    # 绕 y 轴旋转
    if y:
        matrix = np.rot90(matrix, k=y, axes=(0, 2))

    # 绕 z 轴旋转
    if z:
        matrix = np.rot90(matrix, k=z, axes=(0, 1))

    return matrix


# import vtk_render

def renderNp(img, threshold=None, camera_position_scale=(0, 0, 2)):
    '''
    输入x,y,z的二值化矩阵，输出渲染后的图片
    默认从z轴正方向两倍距离渲染
    :param img: 二值化矩阵
    :param threshold: 阈值
    :param camera_position_scale: 相机位置
    :return: 渲染后的图片
    '''
    if threshold is not None:
        img = img > threshold
    # 检查是否为二值
    try:
        enshure_binary(img)
    except:
        raise ValueError("The image is not binary, please threshold it first.")
    # 转换为网格
    mesher = npMesher()
    mesh = mesher(img)

    # 检查网格是否包含数据
    if mesh.GetNumberOfPoints() == 0:
        raise ValueError("The mesh contains no data.")

    # 生成渲染器
    renderer = meshRenderer(camera_position_scale=camera_position_scale)
    np_img = renderer.np(mesh, see_pos=False)

    return np_img
