# -*- CODING: UTF-8 -*-
# @time 2023/11/15 20:25
# @Author tyqqj
# @File generate_misguid_data.py
# @
# @Aim 

import matplotlib.pyplot as plt
import numpy as np
from monai.losses import DiceLoss
from scipy.ndimage import gaussian_filter, binary_erosion, binary_dilation
from torch import tensor

import utils.BOX.render as render
import utils.data_loader.load_one_image as load_one_image


# from functools import partial


# from monai.metrics import DiceMetric
# from monai.transforms import Activations, AsDiscrete, Compose
# from monai.utils.enums import MetricReduction
# from monai import __version__
# import numpy as np
# import torch
# from torch.utils import data
# from torch import nn
#
# import numpy as np
# import torch
# from torch.utils import data
# from torch import nn
# import os
# import matplotlib.pyplot as plt
# import pandas as pd
# import torchvision
# 设定运行处理器
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print('using ', device)


class Random3DMask:
    def __init__(self, image_shape, blur_size=15, threshold=0.5, fixed_seed=None, enable_random=True):
        """
        创建一个随机掩码生成器。

        参数:
        image_shape: 图像的形状
        blur_size: 高斯模糊的核大小，基本和实际像素大小相同
        threshold: 生成掩码的阈值
        """
        self.image_shape = image_shape
        self.blur_size = blur_size
        self.threshold = threshold
        self.enable_random = enable_random
        if enable_random:
            self.fixed_seed = np.random.randint(0, 1000)
        else:
            self.fixed_seed = fixed_seed

    def generate(self):
        """
        生成一个随机掩码。

        返回:
        输出的掩码
        """
        # 随机定义数据
        if self.fixed_seed is not None:
            np.random.seed(self.fixed_seed)
        # 创建一个和输入图像同样大小的随机数组
        random_array = np.random.uniform(size=self.image_shape)

        # 使用高斯模糊使随机数组更平滑
        blurred_array = gaussian_filter(random_array, self.blur_size)

        # 使用一个阈值来生成掩码
        random_mask = blurred_array > self.threshold

        return random_mask

    def __call__(self):
        return self.generate()

    def show(self):
        plt.imshow(self.generate()[:, :, self.image_shape[2] // 2])
        plt.show()


class RandomErosionDilation3D:
    def __init__(self, mask_generator, structure=None):
        """
        创建一个随机腐蚀和膨胀操作器。

        参数:
        mask_generator: 用于生成随机掩码的生成器
        structure: 用于腐蚀和膨胀操作的结构元素。如果为None，则使用一个全为1的3x3x3的立方体。
        """
        self.mask_generator = mask_generator
        if structure is None:
            self.structure = np.ones((3, 3, 3))
        else:
            self.structure = structure

    def apply(self, image):
        """
        对输入的3D图像进行随机的腐蚀和膨胀操作。

        参数:
        image: 输入的3D图像

        返回:
        输出的3D图像
        """
        # 使用mask_generator生成随机掩码
        shape = image.shape
        random_mask = self.mask_generator.generate()
        print("image shape:", image.shape)
        print("random_mask shape:", random_mask.shape)
        print("structure shape:", self.structure.shape)

        print("masked image shape:", image[random_mask].shape)

        # 初始化输出的图像
        output_image = np.zeros_like(image)

        # # 对于掩码中为True的部分，进行腐蚀操作
        # output_image[random_mask] = binary_erosion(image[random_mask], self.structure)
        #
        # # 对于掩码中为False的部分，进行膨胀操作
        # output_image[~random_mask] = binary_dilation(image[~random_mask], self.structure)

        # 对整个图像进行腐蚀操作，然后再使用random_mask选择结果中的一部分
        eroded_image = binary_erosion(image, self.structure)
        output_image[random_mask] = eroded_image[random_mask]

        # 对整个图像进行膨胀操作，然后再使用~random_mask选择结果中的一部分
        dilated_image = binary_dilation(image, self.structure)
        output_image[~random_mask] = dilated_image[~random_mask]

        print("output_image shape: ", output_image.shape)
        return output_image.reshape(shape)

    def __call__(self, image):
        return self.apply(image)


def simple_render(img):
    img = render.rotate_3d_matrix(img, (0, 0, 0))
    # print(img.shape)
    np_img = render.renderNp(img, camera_position_scale=(-4, 0, 0))
    plt.imshow(np_img)
    plt.show()


def simple_see_slice(img):
    plt.figure()
    plt.imshow(img[:, :, img.shape[2] // 2])
    plt.show()


# 实现np输入的dice loss
class dice(DiceLoss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, input, target):
        input = tensor(input)
        target = tensor(target)
        return super().forward(input, target)



if __name__ == "__main__":
    # 定义参数解析器
    # parser = argparse.ArgumentParser(description="monai training arguments")
    # 解析出输入参数
    # args = parser.parse_args()
    data = load_one_image.one_label(threshold=True)
    # Random3DMask(data.shape, 1, 0.5).show()
    dls_a = RandomErosionDilation3D(Random3DMask(data.shape, 1, 0.5), structure=np.ones((2, 2, 2)))
    data_no = dls_a(data)
    simple_render(data_no)
    simple_see_slice(data_no)
    loss_func = dice()
    # 扩展通道数量
    data = np.expand_dims(data, axis=0)
    data_no = np.expand_dims(data_no, axis=0)
    # print(data.dtype, data_no.dtype)
    loss = loss_func(data, data_no)
    print(loss)
