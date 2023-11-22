# -*- CODING: UTF-8 -*-
# @time 2023/11/15 20:25
# @Author tyqqj
# @File generate_misguid_data.py
# @
# @Aim
import os

import matplotlib.pyplot as plt
import numpy as np
from monai.losses import DiceLoss
from scipy.ndimage import gaussian_filter, binary_erosion, binary_dilation
from scipy.ndimage import generate_binary_structure
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


msg_arg_dice067 = {
    "discribtion": "模拟真实数据"
    ,
    "mask_generator_1": {
        "blur_size": 1,
        "threshold": 0.45
    },
    "structures": {
        "erosion": np.ones((2, 2, 2)),
        "dilation": np.ones((3, 3, 3))
    },
    "mask_generator_2": {
        "blur_size": 6,
        "threshold": 0.54
    }
}
msg_arg_dice064 = {
    "discribtion": "模拟后误导数据",
    "mask_generator_1": {
        "blur_size": 1,
        "threshold": 0.45
    },
    "structures": {
        "erosion": np.ones((2, 2, 2)),
        "dilation": np.ones((3, 3, 3))
    },
    "mask_generator_2": {
        "blur_size": 6,
        "threshold": 0.5
    }
}
msg_arg_dice056 = {
    "discribtion": "模拟全误导数据",
    "mask_generator_1": {
        "blur_size": 1,
        "threshold": 0.35
    },
    "structures": {
        "erosion": np.ones((2, 2, 2)),
        "dilation": np.ones((4, 4, 4))
    },
    "mask_generator_2": {
        "blur_size": 6,
        "threshold": 0.65
    }
}


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
        #### 归一化
        blurred_array = (blurred_array - blurred_array.min()) / (blurred_array.max() - blurred_array.min())

        # 将阈值相对于数组的最大值最小值进行缩放
        # self.threshold = self.threshold * (blurred_array.max() - blurred_array.min()) + blurred_array.min()
        # print("threshold:", self.threshold)

        # 使用一个阈值来生成掩码
        random_mask = blurred_array > self.threshold

        return random_mask, blurred_array

    def __call__(self):
        #
        return self.generate()[0]

    def show(self):
        plt.imshow(self.generate()[0][:, :, self.image_shape[2] // 2].astype(np.float32))
        plt.show()


class RandomErosionDilation3D:
    def __init__(self, mask_generator, structures=None):
        """
        创建一个随机腐蚀和膨胀操作器。

        参数:
        mask_generator: 用于生成随机掩码的生成器
        structure: 用于腐蚀和膨胀操作的结构元素。如果为None，则使用一个全为1的3x3x3的立方体。
        """
        self.mask_generator = mask_generator
        if structures is None:
            self.structures['erosion'] = np.ones((1, 1, 1))  # 进行腐蚀操作的结构元素
            self.structures['dilation'] = np.ones((2, 2, 2))  # 进行膨胀操作的结构元素
        else:
            self.structures = structures

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
        random_mask = self.mask_generator()
        # print("image shape:", image.shape)
        # print("random_mask shape:", random_mask.shape)
        # print("structure shape:", self.structures['erosion'].shape, self.structures['dilation'].shape)

        # print("masked image shape:", image[random_mask].shape)

        # 初始化输出的图像
        output_image = np.zeros_like(image)

        # # 对于掩码中为True的部分，进行腐蚀操作
        # output_image[random_mask] = binary_erosion(image[random_mask], self.structure)
        #
        # # 对于掩码中为False的部分，进行膨胀操作
        # output_image[~random_mask] = binary_dilation(image[~random_mask], self.structure)

        # # 对整个图像进行腐蚀操作，然后再使用random_mask选择结果中的一部分
        # eroded_image = binary_erosion(image, self.structures['erosion'])
        # output_image[random_mask] = eroded_image[random_mask]
        #
        # # 对整个图像进行膨胀操作，然后再使用~random_mask选择结果中的一部分
        # dilated_image = binary_dilation(image, self.structures['dilation'])
        # output_image[~random_mask] = dilated_image[~random_mask]

        # 对整个图像进行膨胀操作
        dilated_image = binary_dilation(image, self.structures['dilation'])

        # 对膨胀后的图像进行腐蚀操作
        dilated_eroded_image = binary_erosion(dilated_image, self.structures['erosion'])

        # 直接对原始图像进行腐蚀操作
        eroded_image = binary_erosion(image, self.structures['erosion'])

        # 使用随机掩码对步骤2和步骤3的结果进行拼接
        # output_image = np.zeros_like(image)
        output_image[random_mask] = dilated_eroded_image[random_mask]
        output_image[~random_mask] = eroded_image[~random_mask]

        # print("output_image shape: ", output_image.shape)
        return output_image.reshape(shape)

    def __call__(self, image):
        return self.apply(image)


class RandomDelete3D:
    def __init__(self, mask_generator):
        """
        创建一个随机腐蚀和膨胀操作器。

        参数:
        mask_generator: 用于生成随机掩码的生成器
        """
        self.mask_generator = mask_generator

    def apply(self, image):
        """
        对输入的3D图像在随机区域进行删除操作。

        参数:
        image: 输入的3D图像

        返回:
        输出的3D图像
        """
        # 使用mask_generator生成随机掩码
        random_mask = self.mask_generator()

        # 使用掩码对输入图像进行修改
        output_image = image.copy()
        output_image[random_mask] = 0

        # 确保输出的形状与输入的形状相同
        output_image = output_image.reshape(image.shape)

        return output_image

    def __call__(self, image):
        return self.apply(image)


def simple_render(img, tex=""):
    img = render.rotate_3d_matrix(img, (0, 0, 0))
    # print(img.shape)
    np_img = render.renderNp(img, camera_position_scale=(-4, 0, 0))
    plt.imshow(np_img)
    if tex:
        plt.text(0, 0, tex, color='red')
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
        # 如果输入的是3维的，就扩展一下
        if len(input.shape) == 3:
            input = np.expand_dims(input, axis=0)
        if len(target.shape) == 3:
            target = np.expand_dims(target, axis=0)
        input = tensor(input)
        target = tensor(target)
        return super().forward(input, target)


def misguide_one_label(data, msg_arg=msg_arg_dice067, render=False, see_msk=False):
    masktt = Random3DMask(data.shape, **msg_arg['mask_generator_1'])
    # structures = {'erosion': generate_binary_structure(3, 0), 'dilation': generate_binary_structure(3, 2)}
    dls_a = RandomErosionDilation3D(masktt, structures=msg_arg['structures'])

    masktt2 = Random3DMask(data.shape, **msg_arg['mask_generator_2'])
    dls_b = RandomDelete3D(masktt2)
    if see_msk:
        masktt.show()
        masktt2.show()
    data_no = dls_a(data)
    data_no2 = dls_b(data_no)

    # loss = ''
    loss_func = dice()
    loss = loss_func(data, data_no2)
    if render:
        print("loss:", loss)
        simple_render(data_no2, tex=f"dice: {1 - loss:.4f}")
    return data_no2, loss


def add_misguide_to_dataset(input_folder, output_folder, msg_arg=msg_arg_dice067, render=False):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    losses = []
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.endswith('.mha'):
                file_path = os.path.join(root, file)
                img = load_one_image.load_one_label(file_path)

                # 转换为numpy数组，添加噪声，然后再转换回SimpleITK Image

                img_noisy, loss = misguide_one_label(img, msg_arg=msg_arg, render=render, see_msk=False)
                print("file {} loss: {}".format(file, loss))
                # 将噪声图像写入到输出文件夹
                output_file_path = os.path.join(output_folder, file)
                load_one_image.save_one_label(img_noisy, output_file_path)
                losses.append(loss)
    print("average loss:", np.mean(losses))


if __name__ == "__main__":
    # 定义参数解析器
    # parser = argparse.ArgumentParser(description="monai training arguments")
    # 解析出输入参数
    # args = parser.parse_args()
    input_path = "D:/gkw/data/misguide_data/label"
    output_path = "D:/gkw/data/misguide_data/label_dce_front"
    add_misguide_to_dataset(input_path, output_path, msg_arg=msg_arg_dice056, render=True)
