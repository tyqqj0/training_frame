# -*- CODING: UTF-8 -*-
# @time 2023/11/7 11:26
# @Author tyqqj
# @File transformers.py
# @
# @Aim 


from typing import List

from monai import transforms

from ..arg import get_args


class BaseTransforms:
    def __init__(self, keys: List[str]):
        self.keys = keys
        self.transform_list = []  # 用于存储transform的list，子类中会被赋值

    def __call__(self, *args, **kwargs):
        return self.get_transforms()

    def tfs(self):
        return self.get_transforms()

    def get_transforms(self):
        self.gen_list()
        if len(self.transform_list) == 0:
            raise ValueError("transform_list is empty.")
        return transforms.Compose(self.transform_list)

    def gen_list(self):
        '''
        生成transform_list
        子类中需要重写
        在__init__中调用
        '''
        pass

    def __add__(self, other):
        # 不推荐使用+，要注意前后的顺序
        return transforms.Compose(self.transform_list + other.transform_list)


class BaseVesselTransforms(BaseTransforms):
    def __init__(self, cfg_file, keys: List[str], check: bool = False):
        super().__init__(keys)
        # 如果调用该基类，才会check参数
        check = True if self.__class__.__name__ == 'BaseVesselTransforms' else False
        self.keys = keys
        # 从logrbox中获取默认的参数
        self.args = get_args(cfg_file, check=check)


class vessel_train_transforms(BaseVesselTransforms):
    def __init__(self, crop_size=None, cfg_file='./transforms_default.json', keys: List[str] = ['image', 'label'],
                 check: bool = False):
        super().__init__(cfg_file, keys, check)
        # self.args = get_args(cfg_file, check=False)
        self.space_x = self.args.space_x
        self.space_y = self.args.space_y
        self.space_z = self.args.space_z
        self.roi_x = self.args.roi_x
        self.roi_y = self.args.roi_y
        self.roi_z = self.args.roi_z
        self.a_min = self.args.a_min
        self.a_max = self.args.a_max
        self.b_min = self.args.b_min
        self.b_max = self.args.b_max
        self.RandFlipd_prob = self.args.RandFlipd_prob
        self.RandRotate90d_prob = self.args.RandRotate90d_prob
        self.RandScaleIntensityd_prob = self.args.RandScaleIntensityd_prob
        self.RandShiftIntensityd_prob = self.args.RandShiftIntensityd_prob
        if crop_size is not None:
            if len(crop_size) != 3:
                raise ValueError("crop_size must be a list of 3 integers.")
            self.roi_x = crop_size[0]
            self.roi_y = crop_size[1]
            self.roi_z = crop_size[2]

    def gen_list(self):
        self.transform_list = [
            transforms.LoadImaged(keys=self.keys),  # 读取图像和标签
            transforms.AddChanneld(keys=self.keys),  # 增加通道维度
            transforms.Orientationd(keys=self.keys, axcodes="RAS"),  # 调整方向，RAS是右手坐标系
            transforms.Spacingd(  # 调整像素间距
                keys=self.keys, pixdim=(self.space_x, self.space_y, self.space_z),
                mode=("bilinear", "nearest")
            ),
            transforms.ScaleIntensityRanged(  # 调整像素值范围，将像素值范围调整到[0,1]
                keys=["image"], a_min=self.a_min, a_max=self.a_max, b_min=self.b_min, b_max=self.b_max, clip=True
            ),
            transforms.CropForegroundd(keys=self.keys, source_key="image"),  # 剪裁图像
            transforms.RandCropByPosNegLabeld(  # 随机裁剪, 大小为roi_x, roi_y, roi_z，全是96， 另外，正样本和负样本的比例为1:1，样本数量为4
                keys=self.keys,
                label_key="label",
                spatial_size=(self.roi_x, self.roi_y, self.roi_z),
                pos=1,
                neg=1,
                num_samples=4,
                image_key="image",
                image_threshold=0,
            ),
            transforms.RandFlipd(keys=self.keys, prob=self.RandFlipd_prob, spatial_axis=0),  # 随机翻转
            transforms.RandFlipd(keys=self.keys, prob=self.RandFlipd_prob, spatial_axis=1),
            transforms.RandFlipd(keys=self.keys, prob=self.RandFlipd_prob, spatial_axis=2),
            transforms.RandRotate90d(keys=self.keys, prob=self.RandRotate90d_prob, max_k=3),  # 随机旋转90度
            transforms.RandScaleIntensityd(keys=self.keys[0], factors=0.1, prob=self.RandScaleIntensityd_prob),
            # 随机缩放
            transforms.RandShiftIntensityd(keys=self.keys[0], offsets=0.1, prob=self.RandShiftIntensityd_prob),
            # 随机平移
            transforms.ToTensord(keys=self.keys)  # 转换为tensor，因为之前的操作都是对numpy数组进行的
        ]


class vessel_val_transforms(BaseVesselTransforms):
    def __init__(self, cfg_file='./transforms_default.json', keys: List[str] = ['image', 'label'], check: bool = False):
        super().__init__(cfg_file, keys, check)
        self.args = get_args(cfg_file, check=False)
        self.space_x = self.args.space_x
        self.space_y = self.args.space_y
        self.space_z = self.args.space_z
        self.a_min = self.args.a_min
        self.a_max = self.args.a_max
        self.b_min = self.args.b_min
        self.b_max = self.args.b_max

    def gen_list(self):
        self.transform_list = [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.AddChanneld(keys=["image", "label"]),
            transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
            transforms.Spacingd(
                keys=["image", "label"], pixdim=(self.space_x, self.space_y, self.space_z),
                mode=("bilinear", "nearest")
            ),
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=self.a_min, a_max=self.a_max, b_min=self.b_min, b_max=self.b_max, clip=True
            ),
            transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
            transforms.ToTensord(keys=["image", "label"]),
        ]


if __name__ == '__main__':
    testtf = BaseTransforms(['a', 'b'])
