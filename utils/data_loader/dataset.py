# -*- CODING: UTF-8 -*-
# @time 2023/11/7 18:52
# @Author tyqqj
# @File dataset.py
# @
# @Aim 


from . import transformers
from monai import data
from torch.utils.data import dataset
import numpy as np


class MutiTransformDataset(data.Dataset):
    def __init__(self, data, transform_list=None):
        self.current_transform = None
        self.data = data
        self.transform_list = transform_list
        # 如果传入的是一个transform，就直接赋值
        if isinstance(transform_list, transformers.BaseTransforms):
            self.current_transform = transform_list

    def set_transform(self, transform_name_or_index):
        # 如果是一个transform，就直接赋值
        if isinstance(self.transforms, list):
            assert isinstance(transform_name_or_index, int), "Expect an integer for transform index"
            assert transform_name_or_index < len(self.transforms), "Index out of range"
            self.current_transform = self.transforms[transform_name_or_index]
        elif isinstance(self.transforms, dict):
            assert isinstance(transform_name_or_index, str), "Expect a string for transform name"
            assert transform_name_or_index in self.transforms, f"Transform {transform_name_or_index} not found"
            self.current_transform = self.transforms[transform_name_or_index]
        else:
            raise TypeError("Transforms should be either a list or a dict")

    def __getitem__(self, index):
        sample = self.data[index]
        if self.current_transform is not None:
            sample = self.current_transform(sample)
        else:
            # 警告未使用transform
            print("Warning: no transform is used.")
        return sample

    def __len__(self):
        return len(self.data)


# 可随机读取的数据集
# class CacheDataset(data.CacheDataset):
#     def __init__(self, data, transform, cache_num=1, shuffle=False, num_workers=0, pin_memory=True):
#         super().__init__(data, transform, cache_num, shuffle, num_workers, pin_memory)
#
#     def __getitem__(self, index):
#         sample = self.data[index]
#         if self.transform is not None:
#             sample = self.transform(sample)
#         else:
#             # 警告未使用transform
#             print("Warning: no transform is used.")
#         return sample
#
#     # 随机读取n个并返回
#     def random_sample_indix(self, n):
#         # 随机选择n个索引
#         indices = np.random.choice(len(self), n, replace=False)
#         # 返回这些索引
#         return indices


class IndexedDataset(dataset):
    def __init__(self, datasets, indices):
        self.datasets = datasets
        self.indices = indices

    def __getitem__(self, index):
        dataset_index, sample_index = self.indices[index]
        return self.datasets[dataset_index][sample_index]

    def __len__(self):
        return len(self.indices)
