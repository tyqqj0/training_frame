# -*- CODING: UTF-8 -*-
# @time 2023/11/7 19:19
# @Author tyqqj
# @File data_loader.py
# @
# @Aim 


from monai import data
from . import dataset as ds
import numpy as np


class MutiTransformDataloader(data.DataLoader):
    def __init__(self, data, transform_list=None, batch_size=1, shuffle=False, num_workers=0, pin_memory=True):
        self.mtdataset = ds.MutiTransformDataset(data, transform_list)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        super().__init__(self.mtdataset, batch_size, shuffle, num_workers, pin_memory)

    def set_transform(self, transform_name_or_index):
        self.mtdataset.set_transform(transform_name_or_index)


class MultiDatasetLoader(data.DataLoader):

    def __init__(self, datasets, ratios, batch_size=1, shuffle=True, num_workers=0, pin_memory=False):
        self.datasets = datasets
        self.ratios = ratios
        # 如果ratios之和不等于1，就归一化
        if sum(ratios) != 1:
            self.ratios = np.array(ratios) / sum(ratios)
        self.dataset_lengths = [len(dataset) for dataset in datasets]
        indices = self._create_indices()
        indexed_dataset = ds.IndexedDataset(datasets, indices)
        super().__init__(indexed_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                         pin_memory=pin_memory)

    def _create_indices(self):
        indices = []
        for dataset_index, ratio in enumerate(self.ratios):
            replace = False
            len = self.dataset_lengths[dataset_index]
            csl = int(ratio * sum(self.dataset_lengths))
            if csl == 0:
                continue
            if csl > len:
                replace = True
            sample_indices = np.random.choice(len, csl, replace=replace)
            indices.extend([(dataset_index, index) for index in sample_indices])
        np.random.shuffle(indices)
        return indices

# 可拼接loader
# t = data.CacheDataset([1, 2, 3], None, 1, False, 0, True)
# rdm = data.Randomizable(data.CacheDataset([1, 2, 3], None, 1, False, 0, True))
# a = t[:1] + t[:2]


# 比例数据集测试
# a = data.Dataset([1, 2, 3])
# b = data.Dataset([4, 5, 6])
# c = data.Dataset([7, 8, 9])
# mdl = MultiDatasetLoader([a, b, c], [0.1, 0.2, 0.7])
