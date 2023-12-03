# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import os
import re
import json

import numpy as np
import torch
import utils.arg.parser as psr

# import monai

from monai import data, transforms
from monai.data import load_decathlon_datalist

input_train = [
    {
        'image': 'D:/Data/brains/train/image1/Normal002.mha',
        'label': 'D:/Data/brains/train/label1/Normal002-MRA.mha'
    }
]


# def generate_list(data_root='D:Data/brains/train/', check=False, amt=-1):
#     # 读取data_root下的文件夹
#     addition_dir = ""
#     train_dirs = os.path.join(data_root, 'image', addition_dir)
#     label_dirs = os.path.join(data_root, 'label', addition_dir)
#
#     # 读取train_dirs下的文件
#     train_files = os.listdir(train_dirs)
#     label_files = os.listdir(label_dirs)
#
#     # Convert the list of filenames into a dictionary with extracted numbers as keys for faster lookup
#     label_files_dict = {re.findall(r'\d+', f)[0]: f for f in label_files}
#
#     # 生成一个空列表
#     input_train = []
#     # 遍历文件
#     for i, train_file in enumerate(train_files):
#         # Extract number from the train_file
#         number = re.findall(r'\d+', train_file)[0]
#
#         # Only add the file if a matching number exists in the label files
#         if number in label_files_dict:
#             # 生成一个空字典
#             input_dict = {}
#             # 生成文件的路径
#             train_path = os.path.join(train_dirs, train_file)
#             label_path = os.path.join(label_dirs, label_files_dict[number])
#             # 将路径添加到字典中
#             input_dict['image'] = train_path
#             input_dict['label'] = label_path
#             # 将字典添加到列表中
#             input_train.append(input_dict)
#
#         if i == (amt - 1):
#             break
#
#     if check:
#         # 如果check为True，那么就打印列表中的每个元素
#         for i in range(len(input_train)):
#             print(re.findall(r'\d+', input_train[i]['image'])[0], re.findall(r'\d+', input_train[i]['label'])[0])
#     return input_train


class Sampler(torch.utils.data.Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, make_even=True):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.shuffle = shuffle
        self.make_even = make_even
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        indices = list(range(len(self.dataset)))
        self.valid_length = len(indices[self.rank: self.total_size: self.num_replicas])

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))
        if self.make_even:
            if len(indices) < self.total_size:
                if self.total_size - len(indices) < len(indices):
                    indices += indices[: (self.total_size - len(indices))]
                else:
                    extra_ids = np.random.randint(low=0, high=len(indices), size=self.total_size - len(indices))
                    indices += [indices[ids] for ids in extra_ids]
            assert len(indices) == self.total_size
        indices = indices[self.rank: self.total_size: self.num_replicas]
        self.num_samples = len(indices)
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


# 读取数据
def get_loader(data_cfg=None, loader_cfg=None, include_ngcm=False):
    if loader_cfg is None:
        loader_cfg = "./utils/data_loader/loader_old.json"

    # 导入读取的参数
    config_reader = psr.ConfigReader(loader_cfg)
    config = config_reader.get_config()
    config_reader.check()
    arg_parser = psr.ArgParser(config)
    args = arg_parser.parse_args()
    # print(arg_parser)
    # 获取数据集配置文件
    # 如果不存在
    if data_cfg is None:
        try:
            data_cfg = args.data_cfg  # 读取loader默认数据集
        except:
            raise ValueError("can not find data_cfg")
    # 如果是路径
    if data_cfg.endswith('.json'):
        return inside_get_loader(args, data_cfg, include_ngcm=include_ngcm)
    else:
        raise ValueError("data_cfg must be a json file")


def inside_get_loader(args, data_dir_json, include_ngcm=True):
    # create a training data loader
    # data_dir = args.data_dir
    datalist_json = data_dir_json
    train_transform = transforms.Compose(  # 一系列的数据增强操作，compose是将多个操作组合起来
        [
            transforms.LoadImaged(keys=["image", "label"], image_only=False),  # 读取图像和标签
            transforms.EnsureChannelFirstd(keys=["image", "label"]),  # 增加通道维度
            transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),  # 调整方向，RAS是右手坐标系
            transforms.Spacingd(  # 调整像素间距
                keys=["image", "label"], pixdim=(args.space_x, args.space_y, args.space_z), mode=("bilinear", "nearest")
            ),
            transforms.ScaleIntensityRanged(  # 调整像素值范围，将像素值范围调整到[0,1]
                keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
            ),
            transforms.CropForegroundd(keys=["image", "label"], source_key="image"),  # 剪裁图像
            transforms.RandCropByPosNegLabeld(  # 随机裁剪, 大小为roi_x, roi_y, roi_z，全是96， 另外，正样本和负样本的比例为1:1，样本数量为4
                keys=["image", "label"],
                label_key="label",
                spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                pos=1,
                neg=1,
                num_samples=4,
                image_key="image",
                image_threshold=0,
            ),
            transforms.RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=0),  # 随机翻转
            transforms.RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=1),
            transforms.RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=2),
            transforms.RandRotate90d(keys=["image", "label"], prob=args.RandRotate90d_prob, max_k=3),  # 随机旋转90度
            transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=args.RandScaleIntensityd_prob),  # 随机缩放
            transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=args.RandShiftIntensityd_prob),  # 随机平移
            transforms.ToTensord(keys=["image", "label"]),  # 转换为tensor，因为之前的操作都是对numpy数组进行的
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"], image_only=False),
            transforms.EnsureChannelFirstd(keys=["image", "label"]),
            transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
            transforms.Spacingd(
                keys=["image", "label"], pixdim=(args.space_x, args.space_y, args.space_z), mode=("bilinear", "nearest")
            ),
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
            ),
            transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )

    if args.test_mode:
        # 测试
        test_files = load_decathlon_datalist(datalist_json, True, "validation")  # 加载测试数据集

        # test_files = generate_list(args.test_data_dir)
        test_ds = data.Dataset(data=test_files, transform=val_transform)
        test_sampler = Sampler(test_ds, shuffle=False) if args.distributed else None
        test_loader = data.DataLoader(
            test_ds,
            batch_size=1,
            shuffle=False,
            num_workers=args.workers,
            sampler=test_sampler,
            pin_memory=True,
            persistent_workers=True,
        )
        loader = test_loader
    else:
        # 如果不是测试，那么就是训练
        # 此处的datalist是一个列表，列表中的每个元素是一个字典，字典中包含了图像和标签的路径
        datalist = load_decathlon_datalist(datalist_json, True, "train")
        # datalist = generate_list(args.data_dir, check=True, amt=args.amt)
        if args.use_normal_dataset:
            # 如果使用普通的数据集，那么就不需要缓存
            train_ds = data.Dataset(data=datalist, transform=train_transform)
        else:
            # 如果使用缓存数据集，那么就需要缓存
            train_ds = data.CacheDataset(
                data=datalist, transform=train_transform, cache_num=37, cache_rate=1.0, num_workers=args.workers
            )
        # train_ds是一个数据集，包含了训练数据和标签，transform是对数据集进行的一系列操作
        train_sampler = Sampler(train_ds) if args.distributed else None
        train_loader = data.DataLoader(  # 创建训练数据集的加载器
            train_ds,
            batch_size=args.batch_size,
            shuffle=(train_sampler is None),
            num_workers=args.workers,
            sampler=train_sampler,
            pin_memory=True,
            persistent_workers=True,
        )
        val_files = load_decathlon_datalist(datalist_json, True, "val")
        # val_files = generate_list(args.val_dir)
        val_ds = data.Dataset(data=val_files, transform=val_transform)
        val_sampler = Sampler(val_ds, shuffle=False) if args.distributed else None
        val_loader = data.DataLoader(
            val_ds,
            batch_size=1,
            shuffle=False,
            num_workers=args.workers,
            sampler=val_sampler,
            pin_memory=True,
            persistent_workers=True,
        )
        vis_files = load_decathlon_datalist(datalist_json, True, "vis")
        vis_ds = data.Dataset(data=vis_files, transform=val_transform)
        vis_sampler = Sampler(vis_ds, shuffle=False) if args.distributed else None
        vis_loader = data.DataLoader(
            vis_ds,
            batch_size=1,
            shuffle=False,
            num_workers=args.workers,
            sampler=vis_sampler,
            pin_memory=True,
            persistent_workers=True,
        )
        if include_ngcm:
            ngcm_yc_files = load_decathlon_datalist(datalist_json, True, "ngcm_yc")
            ngcm_yc_ds = data.Dataset(data=ngcm_yc_files, transform=train_transform)
            ngcm_yc_sampler = Sampler(ngcm_yc_ds, shuffle=False) if args.distributed else None
            ngcm_yc_loader = data.DataLoader(
                ngcm_yc_ds,
                batch_size=1,
                shuffle=False,
                num_workers=args.workers,
                sampler=ngcm_yc_sampler,
                pin_memory=True,
                persistent_workers=True,
            )
            ngcm_y_files = load_decathlon_datalist(datalist_json, True, "ngcm_y")
            ngcm_y_ds = data.Dataset(data=ngcm_y_files, transform=train_transform)
            ngcm_y_sampler = Sampler(ngcm_y_ds, shuffle=False) if args.distributed else None
            ngcm_y_loader = data.DataLoader(
                ngcm_y_ds,
                batch_size=1,
                shuffle=False,
                num_workers=args.workers,
                sampler=ngcm_y_sampler,
                pin_memory=True,
                persistent_workers=True,
            )
            loader = [train_loader, val_loader, vis_loader, ngcm_yc_loader, ngcm_y_loader], data_dir_json
        else:
            print(train_ds, val_ds, vis_ds)
            loader = [train_loader, val_loader, vis_loader], data_dir_json

    return loader

#
# lists = generate_list('D:\\zhangchaoran\\data_301\\test', check=True)
# datas = data.Dataset(data=lists)
# print(datas)
# # 打印图像和标签的形状
# for i in range(len(datas)):
#     print(datas[i]['image'], datas[i]['label'])

# print(lists)
