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

import numpy as np
import torch
from monai import data
from monai.data import load_decathlon_datalist

import utils.arg.parser as psr
from . import transformers

# import monai

input_train = [
    {
        'image': 'D:/Data/brains/train/image1/Normal002.mha',
        'label': 'D:/Data/brains/train/label1/Normal002-MRA.mha'
    }
]




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
def get_loader(data_cfg=None, loader_cfg=None):
    if loader_cfg is None:
        loader_cfg = "./utils/data_loader/loader.json"

    # 导入读取的参数
    config_reader = psr.ConfigReader(loader_cfg)
    config = config_reader.get_config()
    arg_parser = psr.ArgParser(config)
    args = arg_parser.parse_args()
    print(arg_parser)
    # 获取数据集配置文件
    # 如果不存在
    if data_cfg is None:
        try:
            data_cfg = args.data_cfg # 读取loader默认数据集
        except:
            raise ValueError("can not find data_cfg")
    # 如果是路径
    if data_cfg.endswith('.json'):
        return inside_get_loader(args, data_cfg)
    else:
        raise ValueError("data_cfg must be a json file")


def inside_get_loader(args, data_dir_json):
    # create a training data loader
    # data_dir = args.data_dir
    datalist_json = data_dir_json
    train_transform = transformers.vessel_train_transforms(check=True)
    val_transform = transformers.vessel_val_transforms(check=True)

    if args.test_mode:
        # 测试
        test_files = load_decathlon_datalist(datalist_json, True, "validation")  # 加载测试数据集, base_dir是数据集的根目录

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
        vis_files = load_decathlon_datalist(datalist_json, True, "vis", base_dir=data_dir)
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
        print(train_ds, val_ds, vis_ds)
        loader = [train_loader, val_loader, vis_loader]

    return loader

#
# lists = generate_list('D:\\zhangchaoran\\data_301\\test', check=True)
# datas = data.Dataset(data=lists)
# print(datas)
# # 打印图像和标签的形状
# for i in range(len(datas)):
#     print(datas[i]['image'], datas[i]['label'])

# print(lists)
