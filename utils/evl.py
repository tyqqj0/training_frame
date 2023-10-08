# -*- CODING: UTF-8 -*-
# @time 2023/9/25 21:27
# @Author tyqqj
# @File evl.py
# @
# @Aim 训练指标更新器

import numpy as np
import torch
from torch.utils import data
from torch import nn

import utils.train_metrics


# 指标列表的类，实现返回参数值与字典，输出的功能
# class losses:
#     def __init__(self, dict):
#         self.dict = dict
#
#     def __getitem__(self, item):
#         # 实现返回参数值与字典
#         return self.dict[item]
#
#     # 显示指标列表
#     def __str__(self):
#         # 遍历字典键与键值
#         for key, value in self.dict.items():
#             print(key, value)

class evl:
    def __init__(self, loader, epoch=None):
        self.ACC = []
        self.SEN = []
        self.SPE = []
        self.IOU = []
        self.DSC = []
        self.PRE = []
        if isinstance(loader, torch.utils.data.DataLoader) or isinstance(loader, list):
            len = 0
            for i, data in enumerate(loader):
                len += 1
            self.image = len
        else:
            # 对于其他类型的输入, 你需要决定如何处理
            print("Invalid loader type")
            raise NotImplementedError
            # self.image = len
        self.image = len
        self.epoch = epoch

    def update(self, out, target, batch_size=-1):
        if batch_size == -1:
            batch_size = out.shape[0]
        acc, sen, spe, iou, dsc, pre = utils.train_metrics.metrics3d(out, target, batch_size)
        self.ACC.append(acc)
        self.SEN.append(sen)
        self.SPE.append(spe)
        self.IOU.append(iou)
        self.DSC.append(dsc)
        self.PRE.append(pre)
        print("epoch:{0:d}\tacc:{1:.4f}\tsen:{2:.4f}\tspe:{3:.4f}\tiou:{4:.4f}\tdsc:{5:.4f}\tpre:{6:.4f}".format(
            self.epoch + 1, acc, sen, spe, iou, dsc, pre))
        return {"ACC": acc, "SEN": sen, "SPE": spe, "IOU": iou, "DSC": dsc, "PRE": pre}

    # print(
    #     '{0:d}:] \u2501\u2501\u2501 loss:{1:.10f}\tacc:{2:.4f}\tsen:{3:.4f}\tspe:{4:.4f}\tiou:{5:.4f}\tdsc:{6:.4f}\tpre:{7:.4f}'.format
    #     (epoch + 1, loss.item(), acc / len_img, sen / len_img, spe / len_img,
    #      iou / len_img, dsc / len_img, pre / len_img))

    def print_avr(self):
        print('epoch:{0:d}\tacc:{1:.4f}\tsen:{2:.4f}\tspe:{3:.4f}\tiou:{4:.4f}\tdsc:{5:.4f}\tpre:{6:.4f}'.format
              (self.epoch + 1, np.mean(self.ACC), np.mean(self.SEN), np.mean(self.SPE),
               np.mean(self.IOU), np.mean(self.DSC), np.mean(self.PRE)))

    def end_epoch(self):
        self.print_avr()
        return {"ACC": np.mean(self.ACC), "SEN": np.mean(self.SEN), "SPE": np.mean(self.SPE),
                "IOU": np.mean(self.IOU), "DSC": np.mean(self.DSC), "PRE": np.mean(self.PRE)
                }
