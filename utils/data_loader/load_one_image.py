# -*- CODING: UTF-8 -*-
# @time 2023/11/15 21:00
# @Author tyqqj
# @File load_one_image.py
# @
# @Aim 

import itk
import numpy as np


def load_one_label(path='D:/Data/brains/train/label1/Normal002.mha', threshold=False):
    img = itk.imread(path)
    # 转换成numpy数组
    arr = itk.GetArrayViewFromImage(img)

    # 取大于零的地方
    if threshold:
        arr = arr > 0

    # 转化成float32
    arr = arr.astype(np.float32)
    # 重新加载到np
    # arr = np.array(arr)
    # arr = np.transpose(arr, (1, 2, 0)) # 因为itk读取顺序是z,x,y
    # print("img shape, min, max:", arr.shape, arr.min(), arr.max())
    return arr


def save_one_label(arr, path='D:/Data/brains/train/label1/Normal002.mha'):
    # 转换成itk数组
    img = itk.GetImageViewFromArray(arr)
    # 保存
    itk.imwrite(img, path)
