import numpy as np
import os
import glob
from tqdm import tqdm
import SimpleITK as sitk

from utils.train_metrics import metrics3d
from skimage import filters
import evl

import torch.nn as nn

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import torch, gc

# 参数
args = {
    ## .nii data ##
    'test_path': 'D:\\zhangchaoran\\data_301\\test\\',
    'pred_path': 'D:\\zhangchaoran\\data_301\\test\\'
}

# if not os.path.exists(args['pred_path']):
#     os.makedirs(args['pred_path'])


def standardization_intensity_normalization(dataset, dtype):
    mean = dataset.mean()
    std = dataset.std()
    return ((dataset - mean) / std).astype(dtype)


def rescale(img):
    w, h = img.size
    min_len = min(w, h)
    new_w, new_h = min_len, min_len
    scale_w = (w - new_w) // 2
    scale_h = (h - new_h) // 2
    box = (scale_w, scale_h, scale_w + new_w, scale_h + new_h)
    img = img.crop(box)
    return img


def load_image3d():
    test_images = []
    for file in glob.glob(os.path.join(args['test_path'].replace('\\\\', '\\'), 'image', '*.mha')):
        basename = os.path.basename(file)
        image_name = os.path.join(args['test_path'], 'image', basename)
        test_images.append(image_name)
    return test_images


def load_label3d():
    test_labels = []
    for file in glob.glob(os.path.join(args['pred_path'].replace('\\\\', '\\'), 'label', '*.mha')):
        basename = os.path.basename(file)
        label_name = os.path.join(args['pred_path'], 'label', basename)
        test_labels.append(label_name)
    return test_labels


# def load_net():
#     net = torch.load('D:/zhangchaoran/NEW_DATA_TRAIN/CT with MRI/to MRI/vnet_Dice2100.pkl')
#     # print(net)
#     net = nn.DataParallel(net).cuda()
#     return net







class mtester:
    # 初始化
    def __init__(self, net, args):
        self.net = net
        self.m_path = args.predict_model_dir

    def predict(self):
        # net = load_net().cuda()

        images = load_image3d()
        labels = load_label3d()
        evl = evl.evl(images, labels)
        with torch.no_grad():
            self.net.eval()
            # print(len(images))
            # print(len(labels))
            for i in tqdm(range(len(images))):
                image = sitk.ReadImage(images[i])
                Image = standardization_intensity_normalization(image, 'float32')
                label = sitk.ReadImage(labels[i])
                label = sitk.GetArrayFromImage(label).astype(np.float32)
                image = torch.from_numpy(np.ascontiguousarray(Image)).unsqueeze(0).unsqueeze(0)
                label = torch.from_numpy(np.ascontiguousarray(label)).unsqueeze(0)
                output = self.net(image)
