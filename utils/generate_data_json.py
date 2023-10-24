# -*- CODING: UTF-8 -*-
# @time 2023/10/20 16:07
# @Author tyqqj
# @File generate_data_json.py
# @
# @Aim


import json
import os


def get_a_set(data_dir="/path/to/my_dataset", path_ait=""):
    # global data_dir
    # 数据集的目录

    # 图像和标签的目录
    images_dir = os.path.join(data_dir, "image" + path_ait)
    labels_dir = os.path.join(data_dir, "label" + path_ait)
    print(images_dir, labels_dir)
    # 获取所有图像和标签文件的路径
    image_files = sorted(os.listdir(images_dir))
    label_files = sorted(os.listdir(labels_dir))
    # 创建数据列表
    datalist = [
        {"image": os.path.join(images_dir, img).replace('\\', '/'),
         "label": os.path.join(labels_dir, lbl).replace('\\', '/')}
        for img, lbl in zip(image_files, label_files)
    ]
    # 将数据列表保存为 JSON 文件
    # with open("datalist.json", "w") as f:
    # json_set = json.dumps(datalist, indent=4)
    return datalist


def get_dsets(train_dir="", val_dir="", path_ait_train="", path_ait_val=""):
    all_lists = {"train": get_a_set(train_dir, path_ait_train), "val": get_a_set(val_dir, path_ait_val)}
    # all_lists = json.dumps(all_lists)
    print(json.dumps(all_lists, indent=4))
    return all_lists


if __name__ == "__main__":
    train_dir = "D:\\zhangchaoran\\miccai_achieve\\data\\train"
    val_dir = "D:\\zhangchaoran\\miccai_achieve\\data\\test"
    all_lists = get_dsets(train_dir, val_dir, "", "")

    # print(all_lists)
    with open("../data/vessel.json", "w") as dlj:
        json.dump(all_lists, dlj, indent=4)

# get_a_set()
