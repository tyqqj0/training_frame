# -*- CODING: UTF-8 -*-
# @time 2023/10/20 16:07
# @Author tyqqj
# @File generate_data_json.py
# @
# @Aim


import json
import os


def get_a_set(data_dir):
    # global data_dir
    # 数据集的目录

    # 图像和标签的目录
    images_dir = os.path.join(data_dir["image"])
    labels_dir = os.path.join(data_dir["label"])
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


def get_dsets(train_dir, val_dir):
    all_lists = {"training": get_a_set(train_dir), "validation": get_a_set(val_dir)}
    # all_lists = json.dumps(all_lists)
    print(json.dumps(all_lists, indent=4))
    return all_lists


if __name__ == "__main__":
    train_dir = {
        "image": "D:\\gkw\\data\\misguide_data\\image",
        "label": "D:\\gkw\\data\\misguide_data\\label_dce064"
    }
    val_dir = {
        "image": "D:\\gkw\\data\\misguide_data\\image",
        "label": "D:\\gkw\\data\\misguide_data\\label"
    }
    all_lists = get_dsets(train_dir, val_dir)

    # print(all_lists)
    with open("../data/vessel.json", "w") as dlj:
        json.dump(all_lists, dlj, indent=4)

# get_a_set()
