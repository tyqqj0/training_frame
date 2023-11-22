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
    # 如果有最大数量限制，则获取最大数量

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
    # if "max_amount" in data_dir.keys():

    if "max_amount" in data_dir.keys():
        max_amount = data_dir["max_amount"]
        datalist = datalist[:max_amount]
    # 将数据列表保存为 JSON 文件
    # with open("datalist.json", "w") as f:
    # json_set = json.dumps(datalist, indent=4)
    return datalist


def get_dsets(dirstt):
    # all_lists = {"train": get_a_set(train_dir), "val": get_a_set(val_dir)}
    # 遍历内容添加键值对
    all_lists = {}
    for key, value in dirstt.items():
        all_lists[key] = get_a_set(value)
    # all_lists = json.dumps(all_lists)
    print(json.dumps(all_lists, indent=4))
    return all_lists


if __name__ == "__main__":
    dirstt = {
        "train": {
            "image": "D:\\gkw\\data\\misguide_data\\image",
            "label": "D:\\gkw\\data\\misguide_data\\label_dce_front"
        },
        "val": {
            "image": "D:\\gkw\\data\\misguide_data\\image",
            "label": "D:\\gkw\\data\\misguide_data\\label",
            "max_amount": 6
        },
        "vis": {
            "image": "D:\\gkw\\data\\vis\\image",
            "label": "D:\\gkw\\data\\vis\\label",
            "max_amount": 1
        }
    }
    # train_dir = {
    #     "image": "D:\\gkw\\data\\misguide_data\\image",
    #     "label": "D:\\gkw\\data\\misguide_data\\label_dce064"
    # }
    # val_dir = {
    #     "image": "D:\\gkw\\data\\misguide_data\\image",
    #     "label": "D:\\gkw\\data\\misguide_data\\label"
    # }
    all_lists = get_dsets(dirstt)

    # print(all_lists)
    with open("../data/msg_front.json", "w") as dlj:
        json.dump(all_lists, dlj, indent=4)

# get_a_set()
