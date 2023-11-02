# -*- CODING: UTF-8 -*-
# @time 2023/10/17 16:38
# @Author tyqqj
# @File __init__.py.py
# @
# @Aim

import utils.arg


def get_args():
    cfg_file = './networks/UNETR/UNETR.json'
    args = utils.arg.get_args(cfg_file)
    return args
# get_args()
