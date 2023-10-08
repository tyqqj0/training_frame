# -*- CODING: UTF-8 -*-
# @time 2023/9/26 23:06
# @Author tyqqj
# @File box.py
# @
# @Aim

import os
import time

import utils.vis

import numpy as np
import torch
import mlflow
import mlflow.pytorch
from tensorboardX import SummaryWriter

# 用来规范化保存，日志，可视化等路径
'''
:param args: 参数
:param name: 保存的文件名, 有一个默认训练名字, 如果不指定重复覆盖
:param mkdir: 是否创建文件夹,默认为True
'''


class box:
    def __init__(self, args, name='train', mkdir=True, custom_root=None, name_add_time=True, name_add_date=True):
        self.args = args
        self.mkdir = mkdir
        self.get_root_path(custom_root)  # 获取根目录
        self.name_add_time = name_add_time
        self.name_add_date = name_add_date
        self.name = self.gen_name(name)
        self.iscontinue = False
        self.paths = self.gen_box()
        self.model_name = "model.pt"
        self.test_mode = self.test
        self.best_acc = 0
        if args.logdir is not None and args.rank == 0:
            self.writer = SummaryWriter(log_dir=args.paths['vis_score'])
            if args.rank == 0:
                print("Writing Tensorboard logs to ", args.paths['vis_score'])

        # 如果iscontinue为True，判断checkpoint是否为真，如果为假，报错
        if self.iscontinue:
            if not self.args.checkpoint:
                raise RuntimeError("model exists, but checkpoint is False")

    def get_root_path(self, custom_root=None):
        # 获取路径

        # 当没有自定义保存子目录时，如果是测试模式，则为root/test，否则为root/train
        # 若无根目录，报错
        if not os.path.exists(self.args.box_root):
            raise RuntimeError("box_root does not exist")

        if custom_root is None:
            if self.args.test:
                self.root = os.path.join(self.args.box_root, 'test')
            else:
                self.root = os.path.join(self.args.box_root, 'train')
        else:
            self.root = os.path.join(self.args.box_root, custom_root)
        # # 如果名称是默认的，就存在root/default
        # if self.name == 'train':
        #     self.root = os.path.join(self.root, 'default')
        # 如果根目录不存在，创建根目录
        if not os.path.exists(self.root):
            if (self.mkdir):
                os.makedirs(self.root)
            else:
                raise RuntimeError("root space does not exist")
        # self.root = args.box_root
        return self.root

    def gen_box(self):
        # 生成box，如果box存在，就返回box，否则创建box
        '''
        box结构:
        root/name
                /model
                /log
                    /vis_score
                    /vis2d
                    /vismha
        :return:
        '''
        box_path = os.path.join(self.root, self.name)
        model_path = os.path.join(box_path, "model")
        log_path = os.path.join(box_path, "log")

        vis_score_path = os.path.join(log_path, "vis_score")
        vis2d_path = os.path.join(log_path, "vis2d")
        vismha_path = os.path.join(log_path, "vismha")

        paths = [model_path, vis_score_path, vis2d_path, vismha_path]

        for path in paths:
            if not os.path.exists(path):
                if self.mkdir:
                    os.makedirs(path)
                else:
                    raise RuntimeError("path {} does not exist".format(path))

        # model_file_path = os.path.join(model_path, "model.pt")
        model_file_path = os.path.join(model_path, self.model_name)

        if os.path.exists(model_file_path):
            self.iscontinue = True
        paths = {"box": box_path, "model": model_path, "log": log_path, "vis_score": vis_score_path,
                 "vis2d": vis2d_path, "vismha": vismha_path, "model_file": model_file_path}
        return paths

    def gen_name(self, sname):
        # 生成文件名，   yy.mm.dd/hh:mm_name(n)
        name = ''
        if self.name_add_date:
            name += time.strftime("%Y.%m.%d", time.localtime())
        if self.name_add_time:
            name += time.strftime("/%H:%M", time.localtime())
        if name != '':
            name += '_'
        name += sname
        # 如果已经存在，就在后面加上(n)
        if os.path.exists(os.path.join(self.root, name)):
            n = 1
            while os.path.exists(os.path.join(self.root, name + '(' + str(n) + ')')):
                n += 1
            name += '(' + str(n) + ')'
        return name

    def save_model(self, model, epoch, args, filename=None, best_acc=0, optimizer=None, scheduler=None):
        if filename is None:
            filename = self.model_name
        state_dict = model.state_dict() if not args.distributed else model.module.state_dict()
        save_dict = {"epoch": epoch, "best_acc": best_acc, "state_dict": state_dict}
        if optimizer is not None:
            save_dict["optimizer"] = optimizer.state_dict()
        if scheduler is not None:
            save_dict["scheduler"] = scheduler.state_dict()
        filename = os.path.join(self.paths['model'], filename)
        torch.save(save_dict, filename)
        print("Saving checkpoint ", filename)

    def vis(self, epoch, image, outputs, label=None, add_text=''):
        if (self.args.vis):
            if (self.args.vis_every > 0 and (epoch + 1) % self.args.vis_every == 0):
                # print("out")
                utils.vis.vis(self.paths['vis2d'], epoch, image, outputs, label, self.test_mode, add_text=add_text)
        if (self.args.vis3d):
            if (self.args.vis3d_every > 0 and (epoch + 1) % self.args.vis3d_every == 0):
                # print("out")
                utils.vis.vis3d(self.paths['vismha'], epoch, image, outputs, label, self.test_mode, add_text=add_text)

    def vis_score(self, loss, epoch):
        pass
