# -*- CODING: UTF-8 -*-
# @time 2023/9/26 23:06
# @Author tyqqj
# @File box.py
# @
# @Aim

import os
import re
import shutil
import time

import utils.vis
import utils.evl

import numpy as np
import torch
import mlflow
import mlflow.pytorch
from tensorboardX import SummaryWriter
from monai import __version__

from torch.cuda.amp import GradScaler, autocast
from functools import partial
from monai.inferers import sliding_window_inference

# 用来规范化保存，日志，可视化等路径
'''
:param args: 参数
:param name: 保存的文件名, 有一个默认训练名字, 如果不指定重复覆盖
:param mkdir: 是否创建文件夹,默认为True
'''


# def save_checkpoint(net, optimizer, epoch, loss, metric):
#     """
#     两种保存方式，一种是直接保存模型，一种是保存参数
#     优先第一种，当第一种不可用时使用第二种，并且第二种会提示询问模型代码
#     :param net:
#     :param optimizer:
#     :param epoch:
#     :param loss:
#     :param metric:
#     :return:
#     """
#     pass


class box:
    def __init__(self, args, model):
        self.run_id = None
        self.loader_len = None
        self.epoch = None
        self.use_vis = None
        self.rank = 0
        self.epoch_stage = None
        self.evler = None
        self.args = args
        self.run = None
        self.artifact_location = None
        self.vis_3d = args.vis_3d
        self.vis_2d = args.vis_2d
        self.log_frq = args.log_frq
        self.save_frq = args.save_frq
        self.vis_2d_slice_loc = 96 / 2
        self.vis_2d_cache_loc = './run_cache/vis_2d'
        self.vis_3d_cache_loc = './run_cache/vis_3d'
        self.vis_2d_cover = args.vis_2d_cover
        self.vis_3d_cover = args.vis_3d_cover

        inf_size = [args.roi_x, args.roi_y, args.roi_z]
        self.model_inferer = partial(
            sliding_window_inference,
            roi_size=inf_size,
            sw_batch_size=args.sw_batch_size,
            predictor=model,
            overlap=args.infer_overlap,
        )

        # 实验模式检查
        if args.train and args.test:
            raise ValueError("Cannot set both train and test to True")

        if args.is_continue and not args.train:
            raise ValueError("Cannot set is_continue to True when train is False")

        # mlflow 实验设定
        mlflow.set_tracking_uri("http://localhost:5000")
        # print()
        experiment = mlflow.get_experiment_by_name(args.exp_name)
        if experiment is None:
            swc = input("experiment {} not exists, create it? (y/n)".format(args.exp_name))
            if swc == "y":
                print("create experiment: ", args.exp_name)
                mlflow.create_experiment(name=args.exp_name, tags={"mlflow.user": args.user_name, "type": "run_test"})
            else:
                raise ValueError("experiment {} not exists".format(args.exp_name))

        mlflow.set_experiment(args.exp_name)
        # 检查当前的实验
        # 输出实验的信息
        print("use experiment: ", args.exp_name)
        # print(experiment)
        print("experiment id: ", experiment.experiment_id)
        print("artifact location: ", experiment.artifact_location)

        # 检查实验的状况
        # experiment = mlflow.get_experiment_by_name(args.exp_name)
        # print(experiment)
        # mlflow 运行设置参数
        # 默认继续的运行
        if args.is_continue:
            if args.run_id is None:
                # 尝试获取最后一个运行的id
                runs = mlflow.search_runs(order_by=["attribute.start_time DESC"], max_results=1)
                if runs.empty:
                    raise ValueError(
                        "no run exists, please check if the experiment name is correct or muti-user conflict on",
                        mlflow.get_tracking_uri())  # 遇到了开多个不同源的mlflow server的问题，会在不同地方创建同名的实验
                # print(runs)
                last_run_id = runs.loc[0, "run_id"]
                # print(runs)
                print("using last run id: {}, name: {}".format(last_run_id, runs.loc[0, "tags.mlflow.runName"]))
                if last_run_id is None:
                    # raise ValueError("Cannot set is_continue to True when name is None")
                    raise ValueError("Cannot find last run id")
                args.run_id = last_run_id
                self.args.run_id = last_run_id
                # 默认使用最后一个运行的id
                self.run_id = last_run_id
            else:
                # 检查run_id是否存在
                run = mlflow.get_run(run_id=str(args.run_id))
                if run is None:
                    raise ValueError("run_id {} not exists".format(args.run_id))
                # print(runs)
                print("using run id: {}, name: {}".format(args.run_id, run.data.tags["mlflow.runName"]))
                # 默认使用最后一个运行的id
                self.run_id = args.run_id
                self.args.run_id = args.run_id

        if args.test and args.run_id is None:
            # 询问是否使用最后一个运行的id
            swc = input("run_id is None, use last run id? (y/n)")
            if swc == "y":
                runs = mlflow.search_runs(order_by=["attribute.start_time DESC"], max_results=1)
                if runs.empty:
                    raise ValueError(
                        "no run exists, please check if the experiment name is correct or muti-user conflict on",
                        mlflow.get_tracking_uri())
                last_run_id = runs.loc[0, "run_id"]
                print("using last run id: {}, name: {}".format(last_run_id, runs.loc[0, "tags.mlflow.runName"]))
                if last_run_id is None:
                    raise ValueError("Cannot set is_continue to True when name is None")
                args.run_id = last_run_id
            else:
                # raise ValueError("Cannot set test to True when run_id is None")
                raise ValueError("Cannot set test to True when name is None")

        # 置入内部参数
        self.artifact_location = experiment.artifact_location

    # 内部函数 标准化标签
    def _normalize_tag(self, tag=None):
        mlflow.set_tag("mlflow.user", self.args.user_name)
        mlflow.set_tag("mlflow.note.content", "test" if self.args.test else "train")
        mlflow.set_tag("mlflow.note.run_id", self.args.run_id if self.args.run_id is not None else self.run.info.run_id)
        # mlflow 参数
        mlflow.log_param("monai_version", __version__)
        for k, v in vars(self.args).items():
            mlflow.log_param(k, v)

    def start_epoch(self, loader, stage, epoch, use_vis=None):
        self.epoch = epoch
        self.epoch_stage = stage
        if use_vis is None:
            if self.args.test or (stage is 'val'):
                self.use_vis = True
        self.evler = utils.evl.evl(loader, epoch)
        # 计算loader长度
        len = 0
        for i, data in enumerate(loader):
            len += 1
        self.loader_len = len
        # if use_vis:
        #     self.writer = SummaryWriter(os.path.join(self.artifact_location, self.run.info.run_id, "log", stage))

        # self.evler = utils.evl.evl(loader, epoch)

        # 清空并创建文件夹
        for dirpath in [self.vis_2d_cache_loc, self.vis_3d_cache_loc]:
            if os.path.exists(dirpath):
                for filename in os.listdir(dirpath):
                    filepath = os.path.join(dirpath, filename)
                    if os.path.isfile(filepath):
                        os.remove(filepath)
                    elif os.path.isdir(filepath):
                        shutil.rmtree(filepath)
            else:
                os.makedirs(dirpath)

    # 参数
    # 计算参数列表,获取参数
    # self._normalize_tag()
    # self._normalize_tag(stage)
    # self._normalize_tag(epoch)
    # self._normalize_tag(use_vis)

    # 保存
    # self.log(stage, epoch, input, output, target, loss, metric, use_vis)

    def update_in_epoch(self, step, out, target, batch_size=-1, stage="train"):
        print("box updating")
        # 如果out是概率，我们需要转换成预测, 判断最小值是否为0
        if out.min() < 0:
            target = target[1] < 0

        # 如果当前阶段是训练（train）阶段，我们需要进行参数更新
        if stage == "train":
            metrics_dict = self.evler.update(out, target, batch_size) # 暂时
            # 记录和上传参数
            for metric, value in metrics_dict.items():
                # step = self.epoch + step * 1 / self.loader_len # 不能用浮点
                step = self.epoch * self.loader_len + step
                mlflow.log_metric(metric + '_in_epoch', value, step=step)

        # 如果当前阶段是验证（val）阶段，我们只需要计算和显示参数
        elif stage == "val":
            # with torch.no_grad():
            metrics_dict = self.evler.update(out, target, batch_size)
            # 记录和上传参数, val阶段不需要上传参数
            # for metric, value in metrics_dict.items():
            #     mlflow.log_metric(metric, value, step=self.epoch)

        else:
            # test目前打算不在这里做
            print("Invalid stage")
            raise ValueError

    # 保存
    def end_epoch_log(self, model, loader):
        # if use_vis is None:
        #     if self.args.test or (stage is 'val'):
        #         use_vis = True
        # 获取first_batch
        first_batch = None
        for i, data in enumerate(loader):
            first_batch = data
            break
        # 参数
        # 计算参数列表,获取参数
        metrics_dict = self.evler.end_epoch()
        for metric, value in metrics_dict.items():
            mlflow.log_metric(metric + '_' + self.epoch_stage, value, step=self.epoch)

        if self.log_frq is not None and self.use_vis:
            if self.epoch % self.log_frq == 0:
                # 显示
                print("loging epoch: ", self.epoch)
                start_time = time.time()
                # 测试一次运行的
                with torch.no_grad():
                    if isinstance(first_batch, list):
                        data, target = first_batch
                    else:
                        data, target = first_batch["image"], first_batch["label"]
                    data, target = data.cuda(self.rank), target.cuda(self.rank)
                    # print(data.shape)
                    with autocast(enabled=self.args.amp):
                        if self.model_inferer is not None and data.shape[-1] != 96:  # TODO: 这里是干啥的
                            logits = self.model_inferer(data)
                        else:
                            logits = model(data)
                    # logits = model(data)

                    # logits = model(data)
                    if not logits.is_cuda:
                        target = target.cpu()
                # 可视化
                if self.vis_2d:
                    utils.vis.vis(self.vis_2d_cache_loc, self.epoch, image=data, outputs=logits, label=target,
                                  add_text=self.epoch_stage, rank=self.rank)
                    # 检查缓存位置是否存在
                    if not os.path.exists(self.vis_2d_cache_loc):
                        raise ValueError("vis_2d_cache_loc not exists")
                    # 找到缓存的文件，并且上传到mlflow上面
                    for filename in os.listdir(self.vis_2d_cache_loc):
                        filepath = os.path.join(self.vis_2d_cache_loc, filename)
                        if os.path.isfile(filepath):
                            mlflow.log_artifact(filepath, artifact_path="vis_2d")

                if self.vis_3d:
                    utils.vis.vis_mha(self.vis_3d_cache_loc, self.epoch, image=data, outputs=logits, label=target,
                                      add_text=self.epoch_stage, rank=self.rank)
                    # 检查缓存位置是否存在
                    if not os.path.exists(self.vis_3d_cache_loc):
                        raise ValueError("vis_3d_cache_loc not exists")
                    # 找到缓存的文件夹，并且上传到mlflow上面
                    for filename in os.listdir(self.vis_3d_cache_loc):
                        filepath = os.path.join(self.vis_3d_cache_loc, filename)  # 应该是一个文件夹
                        # 检查是否是文件夹
                        if os.path.isdir(filepath):
                            mlflow.log_artifacts(filepath, artifact_path="vis_3d/" + filename)
                        # if os.path.isfile(filepath):
                        #     mlflow.log_artifacts(filepath, artifact_path="vis_3d/")
                end_time = time.time()
                print("vis using time: ", end_time - start_time)

    def save_model(self, model, epoch, args, filename=None, best_acc=0, optimizer=None, scheduler=None):
        if filename is None:
            filename = 'model.pt'
        state_dict = model.state_dict() if not args.distributed else model.module.state_dict()
        save_dict = {"epoch": epoch, "best_acc": best_acc, "state_dict": state_dict}
        if optimizer is not None:
            save_dict["optimizer"] = optimizer.state_dict()
        if scheduler is not None:
            save_dict["scheduler"] = scheduler.state_dict()

        # 假设 self.artifact_location 是 'mlflow-artifacts:/243272572310566340'
        location = self.artifact_location

        # 使用正则表达式从artifact_location中获取最后的数字
        match = re.search(r'/(\d+)$', location)
        if match is not None:
            number = match.group(1)
        else:
            raise ValueError("No number found in artifact location")

        # 附加到'.art/model'路径中，并在数字前后添加'.'
        new_path = os.path.join('./mlruns', number, 'models')  # mlruns文件夹下不能随便有model
        filename = os.path.join(new_path, filename)
        if not os.path.exists(new_path):
            os.mkdir(new_path)
        torch.save(save_dict, filename)
        print("Saving checkpoint ", filename)

    def __enter__(self):
        # global args
        # 出示服务器
        print("mlflow server: ", mlflow.get_tracking_uri())
        self._normalize_tag()
        # 模拟参数
        run_name = None
        if self.args.new_run_name is not None:
            run_name = self.args.exp_name + '-' + self.args.new_run_name
        # run_id是用来指定运行的，run_name是用来新建的，都可以没有但是功能不共用
        if self.args.run_id is not None:
            print("using run id: ", self.args.run_id)
            self.run = mlflow.start_run(run_id=self.run_id, run_name=run_name)
        else:
            print("using new run name: ", run_name)
            self.run = mlflow.start_run(run_name=run_name)
        return self.run

    def __exit__(self, exc_type, exc_val, exc_tb):
        # save_checkpoint()
        # 记录最后运行run_id
        mlflow.log_param("last_run_id", self.run.info.run_id)
        # 出示服务器
        print("run {} finished".format(self.run.info.run_name))
        print("mlflow server: ", mlflow.get_tracking_uri())
        return mlflow.end_run()
