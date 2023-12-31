# -*- CODING: UTF-8 -*-
# @time 2023/9/26 23:06
# @Author tyqqj
# @File BOX.py
# @
# @Aim

import os
import shutil
import time
import json
import argparse

from mlflow import MlflowClient
from monai.transforms import AsDiscrete
import numpy as np

import utils.BOX.vis
import utils.BOX.evl
import utils.arg.parser

import torch
import mlflow
import mlflow.pytorch
from monai import __version__

from torch.cuda.amp import autocast
from functools import partial
from monai.inferers import sliding_window_inference
import SimpleITK as sitk

# 用来规范化保存，日志，可视化等路径
'''
运行顺序请遵循
1. 创建box __init__
2. 设置模型推理器 set_model_inferer
3. load_model
4. 进入box上下文中 __enter__

在train/val中
    train
        5. start_epoch
        6. update_in_epoch
        7. end_epoch
    val
        5. start_epoch
        6. update_in_epoch
        7. end_epoch
    8. visualizes
    9. save_model
9. 退出box上下文中 __exit__
'''


def parser_cfg_loader(mode='train', path=""):
    if path != "":
        mode = path
    else:
        mode = os.path.join(".\\utils\\BOX\\cfg", mode + ".json")
    cfg = {}  # 默认的空配置
    if os.path.exists(mode):
        config_reader = utils.arg.parser.ConfigReader(mode)
        cfg = config_reader.get_config()
        arg_parser = utils.arg.parser.ArgParser(cfg)
        args = arg_parser.parse_args()
    else:
        raise ValueError("mode {} not exists".format(mode))
    return args


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
    def __init__(self, mode='train', path=''):
        self.vis_loader = None
        self.model_name = None
        self.threshold = None
        self.post_pred = None
        self.post_label = None
        args = parser_cfg_loader(mode=mode, path=path)
        # stop_all_runs()
        # return
        self.mode = mode
        self.best_acc = -1
        self.default_model_name = None
        self.run_id = None
        # self.loader = None
        self.loader_len = None
        self.epoch = None
        self.use_vis = None
        self.rank = 0
        self.epoch_stage = None
        self.evler = None
        self.args = args
        # print(self.args)
        # self.check_args()
        self.run = None
        self.artifact_location = None
        self.vis_3d = args.vis_3d
        self.vis_2d = args.vis_2d
        self.vis_2d_tb = args.vis_2d_tb
        self.log_frq = args.log_frq
        self.save_frq = args.save_frq
        self.vis_3d_frq = args.vis_3d_frq
        self.vis_2d_slice_loc = 96 / 2
        self.vis_2d_cache_loc = './run_cache/vis_2d'
        self.vis_2d_tb_cache_loc = './run_cache/vis_2d_tb'
        self.vis_3d_cache_loc = './run_cache/vis_3d'
        self.model_inferer = None
        self.signatures = None
        # self.epoch_start_time = {}
        self.timer = None

        print_line('up')
        print("Initializing BOX")

        # 实验模式检查
        # if self.mode is None:
        #     raise RuntimeError()

        print('set mlflow')
        # mlflow 实验设定
        mlflow.set_tracking_uri(args.log_url)

        # print()
        experiment = mlflow.get_experiment_by_name(args.exp_name)
        if experiment is None:
            swc = input("experiment {} not exists, create it? (y/n)".format(args.exp_name))
            if swc == "y":
                print("create experiment: ", args.exp_name)
                experiment_id = mlflow.create_experiment(name=args.exp_name,
                                                         tags={"mlflow.user": args.user_name, "type": "run_test"})
                experiment = mlflow.get_experiment(experiment_id)
            else:
                raise ValueError("experiment {} not exists".format(args.exp_name))

        mlflow.set_experiment(args.exp_name)
        # 检查当前的实验
        # 输出实验的信息
        print("use experiment: ", args.exp_name)
        # print(experiment)
        print("experiment id: ", experiment.experiment_id)
        # 设置工件位置
        # mlflow.set_artifact_location(args.artifact_dir)
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
                # self.args.run_id = last_run_id
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
                # self.args.run_id = args.run_id

        if args.mode == "test" and args.run_id is None:
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
        print('Initializing BOX complete')
        print_line('down')

    def check_args(self):
        print("BOX load args: \n", json.dumps(vars(self.args), indent=4))
        return

    def set_model_inferer(self, model, out_channels, vis_loader, inf_size=[96, 96, 96], threshold=0):
        print("set model inferer")
        self.threshold = threshold
        self.post_label = AsDiscrete(to_onehot=out_channels, n_classes=out_channels)  # 将数据onehot 应该是
        self.post_pred = AsDiscrete(argmax=True, to_onehot=out_channels, n_classes=out_channels)
        self.model_inferer = partial(
            sliding_window_inference,
            roi_size=inf_size,
            sw_batch_size=1,
            predictor=model,
            overlap=self.args.infer_overlap,  # overlap是重叠的部分
        )
        # 检查vis_loader长度是否为1
        for i, _ in enumerate(vis_loader):
            if i > 0:
                raise ValueError("vis_loader length must be 1")
        self.vis_loader = vis_loader
        if self.vis_3d:
            print("vis_3d is True, logging vis data image and label to mlflow")
            # 保存vis_loader的第一个batch的image和label到mlflow
            first_batch = None
            try:
                for i, adata in enumerate(self.vis_loader):
                    first_batch = adata
                    break
                if first_batch is None:
                    raise ValueError("first batch is None")
            except:
                first_batch = adata
            # 测试一次运行的
            if isinstance(first_batch, list):
                first_batch, target = first_batch
            else:
                first_batch, target = first_batch["image"], first_batch["label"]
            img = first_batch[0].squeeze(0).cpu()
            lb = target[0].squeeze(0).cpu()
            img = sitk.GetImageFromArray(img.astype(np.float64))
            lb = sitk.GetImageFromArray(lb.astype(np.float64))
            file_name = self.vis_3d_cache_loc
            sitk.WriteImage(img, file_name + "/image.mha")
            sitk.WriteImage(lb, file_name + "/label.mha")
            upload_cache(self.vis_3d_cache_loc, "vis_3d")

    # 内部函数 标准化标签
    def _normalize_tag(self, tag=None):
        mlflow.set_tag("mlflow.user", self.args.user_name)
        mlflow.set_tag("mlflow.note.content", self.args.mode)
        mlflow.set_tag("mlflow.note.run_id", self.args.run_id if self.args.run_id is not None else self.run.info.run_id)
        # mlflow 参数
        mlflow.log_param("monai_version", __version__)
        # 设置网络名称
        try:
            mlflow.log_param("model_name", self.model_name)
        except:
            pass
        for k, v in vars(self.args).items():
            mlflow.log_param(k, v)

    def start_epoch(self, loader, stage, epoch, use_vis=None):
        # self.epoch_start_time = time.time()
        if stage == 'train':
            self.timer = epoch_timer()
            self.timer.start(epoch)
        print(f"BOX start {stage} epoch: ", epoch + 1)
        self.epoch = epoch
        self.epoch_stage = stage
        self.use_vis = use_vis
        if use_vis is None:
            if self.args.mode == 'test' or (stage == 'val'):
                self.use_vis = True
        self.evler = utils.BOX.evl.evl(loader, epoch)
        # 计算loader长度
        lenghtt = 0
        for i, _ in enumerate(loader):
            lenghtt += 1
        self.loader_len = lenghtt
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
        # print("BOX updating")
        # 如果out是概率，我们需要转换成预测, 判断最小值是否为0
        # if out.min() < 0:
        out = out > self.threshold

        # 如果当前阶段是训练（train）阶段，我们需要进行参数更新
        if stage == "train":
            metrics_dict = self.evler.update(out, target, batch_size, stage)  # 暂时
            # 记录和上传参数
            for metric, value in metrics_dict.items():
                # step = self.epoch + step * 1 / self.loader_len # 不能用浮点
                step = self.epoch * self.loader_len + step
                mlflow.log_metric('in_epoch_' + metric, value, step=step)

        # 如果当前阶段是验证（val）阶段，我们只需要计算和显示参数
        elif stage == "val":
            # with torch.no_grad():
            self.evler.update(out, target, batch_size)
            # 记录和上传参数, val阶段不需要上传参数
            # for metric, value in metrics_dict.items():
            #     mlflow.log_metric(metric, value, step=self.epoch)

        else:
            # test目前打算不在这里做
            print("Invalid stage")
            raise ValueError

    # 保存
    def end_epoch(self):
        # if use_vis is None:
        #     if self.args.test or (stage is 'val'):
        #         use_vis = True
        # 获取first_batch

        # 参数
        # 计算参数列表,获取参数

        print_line('up')
        print("BOX end epoch, epoch: ", self.epoch + 1, " stage: ", self.epoch_stage)
        metrics_dict = self.evler.end_epoch()
        print_line('down')
        for metric, value in metrics_dict.items():
            mlflow.log_metric(self.epoch_stage + '_' + metric, value, step=self.epoch + 1)

    # TODO: box只有基础参数托管，通用性不强，需要改进
    # def calculate_metrics(self, model, loader, epoch):
    #     # 计算附加的指标指标
    #     print_line('up')
    #     print("BOX calculate metrics, epoch: ", epoch + 1)
    #     print_line('down')

    # self.visualizes(loader, model)

    def visualizes(self, model):
        loader = self.vis_loader

        if self.log_frq is not None and self.use_vis:
            if (self.epoch + 1) % self.log_frq == 0:
                # 显示
                print("visualize epoch: ", self.epoch + 1)
                start_time = time.time()
                data, logits, output, target = self.predict_one_3d(loader, model, self.threshold)
                if self.vis_2d:
                    utils.BOX.vis.vis_2d(self.vis_2d_cache_loc, self.epoch, image=data, logits=logits, outputs=output,
                                         label=target,
                                         add_text=self.epoch_stage, rank=self.rank)
                    upload_cache(self.vis_2d_cache_loc, "vis_2d")
                    print('vis_2d complete')

                if self.vis_2d_tb:
                    utils.BOX.vis.vis_2d_tensorboard(self.vis_2d_tb_cache_loc, self.epoch, image=data, logits=logits,
                                                     outputs=output,
                                                     label=target,
                                                     add_text=self.epoch_stage, rank=self.rank)
                    upload_cache(self.vis_2d_tb_cache_loc, "vis_2d_tb")
                    print('vis_2d_tensorboard complete')

                if self.vis_3d and self.vis_3d_frq is not None and (self.epoch + 1) % self.vis_3d_frq == 0:
                    utils.BOX.vis.vis_mha(self.vis_3d_cache_loc, self.epoch, image=data, logits=logits, outputs=output,
                                          label=target,
                                          add_text=self.epoch_stage, rank=self.rank)
                    upload_cache(self.vis_3d_cache_loc, "vis_3d")
                    print('vis_3d complete')
                end_time = time.time()
                print("vis using time: ", end_time - start_time)

    def predict_one_3d(self, data, model, threshold=0):
        # 如果data是一整个loader, 找到第一个batch

        first_batch = None
        try:
            for i, adata in enumerate(data):
                first_batch = adata
                break
            if first_batch is None:
                raise ValueError("first batch is None")
        except:
            first_batch = data
        # 测试一次运行的
        with torch.no_grad():
            if isinstance(first_batch, list):
                first_batch, target = first_batch
            else:
                first_batch, target = first_batch["image"], first_batch["label"]
            first_batch, target = first_batch.cuda(self.rank), target.cuda(self.rank)
            # print(first_batch.shape)
            with autocast(enabled=True):  # TODO: 这里是干啥的
                if self.model_inferer is not None and first_batch.shape[-1] != 96:
                    logits = self.model_inferer(first_batch)
                else:
                    if first_batch.shape[-1] == 96:
                        # logits = model(first_batch)
                        print("input not match and model_inferer is None, please set model_inferer")
                    logits = model(first_batch)
            # logits = model(first_batch)

            # logits = model(first_batch)
            if not logits.is_cuda:
                target = target.cpu()

        data = first_batch.squeeze(0).cpu()
        logits = logits.squeeze(0).cpu()
        # output = self.post_pred(logits)
        output = logits > threshold
        target = self.post_label(target.squeeze(0))
        if 0:
            print("data shape:", data.shape)
            print("logits shape:", logits.shape)
            print("output shape:", output.shape)
            print("target shape:", target.shape)

        # 处理数据到三维张量
        # (batch, channel, x, y, z) -> (x, y, z)
        if len(logits.shape) == 4:  # 检查张量的维度是否为5
            print("cutting 5d tensor")
            # 如果输出是通道是二，保留第二个通道
            if logits.shape[0] == 2:
                logits = logits[1:2, :, :, :]
            if output.shape[0] == 2:
                output = output[1:2, :, :, :]
            if target.shape[0] == 2:
                target = target[1:2, :, :, :]
            data = data.squeeze(0)
            logits = logits.squeeze(0)
            output = output.squeeze(0)
            target = target.squeeze(0)

        else:
            raise ValueError("logits shape not match")

        # if self.signatures is None:
        #     self.signatures = infer_signature(first_batch, logits)
        # 可视化

        return data, logits, output, target

    def save_model(self, model, epoch, filename=None):
        speed = self.timer.end()
        mlflow.log_metric("epoch/h", speed, step=self.epoch)
        print("BOX saving model")
        if filename is None:
            filename = self.default_model_name

        # 保存模型
        # 检查是否应保存模型
        if (epoch + 1) % self.save_frq != 0:
            return

        # 结束 epoch 并获取准确度信息
        metrics = self.evler.end_epoch()
        accuracy = metrics['DSC']  # 假设 evler.end_epoch() 返回一个字典，其中包含准确度

        # 记录当前的 epoch
        mlflow.log_metric(f"{filename}_epoch", epoch)
        mlflow.log_metric(f"{filename}_accuracy", accuracy)
        # 使用 mlflow.pytorch.save_model 保存模型
        mlflow.pytorch.log_model(model, filename, registered_model_name=filename, signature=self.signatures)

        # 检查是否应更新 best_acc 并保存最佳模型
        if accuracy > self.best_acc:
            print("BOX saving best model")
            self.best_acc = accuracy
            # # 删除旧的最佳模型
            # best_model_path = f"{self.artifact_location}/{filename}_best"
            # if os.path.isfile(best_model_path):  # 如果是文件，使用os.remove()
            #     os.remove(best_model_path)
            # elif os.path.isdir(best_model_path):  # 如果是目录，使用shutil.rmtree()
            #     shutil.rmtree(best_model_path)
            # 保存最佳模型
            mlflow.log_metric(f"{filename}_best_epoch", epoch)
            mlflow.log_metric(f"{filename}_best_accuracy", accuracy)
            mlflow.pytorch.log_model(model, filename + "-best", registered_model_name=filename + "_best",
                                     signature=self.signatures)

    def load_model(self, model, set_model_name="unetr", load_run_id=None, dict=True, model_version='latest',
                   best_model=True, load_model_name=None):
        print(load_run_id)
        self.model_name = set_model_name
        # 加载模型
        # 检查是否应加载模型
        if not self.args.is_continue and (load_run_id is None or load_run_id is '') and (
                load_model_name is None or load_model_name is ''):
            return model, 0, -1
        # 默认从继续运行的run_id中加载模型
        if load_run_id is None or load_run_id is '':
            load_run_id = self.run_id
        if load_run_id is None:
            return

        # models:/<model_name>/<model_version>
        # 从给定run_id获取模型名称
        # 获取运行的详细信息
        client = mlflow.tracking.MlflowClient()
        run = client.get_run(load_run_id)
        if run is None:
            raise ValueError("run_id {} not exists".format(load_run_id))

        # 使用运行的名称作为模型名称
        model_name = run.data.tags['mlflow.runName']
        if load_model_name is not None and load_model_name is not '':
            model_name = load_model_name
        if best_model:
            model_name += "_best"

        # 检查模型名称
        if set_model_name not in model_name:
            raise ValueError("model_name {} not match set_model_name {}".format(model_name, set_model_name))
        # 从模型注册中心加载模型
        print("loading model")
        model_uri = f"models:/{model_name}/{model_version}"
        loaded_model = mlflow.pytorch.load_model(model_uri)
        print("model load complete")

        if model_version == 'latest':
            model_version = get_latest_model_version(model_name)
            # model_version = [model_version, "latest"]

        # 获取模型的批次准确
        epoch, accuracy = get_model_epoch_and_accuracy(model_name, model_version)
        print(f"Loaded model: {model_name}, version: {model_version}, epoch: {epoch}, accuracy: {accuracy}")

        # 将加载的模型参数复制到当前模型
        if dict:
            model.load_state_dict(loaded_model.state_dict())
        else:
            model = loaded_model

        return model, int(epoch), accuracy

    def get_frq(self):
        return self.args.val_frq

    def __enter__(self):
        # global args
        # 出示服务器
        print("mlflow server: ", mlflow.get_tracking_uri())

        # 模拟参数
        run_name = None
        model_name = self.model_name
        if model_name is None:
            model_name = ''
        if self.args.new_run_name is not None:
            run_name = self.args.exp_name + '-' + model_name + '-' + self.args.new_run_name
        # run_id是用来指定运行的，run_name是用来新建的，都可以没有但是功能不共用
        if self.run_id is not None:
            print("using run id: ", self.run_id)
            self.run = mlflow.start_run(run_id=self.run_id, run_name=run_name)
        else:
            print("using new run name: ", run_name)
            self.run = mlflow.start_run(run_name=run_name)
        if not self.args.is_continue:
            self._normalize_tag()  # 注意一些方法会提前启动运行
        '''
        mlflow.log_param()
        mlflow.log_params()
        mlflow.log_metric()
        mlflow.log_metrics()
        mlflow.set_tag()
        mlflow.set_tags()
        mlflow.log_artifact()
        mlflow.log_artifacts()
        '''
        artifact_uri = mlflow.get_artifact_uri()
        print(f"Artifact URI: {artifact_uri}")
        # 将self.default_modelname设置为run_name
        if self.default_model_name is None:
            self.default_model_name = self.run.info.run_name
        return self.run

    def __exit__(self, exc_type, exc_val, exc_tb):
        # save_checkpoint()
        # 记录最后运行run_id
        mlflow.log_param("last_run_id", self.run.info.run_id)
        # 出示服务器
        print("run {} finished".format(self.run.info.run_name))
        print("mlflow server: ", mlflow.get_tracking_uri())
        return mlflow.end_run()


def upload_cache(cache_loc, artifact_path=None):
    if artifact_path is None:
        artifact_path = os.path.basename(cache_loc)  # 例:./runcache/vis_2d_200 -> vis_2d_200
    # 检查缓存位置是否存在
    if not os.path.exists(cache_loc):
        raise ValueError("vis_3d_cache_loc not exists")
    # 如果是文件
    if os.path.isfile(cache_loc):
        # 找到缓存的文件
        for filename in os.listdir(cache_loc):
            file_path = os.path.join(cache_loc, filename)
            if os.path.isfile(file_path):
                mlflow.log_artifact(file_path, artifact_path=artifact_path)
    elif os.path.isdir(cache_loc):
        for filename in os.listdir(cache_loc):
            file_path = os.path.join(cache_loc, filename)
            mlflow.log_artifacts(file_path, artifact_path=artifact_path)
    else:
        raise ValueError("vis_3d_cache_loc is not file or dir")


def stop_all_runs():
    # 创建一个 MlflowClient 实例
    client = MlflowClient()
    # 获取所有的运行
    experiment_id = client.get_experiment_by_name("traint1").experiment_id
    runs = client.search_runs(experiment_ids=[experiment_id])
    print("experiment_id:", experiment_id)
    # 遍历所有的运行
    for run in runs:
        # 如果运行还在进行中，结束它
        try:
            if run.info.status == "RUNNING":
                print("run_id:", run.info.run_id)
                client.set_terminated(run.info.run_id)
        except mlflow.exceptions.MissingConfigException:
            print(f"Skipping run with missing meta.yaml file: {run.info.run_id}")
    # check_run()


def check_run():
    active_run = mlflow.active_run()
    if active_run is not None:
        print(f"A run with UUID {active_run.info.run_id} is already active.")
    else:
        print("No active run.")


def get_model_epoch_and_accuracy(model_name, model_version):
    client = MlflowClient()
    model_version_details = client.get_model_version(model_name, model_version)
    run_id = model_version_details.run_id
    run = client.get_run(run_id)
    metrics = run.data.metrics
    epoch = metrics.get(f"{model_name}_epoch")
    accuracy = metrics.get(f"{model_name}_accuracy")
    if epoch is None:
        epoch = 0
    if accuracy is None:
        accuracy = -1
    return epoch, accuracy


def get_latest_model_version(model_name):
    client = MlflowClient()
    model_versions = client.get_latest_versions(model_name)
    if model_versions:
        return int(model_versions[0].version)
    else:
        print("model_versions not found")
        return None


class epoch_timer:
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.epoch = None

        self.speed = None

    def start(self, epoch):
        print_line('up')
        print("epoch {} start".format(epoch + 1))
        self.epoch = epoch
        self.start_time = time.time()
        print_line('down')

    def end(self):
        self.end_time = time.time()
        self.speed = 1 / (self.end_time - self.start_time) * 3600
        print_line('up')
        print("epoch {} using time: ".format(self.epoch + 1), self.end_time - self.start_time)
        print("speed: ", self.end_time - self.start_time, "s/epoch, ", 1 / ((self.end_time - self.start_time) / 3600),
              "epoch/hour")
        print_line('down')
        return self.speed


# ===========================
# ...........................
# absadsasdasd
# ...........................
# ===========================
def print_line(up_or_down, len=65):
    if up_or_down == 'up' or up_or_down == 0:
        print('=' * len)
        print('.' * len)
    elif up_or_down == 'down' or up_or_down == 1:
        print('.' * len)
        print('=' * len)
    else:
        print('Invalid input')
        raise ValueError
