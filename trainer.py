# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import time

import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data.distributed
from torch.cuda.amp import GradScaler, autocast
from utils.utils import distributed_all_gather
from skimage.measure import label, regionprops
import mlflow

from monai.data import decollate_batch


def dice(x, y):
    intersect = np.sum(np.sum(np.sum(x * y)))
    y_sum = np.sum(np.sum(np.sum(y)))
    if y_sum == 0:
        return 0.0
    x_sum = np.sum(np.sum(np.sum(x)))
    return 2 * intersect / (x_sum + y_sum)


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = np.where(self.count > 0, self.sum / self.count, self.sum)


def train_epoch(model, loader, optimizer, scaler, epoch, loss_func, args, box=None):
    model.train()
    box.start_epoch(loader=loader, stage='train', epoch=epoch, use_vis=False)
    start_time = time.time()
    run_loss = AverageMeter()
    # see_loss = evl(loader, epoch=epoch)
    for idx, batch_data in enumerate(loader):
        if isinstance(batch_data, list):
            data, target = batch_data
        else:
            data, target = batch_data["image"], batch_data["label"]
        data, target = data.cuda(args.rank), target.cuda(args.rank)
        for param in model.parameters():
            param.grad = None
        with autocast(enabled=args.amp):  #
            logits = model(data)
            # print(logits.shape, target.shape)
            loss = loss_func(logits, target)  # 这里出现报错了
        if args.amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        if args.distributed:
            loss_list = distributed_all_gather([loss], out_numpy=True, is_valid=idx < loader.sampler.valid_length)
            run_loss.update(
                np.mean(np.mean(np.stack(loss_list, axis=0), axis=0), axis=0), n=args.batch_size * args.world_size
            )
        else:
            run_loss.update(loss.item(), n=args.batch_size)
        box.update_in_epoch(step=idx, out=logits, target=target, stage='train')
        # print(logits.shape, target.shape)
        # if not args.test:
        #     BOX.vis(epoch, args, data, logits, target)
        # see_loss.update(logits < 0, target)
        if args.rank == 0:
            print(
                "Epoch {}/{} {}/{}".format(epoch + 1, args.max_epochs, idx + 1, len(loader)),
                "loss: {:.4f}".format(run_loss.avg),
                "time {:.2f}s".format(time.time() - start_time),
                # "data: {}".format(data.shape)
            )
        start_time = time.time()
    for param in model.parameters():
        param.grad = None
    # 计算批次的平均值
    # len是idx的最大值，是数据数除以batch_size然后向上取整
    box.end_epoch()
    # see_loss.print_avr()
    return run_loss.avg


def val_epoch(model, loader, epoch, acc_func, args, model_inferer=None, post_label=None, post_pred=None, box=None):
    model.eval()
    start_time = time.time()
    # print('valing')
    # see_loss = evl(loader, epoch=epoch)
    box.start_epoch(loader=loader, stage='val', epoch=epoch)
    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            if isinstance(batch_data, list):
                data, target = batch_data
            else:
                data, target = batch_data["image"], batch_data["label"]
            data, target = data.cuda(args.rank), target.cuda(args.rank)
            with autocast(enabled=args.amp):
                if model_inferer is not None:  # TODO: 这里是干啥的
                    logits = model_inferer(data)
                else:
                    logits = model(data)
            if not logits.is_cuda:
                target = target.cpu()
            box.update_in_epoch(step=idx, out=logits, target=target, stage='val')
            # print(logits.shape, target.shape)
            # see_loss.update(logits < 0, target)
            # print("data.shape", data.shape)
            # print("logits.shape", logits.shape)
            # print("logits.shape", logits.shape)
            # print('here')
            val_labels_list = decollate_batch(target)
            # print('here')
            val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]  # 将数据onehot 应该是
            # print('here')
            val_outputs_list = decollate_batch(logits)
            val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]  # 将数据onehot 应该是
            # print(val_labels_convert[0].shape, val_output_convert[0].shape)
            # if args.test:
            # 可视化
            # BOX.vis(epoch, args, data, logits, target, add_text='val')

            acc = acc_func(y_pred=val_output_convert, y=val_labels_convert)
            acc = acc.cuda(args.rank)
            print('here')

            if args.distributed:
                acc_list = distributed_all_gather([acc], out_numpy=True, is_valid=idx < loader.sampler.valid_length)
                avg_acc = np.mean([np.nanmean(l) for l in acc_list])

            else:
                acc_list = acc.detach().cpu().numpy()
                avg_acc = np.mean([np.nanmean(l) for l in acc_list])

            if args.rank == 0:
                # see_loss.print_avr()
                print(
                    "Val {}/{} {}/{}".format(epoch + 1, args.max_epochs, idx, len(loader)),
                    "acc",
                    avg_acc,
                    "time {:.2f}s".format(time.time() - start_time),
                )
            start_time = time.time()
    # if args.save_to_test:
    #     save_ckpt(model, epoch, args)
    box.end_epoch()
    return avg_acc


def save_checkpoint(model, epoch, args, filename="model.pt", best_acc=0, optimizer=None, scheduler=None):
    state_dict = model.state_dict() if not args.distributed else model.module.state_dict()
    save_dict = {"epoch": epoch, "best_acc": best_acc, "state_dict": state_dict}
    if optimizer is not None:
        save_dict["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        save_dict["scheduler"] = scheduler.state_dict()
    filename = os.path.join(args.logdir, filename)
    torch.save(save_dict, filename)
    print("Saving checkpoint", filename)


def additional_matrics(model_inferer, loader, epoch):
    # 预测方法可以取
    # 找到第一个batch
    first_batch = []
    for first_batch in loader:
        break
    if isinstance(first_batch, list):
        data, target = first_batch
    else:
        data, target = first_batch["image"], first_batch["label"]
    data, target = data.cuda(0), target.cuda(0)
    with torch.no_grad():
        logits = model_inferer(data)

    # 如果输出是二维，取1, [batch_size, 2, 96, 96, 96]
    if logits.shape[1] == 2:
        logits = logits[:, 1]
    if len(logits.shape) > 3:
        logits = logits.squeeze(0)
        logits = logits.squeeze(0)
    # 不论logits是不是概率
    logits = logits > 0
    max_volume = calculate_max_component(logits.cpu().numpy())
    mlflow.log_metric("max_volume", max_volume, step=epoch)


def run_training(
        model,
        train_loader,
        val_loader,
        optimizer,
        loss_func,
        acc_func,
        args,
        model_inferer=None,
        scheduler=None,
        start_epoch=0,
        post_label=None,
        post_pred=None,
        box=None,
):
    if args.amp:
        scaler = GradScaler()
    val_acc_max = 0.0
    # print(args.mode)

    for epoch in range(start_epoch, args.max_epochs):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
            torch.distributed.barrier()
        print(args.rank, time.ctime(), "Epoch:", epoch + 1, "/", args.max_epochs)
        epoch_time = time.time()
        # BOX

        train_loss = train_epoch(
            model, train_loader, optimizer, scaler=scaler, epoch=epoch, loss_func=loss_func, args=args, box=box
        )
        if args.rank == 0:
            print(
                "Final training  {}/{}".format(epoch + 1, args.max_epochs - 1),
                "loss: {:.4f}".format(train_loss),
                "time {:.2f}s".format(time.time() - epoch_time),
            )
        if (epoch + 1) % args.val_every == 0:
            if args.distributed:
                torch.distributed.barrier()

            epoch_time = time.time()
            print("val")
            val_avg_acc = val_epoch(
                model,
                val_loader,
                epoch=epoch,
                acc_func=acc_func,
                model_inferer=model_inferer,
                args=args,
                post_label=post_label,
                post_pred=post_pred,
                box=box
            )
            if args.rank == 0:
                print(
                    "Final validation  {}/{}".format(epoch + 1, args.max_epochs - 1),
                    "acc",
                    val_avg_acc,
                    "time {:.2f}s".format(time.time() - epoch_time),
                )
            if val_avg_acc > val_acc_max:
                val_acc_max = val_avg_acc
        additional_matrics(model_inferer, val_loader, epoch)  # TODO: 这个写的不好看
        box.visualizes(model, val_loader)
        box.save_model(model, epoch)
        if scheduler is not None:
            scheduler.step()

    print("Training Finished !, Best Accuracy: ", val_acc_max)

    return val_acc_max


def calculate_max_component(image_3d, connectivity=3):
    print("calculating max component")
    plt.imshow(image_3d[64])
    plt.show()
    # connectivity: 是指连通组件的连接方式，可以是1,2,3,4,6
    # 使用 `label` 函数来找到并标记所有的连通组件
    start_time = time.time()
    labels_3d = label(image_3d, connectivity=connectivity)
    # 使用 `regionprops` 来获取每个连通组件的属性
    props_3d = regionprops(labels_3d)

    # 找到最大的连通组件
    max_volume = 0
    for prop in props_3d:
        if prop.area > max_volume:
            max_volume = prop.area

    print("max_volume", max_volume)
    print("calculate_max_component time", time.time() - start_time)
    return max_volume
