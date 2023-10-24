from functools import partial

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.parallel
import torch.utils.data.distributed

import networks.UNETR
import utils.arg as arg
from networks.UNETR.unetr import UNETR
from optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from trainer import run_training
from utils.data_loader.data_utils import get_loader
from utils.BOX import box

from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete
from monai.utils.enums import MetricReduction
from monai import __version__


########################################################################################################


def main():
    '''
    训练定义
    :return:
    '''
    # 获取模型的参数
    args = networks.UNETR.get_args()
    # utils.arg.parser.save_parser_to_json(parser, "./UNTER.json")
    # utils.arg.parser.save_parser_to_json(box.parser_cfg_loader()[1], "./box.json")
    # return
    logrbox = box.box()
    logrbox.check_args()
    # print(logrbox.args)
    # _, parser = box.parser_cfg_loader()
    # utils.arg.parser.save_parser_to_json(parser, './utils/BOX/cfg/train1.json')
    # 将框架参数同步到模型
    # return
    args.val_every = logrbox.get_frq()
    args.amp = not args.noamp

    if args.distributed:
        args.ngpus_per_node = torch.cuda.device_count()
        print("Found total gpus", args.ngpus_per_node)
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args,))
    else:
        main_worker(args=args, logrbox=logrbox)


def main_worker(args, logrbox):
    # 设置cuda
    set_cuda(args)

    # 获取数据读取器
    # TODO: 重写数据读取器
    loader = get_loader()  # 可以指定数据配置

    # 获取模型并读取
    model, start_epoch = get_model(args, logrbox)

    # 设置模型的推理器
    # TODO: 这个不好看写法，改成自动的更好
    inf_size = [args.roi_x, args.roi_y, args.roi_z]
    logrbox.set_model_inferer(model)

    # 设置损失函数
    dice_loss = DiceCELoss(
        to_onehot_y=args.out_channels, softmax=True, squared_pred=True, smooth_nr=args.smooth_nr,
        smooth_dr=args.smooth_dr
    )

    # 设置小工具
    post_label = AsDiscrete(to_onehot=args.out_channels, n_classes=args.out_channels)  # 将数据onehot 应该是
    post_pred = AsDiscrete(argmax=True, to_onehot=args.out_channels, n_classes=args.out_channels)
    dice_acc = DiceMetric(include_background=True, reduction=MetricReduction.MEAN, get_not_nans=True)
    # loss_f =

    model_inferer = partial(
        sliding_window_inference,
        roi_size=inf_size,
        sw_batch_size=args.sw_batch_size,
        predictor=model,
        overlap=args.infer_overlap,
    )

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total parameters count", pytorch_total_params)

    # 设置优化器
    optimizer = set_optim(model, args.optim_name, args.optim_lr, args.reg_weight, args.momentum)

    # 设置学习率调整器
    scheduler = set_lrschedule(optimizer, start_epoch, args.max_epochs, args.lrschedule, args.warmup_epochs)

    with logrbox as run:
        accuracy = run_training(  # 训练
            model=model,
            train_loader=loader[0],
            val_loader=loader[1],
            optimizer=optimizer,
            loss_func=dice_loss,
            acc_func=dice_acc,
            args=args,
            model_inferer=model_inferer,
            scheduler=scheduler,
            start_epoch=start_epoch,
            post_label=post_label,
            post_pred=post_pred,
            box=logrbox
        )
        if args.save_to_test:
            model.save(args)
    return accuracy


def set_lrschedule(optimizer, start_epoch, max_epochs, lrschedule='warmup_cosine', warmup_epochs=50):
    if lrschedule == "warmup_cosine":
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer, warmup_epochs=warmup_epochs, max_epochs=max_epochs
        )
    elif lrschedule == "cosine_anneal":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
        if start_epoch != 0:
            scheduler.step(epoch=start_epoch)
    else:
        scheduler = None
    return scheduler


def set_optim(model, optim_name="adamw", optim_lr=1e-4, reg_weight=1e-5, momentum=0.99):
    if optim_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=optim_lr, weight_decay=reg_weight)
    elif optim_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=optim_lr, weight_decay=reg_weight)
    elif optim_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=optim_lr, momentum=momentum, nesterov=True,
            weight_decay=reg_weight
        )
    else:
        raise ValueError("Unsupported Optimization Procedure: " + str(optim_name))
    return optimizer


def get_model(args, logrbox):
    if (args.model_name is None) or args.model_name == "unetr":
        model = UNETR(
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            img_size=(args.roi_x, args.roi_y, args.roi_z),
            feature_size=args.feature_size,
            hidden_size=args.hidden_size,
            mlp_dim=args.mlp_dim,
            num_heads=args.num_heads,
            pos_embed=args.pos_embed,
            norm_name=args.norm_name,
            conv_block=True,
            res_block=True,
            dropout_rate=args.dropout_rate,
        )
    else:
        raise ValueError("Unsupported model " + str(args.model_name))
    model, start_epoch, best_acc = logrbox.load_model(model)

    model.cuda(args.gpu)

    if args.distributed:
        torch.cuda.set_device(args.gpu)
        if args.norm_name == "batch":
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model.cuda(args.gpu)
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], output_device=args.gpu, find_unused_parameters=True
        )

    return model, start_epoch


def set_cuda(args):
    if args.distributed:
        torch.multiprocessing.set_start_method("fork", force=True)
    np.set_printoptions(formatter={"float": "{: 0.3f}".format}, suppress=True)  # 设置打印精度
    args.gpu = 0
    if args.distributed:
        args.rank = args.rank * args.ngpus_per_node + 0
        dist.init_process_group(
            backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank
        )
    torch.cuda.set_device(args.gpu)
    torch.backends.cudnn.benchmark = True
    print(args.rank, " gpu", args.gpu)
    if args.rank == 0:
        print("Batch size is:", args.batch_size, "epochs", args.max_epochs)


# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# CUDA_LAUNCH_BLOCKING = 1
# torch.backends.cudnn.enable = True
# torch.backends.cudnn.benchmark = True
# TORCH_USE_CUDA_DSA = 1

if __name__ == "__main__":
    print(torch.__version__)
    print(__version__)
    print(torch.cuda.is_available())
    main()
