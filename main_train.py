
import argparse
import os
from functools import partial

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.parallel
import torch.utils.data.distributed
from networks.unetr import UNETR
from optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from trainer import run_training
from utils.data_utils import get_loader
from utils.utils import save_ckpt
from utils.box import box

from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss, DiceLoss
from monai.metrics import DiceMetric
from monai.transforms import Activations, AsDiscrete, Compose
from monai.utils.enums import MetricReduction
from monai import __version__

# 基础参数
parser = argparse.ArgumentParser(description="UNETR segmentation pipeline")
# mlflow 基础参数
mlflow_parser = parser.add_argument_group('mlflow')
parser.add_argument("-n", "--exp_name", type=str, default=None, help="experiment name, ***must be set***")
parser.add_argument("-i", "--run_id", type=str, default=None,
                    help="run id, ***must be set when test or is_continue, only for continue training***")
parser.add_argument("-c", "--is_continue", action="store_true", help="continue training")
parser.add_argument("--train", action="store_true", help="train mode")
parser.add_argument("--test", action="store_true", help="test mode")
parser.add_argument("--new_run_name", type=str, default=None, help="new run name")
parser.add_argument("--log_dir", type=str, default="./runs", help="log dir")
# parser.add_argument("--artifact_dir", type=str, default="./artifacts", help="artifact dir")
parser.add_argument("--tag_id", type=str, default=None, help="tag id, ***commanded to set***")
parser.add_argument("--log_frq", type=int, default=10, help="log frequency")
parser.add_argument("--save_frq", type=int, default=10, help="save frequency, disabled in test and val")

parser.add_argument("--user_name", type=str, default="tyqqj", help="user name")

parser.add_argument("--vis_3d", action="store_true", help="visualize 3d images")
parser.add_argument("--vis_2d", action="store_true", help="visualize 2d images")
# 覆盖保存显示, 控制可视化是保留最新还是保留每个epoch
parser.add_argument("--vis_2d_cover", action="store_true", help="visualize 2d images")
parser.add_argument("--vis_3d_cover", action="store_true", help="visualize 3d images")


run_parser = parser.add_argument_group('run')
parser.add_argument("--checkpoint", default=None, help="start training from saved checkpoint")
# parser.add_argument("--vis_dir", default="./runs/", type=str, help="dataset directory")
# parser.add_argument("--logdir", default="test", type=str, help="directory to save the tensorboard logs")
# parser.add_argument("--test", action="store_true", help="test mode")
# loger参数
# parser.add_argument("--box_root", default="./run", type=str, help="training box root")
# parser.add_argument("--vis", action="store_true", help="visualize training")
# parser.add_argument("--vis3d", action="store_true", help="visualize training")

# 数据位置参数
parser.add_argument(
    "--pretrained_dir", default="./pretrained_models/", type=str, help="pretrained checkpoint directory"
)
parser.add_argument("--data_dir", default="D:/zhangchaoran/miccai_achieve/data/train/", type=str,
                    help="dataset directory")
parser.add_argument("--val_dir", default="D:/zhangchaoran/miccai_achieve/data/test", type=str,
                    help="dataset directory")
parser.add_argument("--test_data_dir", default="D:/zhangchaoran/miccai_achieve/data/train/", type=str,
                    help="dataset directory")
parser.add_argument("--json_list", default="dataset_0.json", type=str, help="dataset json file")

# 训练参数
parser.add_argument(
    "--pretrained_model_name", default="UNETR_model_best_acc.pth", type=str, help="pretrained model name"
)
parser.add_argument("--predict_model_dir", default="./run/test/model_final.pt", type=str, help="predict model dir")
parser.add_argument("--save_to_test", action="store_true", help="save to test directory")
parser.add_argument("--test_mode", action="store_true", help="test mode")

parser.add_argument("--amt", default=-1, type=int, help="data amount")
parser.add_argument("--save_checkpoint", action="store_true", help="save checkpoint during training")
parser.add_argument("--max_epochs", default=2000, type=int, help="max number of training epochs")
parser.add_argument("--batch_size", default=6, type=int, help="number of batch size")
parser.add_argument("--sw_batch_size", default=1, type=int, help="number of sliding window batch size")
parser.add_argument("--optim_lr", default=1e-4, type=float, help="optimization learning rate")
parser.add_argument("--optim_name", default="adamw", type=str, help="optimization algorithm")
parser.add_argument("--reg_weight", default=1e-5, type=float, help="regularization weight")
parser.add_argument("--momentum", default=0.99, type=float, help="momentum")
parser.add_argument("--noamp", action="store_true", help="do NOT use amp for training")
parser.add_argument("--val_every", default=50, type=int, help="validation frequency")
parser.add_argument("--vis_every", default=50, type=int, help="visualize frequency")
parser.add_argument("--vis3d_every", default=5, type=int, help="visualize frequency")
parser.add_argument("--distributed", action="store_true", help="start distributed training")
parser.add_argument("--world_size", default=1, type=int, help="number of nodes for distributed training")
parser.add_argument("--rank", default=0, type=int, help="node rank for distributed training")
parser.add_argument("--dist-url", default="tcp://127.0.0.1:23456", type=str, help="distributed url")
parser.add_argument("--dist-backend", default="nccl", type=str, help="distributed backend")
parser.add_argument("--workers", default=1, type=int, help="number of workers")
parser.add_argument("--model_name", default="unetr", type=str, help="model name")
parser.add_argument("--pos_embed", default="perceptron", type=str, help="type of position embedding")
parser.add_argument("--norm_name", default="instance", type=str, help="normalization layer type in decoder")
parser.add_argument("--num_heads", default=12, type=int, help="number of attention heads in ViT encoder")
parser.add_argument("--mlp_dim", default=3072, type=int, help="mlp dimention in ViT encoder")
parser.add_argument("--hidden_size", default=768, type=int, help="hidden size dimention in ViT encoder")
parser.add_argument("--feature_size", default=16, type=int, help="feature size dimention")
parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
parser.add_argument("--out_channels", default=2, type=int, help="number of output channels")  #
parser.add_argument("--res_block", action="store_true", help="use residual blocks")
parser.add_argument("--conv_block", action="store_true", help="use conv blocks")
parser.add_argument("--use_normal_dataset", action="store_true", help="use monai Dataset class")
parser.add_argument("--a_min", default=-175.0, type=float, help="a_min in ScaleIntensityRanged")
parser.add_argument("--a_max", default=250.0, type=float, help="a_max in ScaleIntensityRanged")
parser.add_argument("--b_min", default=0.0, type=float, help="b_min in ScaleIntensityRanged")
parser.add_argument("--b_max", default=1.0, type=float, help="b_max in ScaleIntensityRanged")
parser.add_argument("--space_x", default=1, type=float, help="spacing in x direction")
parser.add_argument("--space_y", default=1, type=float, help="spacing in y direction")
parser.add_argument("--space_z", default=1, type=float, help="spacing in z direction")
parser.add_argument("--roi_x", default=96, type=int, help="roi size in x direction")
parser.add_argument("--roi_y", default=96, type=int, help="roi size in y direction")
parser.add_argument("--roi_z", default=96, type=int, help="roi size in z direction")
parser.add_argument("--dropout_rate", default=0.0, type=float, help="dropout rate")
parser.add_argument("--RandFlipd_prob", default=0.2, type=float, help="RandFlipd aug probability")
parser.add_argument("--RandRotate90d_prob", default=0.2, type=float, help="RandRotate90d aug probability")
parser.add_argument("--RandScaleIntensityd_prob", default=0.1, type=float, help="RandScaleIntensityd aug probability")
parser.add_argument("--RandShiftIntensityd_prob", default=0.1, type=float, help="RandShiftIntensityd aug probability")
parser.add_argument("--infer_overlap", default=0.5, type=float, help="sliding window inference overlap")
parser.add_argument("--lrschedule", default="warmup_cosine", type=str, help="type of learning rate scheduler")
parser.add_argument("--warmup_epochs", default=50, type=int, help="number of warmup epochs")
parser.add_argument("--resume_ckpt", action="store_true", help="resume training from pretrained checkpoint")
parser.add_argument("--resume_jit", action="store_true", help="resume training from pretrained torchscript checkpoint")
parser.add_argument("--smooth_dr", default=1e-6, type=float, help="constant added to dice denominator to avoid nan")
parser.add_argument("--smooth_nr", default=0.0, type=float, help="constant added to dice numerator to avoid zero")


def main():
    '''
    训练定义
    :return:
    '''
    #
    args = parser.parse_args()
    args.amp = not args.noamp
    args.logdir = "./runs/" + args.logdir
    if args.save_to_test:
        args.max_epoch = 1
    if args.distributed:
        args.ngpus_per_node = torch.cuda.device_count()
        print("Found total gpus", args.ngpus_per_node)
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args,))
    else:
        main_worker(gpu=0, args=args)


def main_worker(gpu, args):
    logrbox = box(args)
    if args.distributed:
        torch.multiprocessing.set_start_method("fork", force=True)
    np.set_printoptions(formatter={"float": "{: 0.3f}".format}, suppress=True)
    args.gpu = gpu
    if args.distributed:
        args.rank = args.rank * args.ngpus_per_node + gpu
        dist.init_process_group(
            backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank
        )
    torch.cuda.set_device(args.gpu)
    torch.backends.cudnn.benchmark = True
    args.test_mode = False  # 没有testmode实际上
    loader = get_loader(args)  # 此处的loader是一个列表，列表中的每个元素是一个DataLoader对象，每个DataLoader对象中包含了训练集和验证集的数据
    print(args.rank, " gpu", args.gpu)
    if args.rank == 0:
        print("Batch size is:", args.batch_size, "epochs", args.max_epochs)
    inf_size = [args.roi_x, args.roi_y, args.roi_z]
    pretrained_dir = args.pretrained_dir
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

        if args.resume_ckpt:
            model_dict = torch.load(os.path.join(pretrained_dir, args.pretrained_model_name))
            model.load_state_dict(model_dict)
            print("Use pretrained weights")

        if args.resume_jit:
            if not args.noamp:
                print("Training from pre-trained checkpoint does not support AMP\nAMP is disabled.")
                args.amp = args.noamp
            model = torch.jit.load(os.path.join(pretrained_dir, args.pretrained_model_name))
    else:
        raise ValueError("Unsupported model " + str(args.model_name))

    dice_loss = DiceCELoss(
        to_onehot_y=args.out_channels, softmax=True, squared_pred=True, smooth_nr=args.smooth_nr,
        smooth_dr=args.smooth_dr
    )
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

    best_acc = 0
    start_epoch = 0

    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        from collections import OrderedDict

        new_state_dict = OrderedDict()
        for k, v in checkpoint["state_dict"].items():
            new_state_dict[k.replace("backbone.", "")] = v
        model.load_state_dict(new_state_dict, strict=False)
        if "epoch" in checkpoint:
            start_epoch = checkpoint["epoch"]
        if "best_acc" in checkpoint:
            best_acc = checkpoint["best_acc"]
        print("=> loaded checkpoint '{}' (epoch {}) (bestacc {})".format(args.checkpoint, start_epoch, best_acc))

    model.cuda(args.gpu)

    if args.distributed:
        torch.cuda.set_device(args.gpu)
        if args.norm_name == "batch":
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model.cuda(args.gpu)
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], output_device=args.gpu, find_unused_parameters=True
        )
    if args.optim_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.optim_lr, weight_decay=args.reg_weight)
    elif args.optim_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.optim_lr, weight_decay=args.reg_weight)
    elif args.optim_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.optim_lr, momentum=args.momentum, nesterov=True, weight_decay=args.reg_weight
        )
    else:
        raise ValueError("Unsupported Optimization Procedure: " + str(args.optim_name))

    if args.lrschedule == "warmup_cosine":
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer, warmup_epochs=args.warmup_epochs, max_epochs=args.max_epochs
        )
    elif args.lrschedule == "cosine_anneal":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs)
        if args.checkpoint is not None:
            scheduler.step(epoch=start_epoch)
    else:
        scheduler = None

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
        box = logrbox
    )
    if args.save_to_test:
        model.save(args)
    return accuracy


os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
CUDA_LAUNCH_BLOCKING = 1
torch.backends.cudnn.enable = True
torch.backends.cudnn.benchmark = True
TORCH_USE_CUDA_DSA = 1

if __name__ == "__main__":
    print(torch.__version__)
    print(__version__)
    print(torch.cuda.is_available())
    main()
