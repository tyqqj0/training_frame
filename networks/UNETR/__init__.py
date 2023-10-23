# -*- CODING: UTF-8 -*-
# @time 2023/10/17 16:38
# @Author tyqqj
# @File __init__.py.py
# @
# @Aim

import argparse


def get_args():
    # 基础参数
    parser = argparse.ArgumentParser(description="UNETR segmentation pipeline")
    # 训练参数
    parser.add_argument(
        "--pretrained_model_name", default="UNETR_model_best_acc.pth", type=str, help="pretrained model name"
    )
    parser.add_argument("--predict_model_dir", default="./run/test/model_final.pt", type=str, help="predict model dir")
    parser.add_argument("--save_to_test", action="store_true", help="save to test directory")
    parser.add_argument("--test_mode", action="store_true", help="test mode")
    parser.add_argument("--amt", default=-1, type=int, help="data amount")
    parser.add_argument("--max_epochs", default=6000, type=int, help="max number of training epochs")
    parser.add_argument("--batch_size", default=6, type=int, help="number of batch size")
    parser.add_argument("--sw_batch_size", default=1, type=int, help="number of sliding window batch size")
    parser.add_argument("--optim_lr", default=1e-4, type=float, help="optimization learning rate")
    parser.add_argument("--optim_name", default="adamw", type=str, help="optimization algorithm")
    parser.add_argument("--reg_weight", default=1e-5, type=float, help="regularization weight")
    parser.add_argument("--momentum", default=0.99, type=float, help="momentum")
    parser.add_argument("--noamp", action="store_true", help="do NOT use amp for training")
    parser.add_argument("--val_every", default=50, type=int, help="validation frequency")
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
    parser.add_argument("--RandScaleIntensityd_prob", default=0.1, type=float,
                        help="RandScaleIntensityd aug probability")
    parser.add_argument("--RandShiftIntensityd_prob", default=0.1, type=float,
                        help="RandShiftIntensityd aug probability")
    parser.add_argument("--infer_overlap", default=0.5, type=float, help="sliding window inference overlap")
    parser.add_argument("--lrschedule", default="warmup_cosine", type=str, help="type of learning rate scheduler")
    parser.add_argument("--warmup_epochs", default=50, type=int, help="number of warmup epochs")
    parser.add_argument("--resume_ckpt", action="store_true", help="resume training from pretrained checkpoint")
    parser.add_argument("--resume_jit", action="store_true",
                        help="resume training from pretrained torchscript checkpoint")
    parser.add_argument("--smooth_dr", default=1e-6, type=float, help="constant added to dice denominator to avoid nan")
    parser.add_argument("--smooth_nr", default=0.0, type=float, help="constant added to dice numerator to avoid zero")
    args, _ = parser.parse_known_args()
    return args
# get_args()
