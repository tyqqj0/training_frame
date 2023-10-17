# -*- CODING: UTF-8 -*-
# @time 2023/9/21 12:30
# @Author tyqqj
# @File vis.py
# @
# @Aim
import numpy
from monai.data import decollate_batch
from monai.visualize import blend_images, matshow3d, plot_2d_or_3d_image
from torch.utils.tensorboard import SummaryWriter
import torch
import os
import SimpleITK as sitk
from torchvision.utils import save_image

from monai.visualize import GradCAM


# tensorboard --logdir=./runs/test/vis


def vis_2d(path, epoch, image, outputs, label=None, add_text='', rank=0):
    # 传入当前批次，参数，图像，输出，标签
    # 判断是否应该可视化
    # print("vis")

    # 如果数据不是list，使用decollate_batch将数据转换为list
    if not isinstance(image, list):
        image = decollate_batch(image)
    if not isinstance(outputs, list):
        outputs = decollate_batch(outputs)
    if not isinstance(label, list):
        if label is not None:
            label = decollate_batch(label)
    n = 64
    img, out, out_class, lb = image[0][0][n], outputs[0][1][n], outputs[0][1][n] > 0, label[0][0][n]
    # 缩放范围到0-1
    img = (img - img.min()) / (img.max() - img.min())
    out = (out - out.min()) / (out.max() - out.min())
    # out_class = (out_class - out_class.min()) / (out_class.max() - out_class.min())
    lb = (lb - lb.min()) / (lb.max() - lb.min())

    img = img.unsqueeze(0)
    out = out.unsqueeze(0)
    out_class = out_class.unsqueeze(0)
    lb = lb.unsqueeze(0)

    # 添加图像文本提示

    if 0:
        print('img.shape', img.shape)
        print('out.shape', out.shape)
        print('out_class.shape', out_class.shape)
        print('lb.shape', lb.shape)
        # print('combined.shape', combined.shape)

    # 将 PyTorch tensor 转换为 numpy array
    img_np = img.squeeze().cpu().numpy()
    out_np = out.squeeze().cpu().numpy()
    out_class_np = out_class.squeeze().cpu().numpy()
    lb_np = lb.squeeze().cpu().numpy()

    # 创建一个 1x4 的子图，每个图像都有自己的小标题
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    axs[0].imshow(img_np, cmap='gray')
    axs[0].set_title('Image')
    axs[1].imshow(out_np, cmap='gray')
    axs[1].set_title('Output Probability')
    axs[2].imshow(out_class_np, cmap='gray')
    axs[2].set_title('Output Class')
    axs[3].imshow(lb_np, cmap='gray')
    axs[3].set_title('Label')

    # 保存整个子图
    plt.savefig(os.path.join(path, f'combined{add_text}_{epoch}.png'))
    plt.close()


def vis_2d_tensorboard(path, epoch, image, outputs, label=None, add_text='', rank=0):
    # 传入当前批次，参数，图像，输出，标签
    # 判断是否应该可视化
    # print("vis")

    # 如果数据不是list，使用decollate_batch将数据转换为list
    if not isinstance(image, list):
        image = decollate_batch(image)
    if not isinstance(outputs, list):
        outputs = decollate_batch(outputs)
    if not isinstance(label, list):
        if label is not None:
            label = decollate_batch(label)
    n = 64
    img, out, lb = image[0][0][n], outputs[0][1][n], label[0][0][n]
    img = img.unsqueeze(0)
    out = out.unsqueeze(0)
    lb = lb.unsqueeze(0)

    if 0:
        print('img.shape', img.shape)
        print('out.shape', out.shape)
        print('lb.shape', lb.shape)
        # print('combined.shape', combined.shape)

    # 将图像之间画上分割线
    line = torch.ones(1, img.shape[2], 1).cuda(rank)
    combined = torch.cat((img, line, out, line, lb), 2)
    # Now we have the original image, the model's prediction, and the Grad-CAM result
    # We can add them to TensorBoard

    tb_dir = os.path.join(path)
    if not os.path.exists(tb_dir):
        os.makedirs(tb_dir)

    writer = SummaryWriter(log_dir=tb_dir)
    # Concatenate along the width dimension

    # Now we can add the combined image to TensorBoard
    writer.add_image('combined' + add_text, combined, epoch)

    # Don't forget to close the writer when you're done
    writer.close()


def vis_mha(path, epoch, image, outputs, label=None, add_text='', rank=0):
    # 传入当前批次，参数，图像，输出，标签
    # 判断是否应该可视化
    # print("vis3d")

    # 如果数据不是list，使用decollate_batch将数据转换为list
    if not isinstance(image, list):
        image = decollate_batch(image)
    if not isinstance(outputs, list):
        outputs = decollate_batch(outputs)
    if not isinstance(label, list):
        if label is not None:
            label = decollate_batch(label)

    tb_dir = os.path.join(path)
    img, out, lb = image[0][0].cpu().numpy(), outputs[0][1].cpu().numpy(), label[0][0].cpu().numpy()

    img = sitk.GetImageFromArray(img.astype(numpy.float64))
    out = sitk.GetImageFromArray(out.astype(numpy.float64))
    lb = sitk.GetImageFromArray(lb.astype(numpy.float64))

    file_name = os.path.join(tb_dir, add_text + str(epoch))
    # 文件夹名: y/m/d + epoch
    # 创建文件夹
    if not os.path.exists(file_name):
        os.makedirs(file_name)

    file_name_img = file_name + '/' + str(epoch) + '_image.mha'
    file_name_out = file_name + '/' + str(epoch) + '_output.mha'
    file_name_lb = file_name + '/' + str(epoch) + '_label.mha'

    sitk.WriteImage(img, file_name_img)
    sitk.WriteImage(out, file_name_out)
    sitk.WriteImage(lb, file_name_lb)


def vis_temp():
    pass


def vis_cam(cam, val_loader, args):
    # Compute the class activation map
    result = cam(val_loader)

    # Create a Tensorboard writer
    tb_dir = os.path.join(args.logdir, "vis")
    writer = SummaryWriter(log_dir=tb_dir)

    # Visualize the class activation map
    plot_2d_or_3d_image(result, step=0, writer=writer, frame_dim=-1)

    # Close the writer when you're done
    writer.close()