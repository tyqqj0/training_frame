# -*- CODING: UTF-8 -*-
# @time 2023/12/1 19:42
# @Author tyqqj
# @File NGCM.py
# @
# @Aim
import time
import warnings

import numpy as np

from .text import print_line, text_in_box


# from .box import print_line, text_in_box


def count_layer_corr(layer_params):
    # layer_params: (output, conv, batchsize)
    # 计算所有核直接梯度的拨动相关性(output, output)
    cov_layer_params = np.zeros((layer_params.shape[0], layer_params.shape[0]))
    for i in range(layer_params.shape[0]):
        for j in range(0, i):
            cov_layer_params[j, i] = count_corr_mean(np.abs(np.corrcoef(layer_params[j], layer_params[i])))
            # cov_layer_params[i, j] = np.cov(layer_params[i], layer_params[j]).abs().sum()
    return cov_layer_params


def count_corr_mean(corrcoef):
    # 计算相关性非对角线的均值. 因相关性对称，所以计算对角线上的一半
    corrcoefmn = np.triu(corrcoef, 1)
    corrcoefmn = corrcoefmn[corrcoefmn != 0]

    return corrcoefmn.mean()


class GradientStats:
    def __init__(self, model, mode='mean'):
        print_line('up')
        self.loc = 2  # 对unet暂时设成2，因为前面有个model, 要改的
        self.model = model
        # 如果model的参数是model(monai中这么实现的),就将model变为model.model
        # named_params = list(self.model.named_parameters())
        # if len(named_params) == 1 and named_params[0][0] == "model":
        #     self.model = self.model.model
        # self.grads = {name: [] for name, _ in self.model.named_parameters()}
        named_params = list(model.named_parameters())
        if len(named_params) == 1 and named_params[0][0] == "model":
            print("computing model.model")
            model = model.model
            self.loc = self.loc + 1
        self.grads = {name: [] for name, _ in model.named_parameters()}
        self.hooks = []
        self.mode = mode

        for name, param in self.model.named_parameters():
            print("register hook for ", name)
            hook = param.register_hook(self.save_grad(name))
            self.hooks.append(hook)

        print(text_in_box("computing gradient stability"))
        print("self.grads name: ", self.grads.keys())
        print("self.hooks name: ", self.hooks)
        print_line('down')

    def get_model_layers(self, n=None, end_with=None, ignore=None):
        # print("model static keys", self.model.state_dict().keys())
        if n is None:
            n = ["conv", "adn", "residual", "up", "down", "final"]
        if n == "all":
            # 拼接所有的weight和bias，并返回所有的模块名称
            return {name: params for name, params in self.model.state_dict().items() if
                    'weight' in name or 'bias' in name}
        if end_with is None:
            end_with = ["weight", "bias"]
        if ignore is None:
            ignore = ["adn"]
        if isinstance(n, str):
            # 将字符串转换为列表
            n = [n]
        elif isinstance(n, int):
            # 返回指定数量的模块名称
            return set(['.'.join(name.split('.')[0:n]) for name in self.model.state_dict().keys()])
        elif isinstance(n, list):
            # 假设 n 是一个包含最小跟踪单位的列表
            track_units = n
            unique_names = set()
            for name in self.model.state_dict().keys():
                if any(unit in name for unit in ignore):
                    continue
                for unit in track_units:
                    if end_with is None:
                        if unit in name:
                            # 找到单位在名称中的位置，然后取出前面的部分
                            pos = name.index(unit)
                            unique_names.add(name[:pos + len(unit)])
                    else:
                        if unit in name and name.endswith(tuple(end_with)):
                            unique_names.add(name)

                        break
            # 如果返回为空，则警告
            if not unique_names:
                warnings.warn("No layers found with the given units: {}".format(track_units))
            return unique_names
        else:
            raise ValueError(f"Unsupported type of n: {type(n)}")

    def save_grad(self, name):
        def hook(grad):
            # print("update grad")
            self.grads[name].append(grad.clone())

        return hook

    def compute_unstable_perlayer(self, n=None, end_with=None):
        time_cont = simple_timer()
        time_cont.start()
        # 计算每个参数的梯度不稳定性
        grad_instability = {name: [g.detach().cpu().numpy() for g in grads]
                            for name, grads in self.grads.items()}

        # 计算每个层的梯度不稳定性
        layer_instability = {}
        for layer in self.get_model_layers(n, end_with):
            layer_values = np.array([val for key, val in grad_instability.items() if key.startswith(layer)])

            # 处理数据维度(1, 3, 256, 128, 3, 3, 3)
            # (多余维， batch_size, output_channel, input_channel, x, y, z)
            # 获取原始形状
            original_shape = layer_values.shape

            # 计算新的形状: 前三个维度的乘积, 后四个维度的乘积
            new_shape = (np.prod(original_shape[:2]), np.prod(original_shape[2:3]), np.prod(original_shape[3:]))
            if new_shape[1] > 128:
                print("continue")
                layer_instability[layer] = 0
                continue
            else:
                print(new_shape[1])
            # 重塑数组
            layer_values = layer_values.reshape(new_shape)
            layer_values = np.transpose(layer_values, (1, 2, 0))
            print("layer {} value shape: {}".format(layer, np.array(layer_values).shape))

            # 获取模型参数

            # 计算协方差矩阵
            # layer_instability["stb_mean_" + layer] = np.mean(layer_values) if layer_values.size != 0 else 0

            # 获取模型参数的矩阵
            layer_params = np.array(
                [param.data.cpu().numpy() for name, param in self.model.named_parameters() if layer in name])
            # for name, param in self.model.named_parameters():
            #     if layer in name:
            new_shape_params = (
                np.prod(layer_params.shape[:2]), np.prod(layer_params.shape[2:]))
            layer_params = layer_params.reshape(new_shape_params)
            # print("layer {} param shape: {}".format(layer, layer_params.shape))

            # 梯度的(output, conv, batchsize)->(output, conv * batchsize)

            layer_values = layer_values.reshape(layer_values.shape[0], -1)

            # 对参数和梯度分别求协方差
            cov_layer_values = count_layer_corr(layer_values)
            # cov_layer_params = np.cov(layer_params)
            cov_layer_params = 1 - np.abs(np.corrcoef(layer_params))
            # 两个相关性按位乘法
            layer_instability[layer] = count_corr_mean(cov_layer_values * cov_layer_params)
            layer_instability['values_cor_' + layer] = count_corr_mean(cov_layer_values)
            layer_instability['params_cor_' + layer] = count_corr_mean(cov_layer_params)
            print(layer_instability[layer])
            # 清空grads
            self.clear_grad()

            print(time_cont)

        return layer_instability

    def clear_grad(self):
        self.grads = {name: [] for name, _ in self.model.named_parameters()}

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()


class simple_timer():
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.speed = None

    def start(self):
        self.start_time = time.time()
        print("timer start counte")

    def end(self):
        self.end_time = time.time()
        self.speed = 1 / (self.end_time - self.start_time) * 3600
        return self.speed

    def __str__(self):
        self.end()
        run_time = self.end_time - self.start_time
        timett = []
        timett.append("Run Time: " + str(run_time))
        timett.append("Speed: " + str(self.speed))
        self.start()
        return "\n".join(timett)
        return timett
