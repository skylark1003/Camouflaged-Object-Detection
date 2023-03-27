import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.parameter import Parameter

import numpy as np
import scipy.stats as st

# 这是一个使用 PyTorch 框架定义的神经网络模型
def gkern(kernlen=16, nsig=3):
    interval = (2*nsig+1.)/kernlen
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    return kernel


def min_max_norm(in_):
    max_ = in_.max(3)[0].max(2)[0].unsqueeze(2).unsqueeze(3).expand_as(in_)
    min_ = in_.min(3)[0].min(2)[0].unsqueeze(2).unsqueeze(3).expand_as(in_)
    in_ = in_ - min_
    return in_.div(max_-min_+1e-8)

# 整体关注模块
# 继承自 nn.Module，表示它是一个 PyTorch 的模块，可以被用于构建更大的神经网络模型。
class HA(nn.Module):
    # holistic attention module
    # 定义了一个二维高斯卷积核，该卷积核是用于对输入的注意力图像进行模糊化处理，以增强模型的鲁棒性和抗噪性能。
    # 该卷积核是由 gkern 函数生成的。
    def __init__(self):
        super(HA, self).__init__()
        gaussian_kernel = np.float32(gkern(31, 4))
        gaussian_kernel = gaussian_kernel[np.newaxis, np.newaxis, ...]
        self.gaussian_kernel = Parameter(torch.from_numpy(gaussian_kernel))

    # 定义了前向传播过程。
    # 输入的是注意力图像和待处理的特征图像。
    # 首先对注意力图像进行高斯模糊化处理，然后对模糊化后的注意力图像进行归一化处理，使其像素值在 [0, 1] 范围内。
    # 最后，将特征图像与归一化后的注意力图像逐元素相乘，以增强特征图像中与注意力图像对应区域的特征响应。
    def forward(self, attention, x):
        soft_attention = F.conv2d(attention, self.gaussian_kernel, padding=15)
        soft_attention = min_max_norm(soft_attention)
        x = torch.mul(x, soft_attention.max(attention))
        return x
