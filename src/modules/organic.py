import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import numpy as np
from torch.nn.modules.conv import _ConvNd
from utils import ntuple


def make_indices(num_channels, groups, sharing_rates, weight_size):
    shared_size = round(num_channels / groups * sharing_rates)
    nonshared_size = weight_size - shared_size
    shared_indices = torch.arange(shared_size).expand(groups, shared_size)
    nonshared_indices = torch.arange(shared_size, shared_size + nonshared_size).view(groups, -1)
    indices = torch.cat([shared_indices, nonshared_indices], dim=1).view(-1)
    print(indices)
    return indices


class _oConvNd(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, transposed, output_padding,
                 groups, sharing_rates, bias):
        super(_oConvNd, self).__init__()
        _nutple = ntuple(2)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        groups, sharing_rates = _nutple(groups), _nutple(sharing_rates)
        self.groups = _nutple(groups)
        self.sharing_rates = _nutple(sharing_rates)
        self.input_size = self.in_channels - (self.groups[0] - 1) * round(
            self.in_channels / self.groups[0] * self.sharing_rates[0])
        self.output_size = self.out_channels - (self.groups[1] - 1) * round(
            self.in_channels / self.groups[1] * self.sharing_rates[1])
        self.indices = (make_indices(self.in_channels, self.groups[0], self.sharing_rates[0], self.input_size),
                        make_indices(self.out_channels, self.groups[1], self.sharing_rates[1], self.output_size))
        if transposed:
            self.weight = nn.Parameter(torch.Tensor(self.input_size, self.output_size, *kernel_size))
        else:
            self.weight = nn.Parameter(torch.Tensor(self.output_size, self.input_size, *kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.output_size))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != (1, 1):
            s += ', groups={groups}'
        if self.sharing_rates != (0, 0):
            s += ', sharing_rates={sharing_rates}'
        if self.bias is None:
            s += ', bias=False'
        return s.format(**self.__dict__)


class oConv2d(_oConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 sharing_rates=0, bias=True):
        _nutple = ntuple(2)
        kernel_size = _nutple(kernel_size)
        stride = _nutple(stride)
        padding = _nutple(padding)
        dilation = _nutple(dilation)
        super(oConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, False,
                                      _nutple(0), groups, sharing_rates, bias)

    def forward(self, input):
        weight = self.weight.index_select(0, self.indices[1]).index_select(1, self.indices[0])
        return F.conv2d(input, weight, self.bias, self.stride, self.padding, self.dilation, 1)
