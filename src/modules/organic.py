import config
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import numpy as np
from torch.nn.modules.conv import _ConvNd
from utils import ntuple


def make_organic(in_channels, out_channels, sharing_rates):
    shared_size = (out_channels.float() * sharing_rates).round().long()
    nonshared_size = out_channels - shared_size
    input_size = in_channels.max().item()
    output_size = (shared_size.max() + nonshared_size.sum()).item()
    input_indices = [torch.arange(in_channels[i]).to(config.PARAM['device']) for i in range(len(in_channels))]
    output_indices = [[] for _ in range(out_channels.size(0))]
    pivot = shared_size.max()
    for i in range(out_channels.size(0)):
        for j in range(out_channels.size(1)):
            shared_indices = torch.arange(shared_size[i][j])
            nonshared_indices = torch.arange(pivot, pivot + nonshared_size[i][j])
            pivot += nonshared_size[i][j]
            output_indices[i].append(torch.cat([shared_indices, nonshared_indices]).to(config.PARAM['device']))
        output_indices[i] = torch.cat(output_indices[i])
    return input_size, input_indices, output_size, output_indices


class _oConvNd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, transposed, output_padding,
                 sharing_rates, bias):
        super(_oConvNd, self).__init__()
        self.register_buffer('in_channels', torch.tensor(in_channels))
        self.register_buffer('out_channels', torch.tensor(out_channels))
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.register_buffer('sharing_rates', torch.tensor(sharing_rates, dtype=torch.float32))
        self.input_size, self.input_indices, self.output_size, self.output_indices = make_organic(self.in_channels,
                                                                                                  self.out_channels,
                                                                                                  self.sharing_rates)
        print(self.input_size)
        print(self.output_size)
        print(self.input_indices)
        print(self.output_indices)
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
        if self.bias is None:
            s += ', bias=False'
        return s.format(**self.__dict__)


class oConv2d(_oConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, sharing_rates=0,
                 bias=True):
        _nutple = ntuple(2)
        kernel_size = _nutple(kernel_size)
        stride = _nutple(stride)
        padding = _nutple(padding)
        dilation = _nutple(dilation)
        super(oConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, False,
                                      _nutple(0), sharing_rates, bias)

    def forward(self, input):
        input = input.split(self.in_channels.tolist(), dim=1)
        output = []
        for i in range(len(input)):
            weight = self.weight.index_select(1, self.input_indices[i]).index_select(0, self.output_indices[i])
            bias = self.bias.index_select(0, self.output_indices[i]) if self.bias is not None else None
            output.append(F.conv2d(input[i], weight, bias, self.stride, self.padding, self.dilation, 1))
        output = torch.cat(output, dim=1)
        return output
