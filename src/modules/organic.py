import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.nn.modules.conv import _ConvNd
from utils import ntuple


class _oConvNd(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, transposed, output_padding,
                 groups, sharing_rates, bias, padding_mode):
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
        self.groups = (groups[0], groups[0] * groups[1]) if len(_nutple(groups)) == 2 else (groups[0], groups[0])
        self.sharing_rates = sharing_rates if len(sharing_rates) == 2 else (sharing_rates[0], sharing_rates[0])
        self.input_size = int(self.in_channels * (1 - self.sharing_rates[0] + self.sharing_rates[0] / self.groups[0]))
        self.output_size = int(self.out_channels * (1 - self.sharing_rates[1] + self.sharing_rates[1] / self.groups[1]))
        self.padding_mode = padding_mode
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
