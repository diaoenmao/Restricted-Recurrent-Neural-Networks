import config
import copy
import torch
import torch.nn as nn
from modules.organic import oConv2d
from utils import ntuple

device = config.PARAM['device']


def Normalization(cell_info):
    if cell_info['mode'] == 'none':
        return nn.Sequential()
    elif cell_info['mode'] == 'bn':
        return nn.BatchNorm2d(cell_info['input_size'])
    elif cell_info['mode'] == 'in':
        return nn.InstanceNorm2d(cell_info['input_size'])
    else:
        raise ValueError('Normalization mode not supported')
    return


def Activation(cell_info):
    if cell_info['mode'] == 'none':
        return nn.Sequential()
    elif cell_info['mode'] == 'tanh':
        return nn.Tanh()
    elif cell_info['mode'] == 'hardtanh':
        return nn.Hardtanh()
    elif cell_info['mode'] == 'relu':
        return nn.ReLU(inplace=True)
    elif cell_info['mode'] == 'prelu':
        return nn.PReLU()
    elif cell_info['mode'] == 'elu':
        return nn.ELU(inplace=True)
    elif cell_info['mode'] == 'selu':
        return nn.SELU(inplace=True)
    elif cell_info['mode'] == 'celu':
        return nn.CELU(inplace=True)
    elif cell_info['mode'] == 'sigmoid':
        return nn.Sigmoid()
    elif cell_info['mode'] == 'softmax':
        return nn.SoftMax()
    else:
        raise ValueError('Activation mode not supported')
    return


def Dropout(cell_info):
    if cell_info['p'] == 0:
        return nn.Sequential()
    else:
        return nn.Dropout(p=cell_info['p'], inplace=True)
    return


class BasicCell(nn.Module):
    def __init__(self, cell_info):
        super(BasicCell, self).__init__()
        self.cell_info = cell_info
        self.cell = self.make_cell()

    def make_cell(self):
        cell_info = copy.deepcopy(self.cell_info)
        cell = nn.ModuleDict({})
        if cell_info['mode'] == 'Conv2d':
            cell_in_info = {'cell': 'Conv2d', 'input_size': cell_info['input_size'],
                            'output_size': cell_info['output_size'],
                            'kernel_size': cell_info['kernel_size'], 'stride': cell_info['stride'],
                            'padding': cell_info['padding'], 'dilation': cell_info['dilation'],
                            'groups': cell_info['groups'], 'bias': cell_info['bias']}
        elif cell_info['mode'] == 'oConv2d':
            cell_in_info = {'cell': 'oConv2d', 'input_size': cell_info['input_size'],
                            'output_size': cell_info['output_size'],
                            'kernel_size': cell_info['kernel_size'], 'stride': cell_info['stride'],
                            'padding': cell_info['padding'], 'dilation': cell_info['dilation'],
                            'sharing_rates': cell_info['sharing_rates'], 'bias': cell_info['bias']}
        elif cell_info['mode'] == 'ConvTranspose2d':
            cell_in_info = {'cell': 'ConvTranspose2d', 'input_size': cell_info['input_size'],
                            'output_size': cell_info['output_size'],
                            'kernel_size': cell_info['kernel_size'], 'stride': cell_info['stride'],
                            'padding': cell_info['padding'], 'output_padding': cell_info['output_padding'],
                            'dilation': cell_info['dilation'], 'groups': cell_info['groups'], 'bias': cell_info['bias']}
        else:
            raise ValueError('model mode not supported')
        cell['in'] = Cell(cell_in_info)
        cell['activation'] = Cell({'cell': 'Activation', 'mode': cell_info['activation']})
        cell['normalization'] = Cell(
            {'cell': 'Normalization', 'input_size': cell_info['output_size'], 'mode': cell_info['normalization']})
        return cell

    def forward(self, input):
        x = input
        x = self.cell['activation'](self.cell['normalization'](self.cell['in'](x)))
        return x

class RRNNCell(nn.Module):
    def __init__(self, cell_info):
        super(RRNNCell, self).__init__()
        self.cell_info = cell_info
        self.cell = self.make_cell()
        self.hidden = None

    def make_cell(self):
        cell_info = copy.deepcopy(self.cell_info)
        cell = nn.ModuleList([nn.ModuleDict({}) for _ in range(cell_info['num_layers'])])
        _ntuple = ntuple(cell_info['num_layers'])
        cell_info['sharing_rates'] = _ntuple(cell_info['sharing_rates'])
        for i in range(cell_info['num_layers']):
            cell_in_info = {'cell': 'oConv2d', 'input_size': (cell_info['input_size'], cell_info['output_size']),
                            'output_size': ((cell_info['output_size'],),(cell_info['output_size'],)),
                            'kernel_size': cell_info['kernel_size'], 'stride': cell_info['stride'],
                            'padding': cell_info['padding'], 'sharing_rates': cell_info['sharing_rates'][i],
                            'bias': cell_info['bias'], 'normalization': 'none', 'activation': 'none'}
            cell[i]['in'] = Cell(cell_in_info)
            cell[i]['activation'] = Cell({'cell': 'Activation', 'mode': cell_info['activation']})
            cell[i]['dropout'] = Cell({'cell': 'Dropout', 'p': cell_info['dropout']})
            cell_info['input_size'] = cell_info['output_size']
        return cell

    def init_hidden(self, hidden_size):
        hidden = [torch.zeros(hidden_size, device=device) for _ in range(len(self.cell))]
        return hidden

    def free_hidden(self):
        self.hidden = None
        return

    def detach_hidden(self):
        for i in range(len(self.cell)):
            self.hidden[i].detach_()

    def forward(self, input, hidden=None):
        x = input
        if hidden is None:
            self.hidden = self.init_hidden(
                (x.size(0), self.cell_info['output_size'], *x.size()[3:])) if self.hidden is None else self.hidden
        else:
            self.hidden = hidden
        for i in range(len(self.cell)):
            y = []
            for j in range(x.size(1)):
                xhx = torch.cat([x[:, j], self.hidden[i]], dim=1)
                gates = self.cell[i]['in'](xhx).chunk(2, 1)
                cellgate = (gates[0] + gates[1])
                cellgate = self.cell[i]['activation'](cellgate)
                self.hidden[i] = cellgate
                y.append(self.hidden[i])
            x = torch.stack(y, dim=1)
            x = self.cell[i]['dropout'](x)
        return x


class RLSTMCell(nn.Module):
    def __init__(self, cell_info):
        super(RLSTMCell, self).__init__()
        self.cell_info = cell_info
        self.cell = self.make_cell()
        self.hidden = None

    def make_cell(self):
        cell_info = copy.deepcopy(self.cell_info)
        cell = nn.ModuleList([nn.ModuleDict({}) for _ in range(cell_info['num_layers'])])
        _ntuple = ntuple(cell_info['num_layers'])
        cell_info['sharing_rates'] = _ntuple(cell_info['sharing_rates'])
        for i in range(cell_info['num_layers']):
            cell_in_info = {'cell': 'oConv2d', 'input_size': (cell_info['input_size'], cell_info['output_size']),
                            'output_size': ((cell_info['output_size'],) * 4,(cell_info['output_size'],) * 4),
                            'kernel_size': cell_info['kernel_size'], 'stride': cell_info['stride'],
                            'padding': cell_info['padding'], 'sharing_rates': cell_info['sharing_rates'][i],
                            'bias': cell_info['bias'], 'normalization': 'none', 'activation': 'none'}
            cell[i]['in'] = Cell(cell_in_info)
            cell[i]['activation'] = nn.ModuleList(
                [Cell({'cell': 'Activation', 'mode': cell_info['activation']}),
                 Cell({'cell': 'Activation', 'mode': cell_info['activation']})])
            cell[i]['dropout'] = Cell({'cell': 'Dropout', 'p': cell_info['dropout']})
            cell_info['input_size'] = cell_info['output_size']
        return cell

    def init_hidden(self, hidden_size):
        hidden = [[torch.zeros(hidden_size, device=device) for _ in range(len(self.cell))],
                  [torch.zeros(hidden_size, device=device) for _ in range(len(self.cell))]]
        return hidden

    def free_hidden(self):
        self.hidden = None
        return

    def detach_hidden(self):
        for i in range(len(self.cell)):
            self.hidden[0][i].detach_()
            self.hidden[1][i].detach_()

    def forward(self, input, hidden=None):
        x = input
        if hidden is None:
            self.hidden = self.init_hidden(
                (x.size(0), self.cell_info['output_size'], *x.size()[3:])) if self.hidden is None else self.hidden
        else:
            self.hidden = hidden
        for i in range(len(self.cell)):
            y = []
            for j in range(x.size(1)):
                xhx = torch.cat([x[:, j], self.hidden[0][i]], dim=1)
                gates = self.cell[i]['in'](xhx).chunk(2, 1)
                ingate, forgetgate, cellgate, outgate = (gates[0] + gates[1]).chunk(4, 1)
                ingate = torch.sigmoid(ingate)
                forgetgate = torch.sigmoid(forgetgate)
                cellgate = self.cell[i]['activation'][0](cellgate)
                outgate = torch.sigmoid(outgate)
                self.hidden[1][i] = (forgetgate * self.hidden[1][i]) + (ingate * cellgate)
                self.hidden[0][i] = outgate * self.cell[i]['activation'][1](self.hidden[1][i])
                y.append(self.hidden[0][i])
            x = torch.stack(y, dim=1)
            x = self.cell[i]['dropout'](x)
        return x


class RGRUCell(nn.Module):
    def __init__(self, cell_info):
        super(RGRUCell, self).__init__()
        self.cell_info = cell_info
        self.cell = self.make_cell()
        self.hidden = None

    def make_cell(self):
        cell_info = copy.deepcopy(self.cell_info)
        cell = nn.ModuleList([nn.ModuleDict({}) for _ in range(cell_info['num_layers'])])
        _ntuple = ntuple(cell_info['num_layers'])
        cell_info['sharing_rates'] = _ntuple(cell_info['sharing_rates'])
        for i in range(cell_info['num_layers']):
            cell_in_info = {'cell': 'oConv2d', 'input_size': (cell_info['input_size'], cell_info['output_size']),
                            'output_size': ((cell_info['output_size'],) * 3,(cell_info['output_size'],) * 3),
                            'kernel_size': cell_info['kernel_size'], 'stride': cell_info['stride'],
                            'padding': cell_info['padding'], 'sharing_rates': cell_info['sharing_rates'][i],
                            'bias': cell_info['bias'], 'normalization': 'none', 'activation': 'none'}
            cell[i]['in'] = Cell(cell_in_info)
            cell[i]['activation'] = Cell({'cell': 'Activation', 'mode': cell_info['activation']})
            cell[i]['dropout'] = Cell({'cell': 'Dropout', 'p': cell_info['dropout']})
            cell_info['input_size'] = cell_info['output_size']
        return cell

    def init_hidden(self, hidden_size):
        hidden = [torch.zeros(hidden_size, device=device) for _ in range(len(self.cell))]
        return hidden

    def free_hidden(self):
        self.hidden = None
        return

    def detach_hidden(self):
        for i in range(len(self.cell)):
            self.hidden[i].detach_()

    def forward(self, input, hidden=None):
        x = input
        if hidden is None:
            self.hidden = self.init_hidden(
                (x.size(0), self.cell_info['output_size'], *x.size()[3:])) if self.hidden is None else self.hidden
        else:
            self.hidden = hidden
        for i in range(len(self.cell)):
            y = []
            for j in range(x.size(1)):
                xhx = torch.cat([x[:, j], self.hidden[i]], dim=1)
                gates = self.cell[i]['in'](xhx).chunk(2, 1)
                gates_0, gates_1 = gates[0].chunk(3, 1), gates[1].chunk(3, 1)
                forgetgate = torch.sigmoid(gates_0[0] + gates_1[0])
                outgate = torch.sigmoid(gates_0[1] + gates_1[1])
                cellgate = self.cell[i]['activation'](gates_0[2] + forgetgate*gates_1[2])
                self.hidden[i] = (1-outgate) * cellgate + outgate * self.hidden[i]
                y.append(self.hidden[i])
            x = torch.stack(y, dim=1)
            x = self.cell[i]['dropout'](x)
        return x


class Cell(nn.Module):
    def __init__(self, cell_info):
        super(Cell, self).__init__()
        self.cell_info = cell_info
        self.cell = self.make_cell()

    def make_cell(self):
        if self.cell_info['cell'] == 'none':
            cell = nn.Sequential()
        elif self.cell_info['cell'] == 'Normalization':
            cell = Normalization(self.cell_info)
        elif self.cell_info['cell'] == 'Activation':
            cell = Activation(self.cell_info)
        elif self.cell_info['cell'] == 'Dropout':
            cell = Dropout(self.cell_info)
        elif self.cell_info['cell'] == 'Conv2d':
            default_cell_info = {'kernel_size': 3, 'stride': 1, 'padding': 1, 'dilation': 1, 'groups': 1, 'bias': False}
            self.cell_info = {**default_cell_info, **self.cell_info}
            cell = nn.Conv2d(self.cell_info['input_size'], self.cell_info['output_size'], self.cell_info['kernel_size'],
                             self.cell_info['stride'], self.cell_info['padding'], self.cell_info['dilation'],
                             self.cell_info['groups'], self.cell_info['bias'])
        elif self.cell_info['cell'] == 'oConv2d':
            default_cell_info = {'kernel_size': 3, 'stride': 1, 'padding': 1, 'dilation': 1, 'in_sharing_rates': 0,
                                 'out_sharing_rates': 0, 'bias': False}
            self.cell_info = {**default_cell_info, **self.cell_info}
            cell = oConv2d(self.cell_info['input_size'], self.cell_info['output_size'],
                           self.cell_info['kernel_size'],
                           self.cell_info['stride'], self.cell_info['padding'], self.cell_info['dilation'],
                           self.cell_info['sharing_rates'], self.cell_info['bias'])
        elif self.cell_info['cell'] == 'ConvTranspose2d':
            default_cell_info = {'kernel_size': 3, 'stride': 1, 'padding': 1, 'output_padding': 0, 'dilation': 1,
                                 'groups': 1, 'bias': False}
            self.cell_info = {**default_cell_info, **self.cell_info}
            cell = nn.ConvTranspose2d(self.cell_info['input_size'], self.cell_info['output_size'],
                                      self.cell_info['kernel_size'],
                                      self.cell_info['stride'], self.cell_info['padding'],
                                      self.cell_info['output_padding'], self.cell_info['groups'],
                                      self.cell_info['bias'], self.cell_info['dilation'])
        elif self.cell_info['cell'] == 'BasicCell':
            default_cell_info = {'mode': 'Conv2d', 'kernel_size': 3, 'stride': 1, 'padding': 1, 'output_padding': 0,
                                 'dilation': 1, 'groups': 1, 'bias': False, 'normalization': 'bn', 'activation': 'relu'}
            self.cell_info = {**default_cell_info, **self.cell_info}
            cell = BasicCell(self.cell_info)
        elif self.cell_info['cell'] == 'RRNNCell':
            default_cell_info = {'activation': 'tanh'}
            self.cell_info = {**default_cell_info, **self.cell_info}
            cell = RRNNCell(self.cell_info)
        elif self.cell_info['cell'] == 'RGRUCell':
            default_cell_info = {'activation': 'tanh'}
            self.cell_info = {**default_cell_info, **self.cell_info}
            cell = RGRUCell(self.cell_info)
        elif self.cell_info['cell'] == 'RLSTMCell':
            default_cell_info = {'activation': 'tanh'}
            self.cell_info = {**default_cell_info, **self.cell_info}
            cell = RLSTMCell(self.cell_info)
        else:
            raise ValueError('Not valid {} model mode'.format(self.cell_info['cell']))
        return cell

    def forward(self, *input):
        x = self.cell(*input)
        return x
