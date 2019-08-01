import config
import copy
import torch
import torch.nn as nn

device = config.PARAM['device']


def Normalization(cell_info):
    if (cell_info['mode'] == 'none'):
        return nn.Sequential()
    elif (cell_info['mode'] == 'bn'):
        return nn.BatchNorm1d(cell_info['input_size'])
    elif (cell_info['mode'] == 'in'):
        return nn.InstanceNorm1d(cell_info['input_size'])
    else:
        raise ValueError('Normalization mode not supported')
    return


def Activation(cell_info):
    if (cell_info['mode'] == 'none'):
        return nn.Sequential()
    elif (cell_info['mode'] == 'tanh'):
        return nn.Tanh()
    elif (cell_info['mode'] == 'hardtanh'):
        return nn.Hardtanh()
    elif (cell_info['mode'] == 'relu'):
        return nn.ReLU(inplace=True)
    elif (cell_info['mode'] == 'prelu'):
        return nn.PReLU()
    elif (cell_info['mode'] == 'elu'):
        return nn.ELU(inplace=True)
    elif (cell_info['mode'] == 'selu'):
        return nn.SELU(inplace=True)
    elif (cell_info['mode'] == 'celu'):
        return nn.CELU(inplace=True)
    elif (cell_info['mode'] == 'sigmoid'):
        return nn.Sigmoid()
    elif (cell_info['mode'] == 'softmax'):
        return nn.SoftMax()
    else:
        raise ValueError('Activation mode not supported')
    return


class BasicCell(nn.Module):
    def __init__(self, cell_info):
        super(BasicCell, self).__init__()
        self.cell_info = cell_info
        self.cell = self.make_cell()

    def make_cell(self):
        cell_info = copy.deepcopy(self.cell_info)
        cell = nn.ModuleDict({})
        if (cell_info['mode'] == 'Conv1d'):
            cell_in_info = {'cell': 'Conv1d', 'input_size': cell_info['input_size'],
                            'output_size': cell_info['output_size'],
                            'kernel_size': cell_info['kernel_size'], 'stride': cell_info['stride'],
                            'padding': cell_info['padding'], 'dilation': cell_info['dilation'],
                            'groups': cell_info['groups'], 'bias': cell_info['bias']}
        elif (cell_info['mode'] == 'ConvTranspose1d'):
            cell_in_info = {'cell': 'ConvTranspose1d', 'input_size': cell_info['input_size'],
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


class ResBasicCell(nn.Module):
    def __init__(self, cell_info):
        super(ResBasicCell, self).__init__()
        self.cell_info = cell_info
        self.cell = self.make_cell()

    def make_cell(self):
        cell_info = copy.deepcopy(self.cell_info)
        cell = nn.ModuleList([nn.ModuleDict({}) for _ in range(cell_info['num_layers'])])
        for i in range(cell_info['num_layers']):
            if (cell_info['mode'] == 'down'):
                cell_shortcut_info = {'input_size': cell_info['input_size'], 'output_size': cell_info['output_size'],
                                      'cell': 'BasicCell', 'mode': 'fc_down',
                                      'normalization': cell_info['normalization'], 'activation': 'none'}
            elif (cell_info['input_size'] != cell_info['output_size']):
                cell_shortcut_info = {'input_size': cell_info['input_size'], 'output_size': cell_info['output_size'],
                                      'cell': 'BasicCell', 'mode': 'fc',
                                      'normalization': cell_info['normalization'], 'activation': 'none'}
            else:
                cell_shortcut_info = {'cell': 'none'}
            cell_in_info = {'input_size': cell_info['input_size'], 'output_size': cell_info['output_size'],
                            'cell': 'BasicCell', 'mode': cell_info['mode'],
                            'normalization': cell_info['normalization'], 'activation': cell_info['activation']}
            cell_out_info = {'input_size': cell_info['output_size'], 'output_size': cell_info['output_size'],
                             'cell': 'BasicCell', 'mode': 'pass',
                             'normalization': cell_info['normalization'], 'activation': 'none'}
            cell[i]['shortcut'] = Cell(cell_shortcut_info)
            cell[i]['in'] = Cell(cell_in_info)
            cell[i]['out'] = Cell(cell_out_info)
            cell[i]['activation'] = Cell({'cell': 'Activation', 'mode': cell_info['activation']})
            cell_info['input_size'] = cell_info['output_size']
            cell_info['mode'] = 'pass'
        return cell

    def forward(self, input):
        x = input
        for i in range(len(self.cell)):
            shortcut = self.cell[i]['shortcut'](x)
            x = self.cell[i]['in'](x)
            x = self.cell[i]['out'](x)
            x = self.cell[i]['activation'](x + shortcut)
        return x


class LSTMCell(nn.Module):
    def __init__(self, cell_info):
        super(LSTMCell, self).__init__()
        self.cell_info = cell_info
        self.cell = self.make_cell()
        self.hidden = None

    def make_cell(self):
        cell_info = copy.deepcopy(self.cell_info)
        cell = nn.ModuleList([nn.ModuleDict({}) for _ in range(cell_info['num_layers'])])
        for i in range(cell_info['num_layers']):
            cell_in_info = {'cell': 'Conv1d', 'input_size': cell_info['input_size'],
                            'output_size': 4 * cell_info['output_size'],
                            'kernel_size': cell_info['kernel_size'], 'stride': cell_info['stride'],
                            'padding': cell_info['padding'], 'bias': cell_info['bias'], 'normalization': 'none',
                            'activation': 'none'}
            cell_hidden_info = {'cell': 'Conv1d', 'input_size': cell_info['output_size'],
                                'output_size': 4 * cell_info['output_size'],
                                'kernel_size': cell_info['kernel_size'], 'stride': cell_info['stride'],
                                'padding': cell_info['padding'], 'bias': cell_info['bias'], 'normalization': 'none',
                                'activation': 'none'}
            cell[i]['in'] = Cell(cell_in_info)
            cell[i]['hidden'] = Cell(cell_hidden_info)
            cell[i]['activation'] = nn.ModuleList(
                [Activation({'cell': 'Activation', 'mode': self.cell_info['activation']}),
                 Activation({'cell': 'Activation', 'mode': self.cell_info['activation']})])
        return cell

    def init_hidden(self, hidden_size):
        hidden = [[torch.zeros(hidden_size, device=device)], [torch.zeros(hidden_size, device=device)]]
        return hidden

    def free_hidden(self):
        self.hidden = None
        return

    def forward(self, input, hidden=None):
        x = input
        hx, cx = [None for _ in range(len(self.cell))], [None for _ in range(len(self.cell))]
        for i in range(len(self.cell)):
            y = [None for _ in range(x.size(1))]
            for j in range(x.size(1)):
                gates = self.cell[i]['in'](x[:, j])
                if (hidden is None):
                    if (self.hidden is None):
                        self.hidden = self.init_hidden(
                            (gates.size(0), self.cell_info['output_size'], *gates.size()[2:]))
                    else:
                        if (i == len(self.hidden[0])):
                            new_hidden = self.init_hidden(
                                (gates.size(0), self.cell_info['output_size'], *gates.size()[2:]))
                            self.hidden[0].extend(new_hidden[0])
                            self.hidden[1].extend(new_hidden[1])
                        else:
                            pass
                if (j == 0):
                    hx[i], cx[i] = self.hidden[0][i], self.hidden[1][i]
                gates += self.cell[i]['hidden'](hx[i])
                ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
                ingate = torch.sigmoid(ingate)
                forgetgate = torch.sigmoid(forgetgate)
                cellgate = self.cell[i]['activation'][0](cellgate)
                outgate = torch.sigmoid(outgate)
                cx[i] = (forgetgate * cx[i]) + (ingate * cellgate)
                hx[i] = outgate * self.cell[i]['activation'][1](cx[i])
                y[j] = hx[i]
            x = torch.stack(y, dim=1)
        self.hidden = [hx, cx]
        return x


class ResLSTMCell(nn.Module):
    def __init__(self, cell_info):
        super(ResLSTMCell, self).__init__()
        self.cell_info = cell_info
        self.cell = self.make_cell()
        self.hidden = None

    def make_cell(self):
        cell_info = copy.deepcopy(self.cell_info)
        cell = nn.ModuleList([nn.ModuleDict({}) for _ in range(cell_info['num_layers'])])
        cell_shortcut_info = {'cell': 'none'} if (cell_info['input_size'] == cell_info['output_size']) else \
            {'cell': 'Conv1d', 'input_size': cell_info['output_size'], 'output_size': cell_info['output_size'],
             'kernel_size': 1, 'stride': 1, 'padding': 0, 'bias': cell_info['bias'], 'normalization': 'none',
             'activation': 'none'}
        cell[0]['shortcut'] = Cell(cell_shortcut_info)
        for i in range(cell_info['num_layers']):
            cell_in_info = {'cell': 'Conv1d', 'input_size': cell_info['input_size'],
                            'output_size': 4 * cell_info['output_size'],
                            'kernel_size': cell_info['kernel_size'], 'stride': cell_info['stride'],
                            'padding': cell_info['padding'], 'bias': cell_info['bias'], 'normalization': 'none',
                            'activation': 'none'}
            cell_hidden_info = {'cell': 'Conv1d', 'input_size': cell_info['output_size'],
                                'output_size': 4 * cell_info['output_size'],
                                'kernel_size': cell_info['kernel_size'], 'stride': cell_info['stride'],
                                'padding': cell_info['padding'], 'bias': cell_info['bias'], 'normalization': 'none',
                                'activation': 'none'}
            cell[i]['in'] = Cell(cell_in_info)
            cell[i]['hidden'] = Cell(cell_hidden_info)
            cell[i]['activation'] = nn.ModuleList(
                [Activation({'cell': 'Activation', 'mode': self.cell_info['activation']}),
                 Activation({'cell': 'Activation', 'mode': self.cell_info['activation']})])
        return cell

    def init_hidden(self, hidden_size):
        hidden = [[torch.zeros(hidden_size, device=device)], [torch.zeros(hidden_size, device=device)]]
        return hidden

    def free_hidden(self):
        self.hidden = None
        return

    def forward(self, input, hidden=None):
        x = input
        hx, cx = [None for _ in range(len(self.cell))], [None for _ in range(len(self.cell))]
        shortcut = [None for _ in range(x.size(1))]
        for i in range(len(self.cell)):
            y = [None for _ in range(x.size(1))]
            for j in range(x.size(1)):
                if (i == 0):
                    shortcut[j] = self.cell[i]['shortcut'](x[:, j])
                gates = self.cell[i]['in'](x[:, j])
                if (hidden is None):
                    if (self.hidden is None):
                        self.hidden = self.init_hidden(
                            (gates.size(0), self.cell_info['hidden'][i]['output_size'], *gates.size()[2:]))
                    else:
                        if (i == len(self.hidden[0])):
                            new_hidden = self.init_hidden(
                                (gates.size(0), self.cell_info['hidden'][i]['output_size'], *gates.size()[2:]))
                            self.hidden[0].extend(new_hidden[0])
                            self.hidden[1].extend(new_hidden[1])
                        else:
                            pass
                else:
                    self.hidden = hidden
                if (j == 0):
                    hx[i], cx[i] = self.hidden[0][i], self.hidden[1][i]
                gates += self.cell[i]['hidden'](hx[i])
                ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
                ingate = torch.sigmoid(ingate)
                forgetgate = torch.sigmoid(forgetgate)
                cellgate = self.cell[i]['activation'][0](cellgate)
                outgate = torch.sigmoid(outgate)
                cx[i] = (forgetgate * cx[i]) + (ingate * cellgate)
                hx[i] = outgate * self.cell[i]['activation'][1](cx[i]) if (i < len(self.cell) - 1) else outgate * (
                            shortcut[j] + self.cell[i]['activation'][1](cx[i]))
                y[j] = hx[i]
            x = torch.stack(y, dim=1)
        self.hidden = [hx, cx]
        return x


class ChannelCell(nn.Module):
    def __init__(self, cell_info):
        super(ChannelCell, self).__init__()
        self.cell_info = cell_info
        self.cell = self.make_cell()

    def make_cell(self):
        cell_info = copy.deepcopy(self.cell_info)
        cell = Channel(cell_info['mode'], cell_info['snr'])
        return cell

    def forward(self, input):
        x = input
        x = self.cell(x)
        return x


class Cell(nn.Module):
    def __init__(self, cell_info):
        super(Cell, self).__init__()
        self.cell_info = cell_info
        self.cell = self.make_cell()

    def make_cell(self):
        if (self.cell_info['cell'] == 'none'):
            cell = nn.Sequential()
        elif (self.cell_info['cell'] == 'Normalization'):
            cell = Normalization(self.cell_info)
        elif (self.cell_info['cell'] == 'Activation'):
            cell = Activation(self.cell_info)
        elif (self.cell_info['cell'] == 'Conv1d'):
            default_cell_info = {'kernel_size': 3, 'stride': 1, 'padding': 1, 'dilation': 1, 'groups': 1, 'bias': False}
            self.cell_info = {**default_cell_info, **self.cell_info}
            cell = nn.Conv1d(self.cell_info['input_size'], self.cell_info['output_size'], self.cell_info['kernel_size'],
                             self.cell_info['stride'], self.cell_info['padding'], self.cell_info['dilation'],
                             self.cell_info['groups'], self.cell_info['bias'])
        elif (self.cell_info['cell'] == 'ConvTranspose1d'):
            default_cell_info = {'kernel_size': 3, 'stride': 1, 'padding': 1, 'output_padding': 0, 'dilation': 1,
                                 'groups': 1, 'bias': False}
            self.cell_info = {**default_cell_info, **self.cell_info}
            cell = nn.ConvTranspose1d(self.cell_info['input_size'], self.cell_info['output_size'],
                                      self.cell_info['kernel_size'],
                                      self.cell_info['stride'], self.cell_info['padding'],
                                      self.cell_info['output_padding'], self.cell_info['groups'],
                                      self.cell_info['bias'], self.cell_info['dilation'])
        elif (self.cell_info['cell'] == 'BasicCell'):
            default_cell_info = {'mode': 'Conv1d', 'kernel_size': 3, 'stride': 1, 'padding': 1, 'output_padding': 0,
                                 'dilation': 1, 'groups': 1, 'bias': False, 'normalization': 'bn', 'activation': 'relu'}
            self.cell_info = {**default_cell_info, **self.cell_info}
            cell = BasicCell(self.cell_info)
        elif (self.cell_info['cell'] == 'ResBasicCell'):
            cell = ResBasicCell(self.cell_info)
        elif (self.cell_info['cell'] == 'LSTMCell'):
            default_cell_info = {'activation': 'tanh'}
            self.cell_info = {**default_cell_info, **self.cell_info}
            cell = LSTMCell(self.cell_info)
        elif (self.cell_info['cell'] == 'ResLSTMCell'):
            default_cell_info = {'activation': 'tanh'}
            self.cell_info = {**default_cell_info, **self.cell_info}
            cell = ResLSTMCell(self.cell_info)
        elif (self.cell_info['cell'] == 'ChannelCell'):
            cell = ChannelCell(self.cell_info)
        else:
            raise ValueError('{} model mode not supported'.format(self.cell_info['cell']))
        return cell

    def forward(self, *input):
        x = self.cell(*input)
        return x
