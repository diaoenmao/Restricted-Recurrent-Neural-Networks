import config
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import itertools
from modules import Cell

device = config.PARAM['device']

def make_model(model):
    if (isinstance(model, dict)):
        if ('cell' in model):
            return Cell(model)
        else:
            cell = nn.ModuleDict({})
            for k in model:
                cell[k] = make_model(model[k])
            return cell
    elif (isinstance(model, list)):
        cell = nn.ModuleList([])
        for i in range(len(model)):
            cell.append(make_model(model[i]))
        return cell
    elif (isinstance(model, tuple)):
        container = []
        for i in range(len(model)):
            container.append(make_model(model[i]))
        cell = nn.Sequential(*container)
        return cell
    else:
        raise ValueError('wrong model info format')
    return


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.model = make_model(config.PARAM['model']['encoder'])

    def forward(self, input):
        x = input
        x = x * 2 - 1
        x = x.view(x.size(0), 1, -1)
        x = self.model['in'](x)
        x = self.model['main'](x)
        x = self.model['out'](x)
        probability = x
        x = (x.round() - x).detach() + x
        x = torch.cat([input, x], dim=1) if (config.PARAM['systematic']) else x
        x = x * 2 - 1
        return x, probability


class Channel(nn.Module):
    def __init__(self):
        super(Channel, self).__init__()
        self.model = make_model(config.PARAM['model']['channel'])

    def forward(self, input):
        x = input
        x = self.model(x)
        return x


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.model = make_model(config.PARAM['model']['decoder'])

    def channel_loss_fn(self, input, output):
        loss = torch.tensor(0, device=device, dtype=torch.float32)
        loss += F.binary_cross_entropy(output['channel']['decoded'], input['bits'], reduction='mean')
        return loss

    def forward(self, input):
        x = input
        x = self.model['in'](x)
        x = self.model['main'](x)
        x = self.model['out'](x)
        x = x.view(x.size(0), -1)
        return x


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.encoder = Encoder()
        self.channel = Channel()
        self.decoder = Decoder()

    def forward(self, input):
        output = {'loss': torch.tensor(0, device=device, dtype=torch.float32),
                  'channel': {}}

        output['channel']['mask'] = ((input['bits'].view(input['bits'].size(0), 1, -1) - input['bits']).abs().sum(
            dim=-1)) != 0
        output['channel']['code'], output['channel']['probability'] = self.encoder(input['bits'])
        output['channel']['received_code'] = self.channel(output['channel']['code'])
        output['channel']['decoded'] = self.decoder(output['channel']['received_code'])
        output['loss'] = self.decoder.channel_loss_fn(input, output)
        output['channel']['bits'] = output['channel']['decoded'].round()
        return output


def smartcode_cnn():
    config.PARAM['systematic'] = False

    config.PARAM['model'] = {}
    config.PARAM['model']['encoder'] = {
        'in': {'cell': 'BasicCell', 'mode': 'Conv1d', 'input_size': 1, 'output_size': 100, 'kernel_size': 3,
               'stride': 1, 'padding': 1, 'normalization': config.PARAM['normalization'],
               'activation': config.PARAM['activation'], 'bias': False},
        'main': ({'cell': 'BasicCell', 'mode': 'Conv1d', 'input_size': 100, 'output_size': 100, 'kernel_size': 3,
                  'stride': 1, 'padding': 1, 'normalization': config.PARAM['normalization'],
                  'activation': config.PARAM['activation'], 'bias': False},) * (config.PARAM['num_layer'] - 2),
        'out': {'cell': 'BasicCell', 'mode': 'Conv1d', 'input_size': 100, 'output_size': config.PARAM['R'],
                'kernel_size': 3, 'stride': 1, 'padding': 1, 'normalization': config.PARAM['normalization'],
                'activation': 'sigmoid', 'bias': False}
    }
    config.PARAM['model']['channel'] = {
        'cell': 'ChannelCell', 'mode': config.PARAM['channel_mode'], 'snr': config.PARAM['snr']
    }
    config.PARAM['model']['decoder'] = {
        'in': {'cell': 'BasicCell', 'mode': 'Conv1d', 'input_size': config.PARAM['R'], 'output_size': 100,
               'kernel_size': 3, 'stride': 1, 'padding': 1, 'normalization': config.PARAM['normalization'],
               'activation': config.PARAM['activation'], 'bias': False},
        'main': ({'cell': 'BasicCell', 'mode': 'Conv1d', 'input_size': 100, 'output_size': 100, 'kernel_size': 3,
                  'stride': 1, 'padding': 1, 'normalization': config.PARAM['normalization'],
                  'activation': config.PARAM['activation'], 'bias': False},) * (config.PARAM['num_layer'] - 2),
        'out': {'cell': 'BasicCell', 'mode': 'Conv1d', 'input_size': 100, 'output_size': 1, 'kernel_size': 3,
                'stride': 1, 'padding': 1, 'normalization': config.PARAM['normalization'], 'activation': 'sigmoid',
                'bias': False}
    }
    model = Model()
    return model