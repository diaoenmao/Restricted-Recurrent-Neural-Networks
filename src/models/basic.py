import config
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import make_model
from utils import apply_fn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.encoder = nn.Embedding(len(config.PARAM['vocab']), config.PARAM['embedding_size'])
        self.rnn = make_model(config.PARAM['model']['rnn'])
        self.decoder = nn.Linear(config.PARAM['hidden_size'], len(config.PARAM['vocab']))
        if config.PARAM['tied']:
            self.decoder.weight = self.encoder.weight

    def loss_fn(self, input, output):
        loss = F.cross_entropy(output['logits'], input['symbol'], reduction='mean')
        return loss

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.deocder.bias.data.zero_()
        self.deocder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input):
        output = {'loss': torch.tensor(0, device=config.PARAM['device'], dtype=torch.float32), 'nlp': {}}
        x = input['line']
        x = self.encoder(x)
        x = x.view(*x.size(),1,1)
        x = self.rnn(x)
        x = x.view(-1,x.size(2))
        x = self.decoder(x)
        output['logits'] = x.view(-1,x.size(-1))
        output['loss'] = self.loss_fn(input, output)
        apply_fn(self, 'detach_hidden')
        return output


def basic():
    output_embedding_size = config.PARAM['embedding_size'] if config.PARAM['tied'] else config.PARAM['hidden_size']
    config.PARAM['model'] = {}
    config.PARAM['model']['rnn'] = ({'cell': 'RLSTMCell', 'input_size': config.PARAM['embedding_size'], 'output_size': config.PARAM['hidden_size'],
        'num_layers': config.PARAM['num_layer'] - 3, 'kernel_size': 1, 'stride': 1, 'padding': 0,
        'sharing_rates': config.PARAM['sharing_rates'], 'activation': config.PARAM['activation'], 'bias': True,
        'dropout': config.PARAM['dropout']}, {'cell': 'RLSTMCell', 'input_size': config.PARAM['hidden_size'], 'output_size': output_embedding_size,
        'num_layers': 1, 'kernel_size': 1, 'stride': 1, 'padding': 0,
        'sharing_rates': config.PARAM['sharing_rates'], 'activation': config.PARAM['activation'], 'bias': True,
        'dropout': 0})
    model = Model()
    return model
