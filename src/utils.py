import collections.abc as container_abcs
import errno
import os
from itertools import repeat

import numpy as np
import torch
from torchvision.utils import make_grid
from torchvision.utils import save_image


def makedir_exist_ok(dirpath):
    try:
        os.makedirs(dirpath)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise
    return


def save(input, path, protocol=2, mode='torch'):
    dirname = os.path.dirname(path)
    makedir_exist_ok(dirname)
    if mode == 'torch':
        torch.save(input, path, pickle_protocol=protocol)
    elif mode == 'numpy':
        np.save(path, input)
    else:
        raise ValueError('Not valid save mode')
    return


def load(path, mode='torch'):
    if mode == 'torch':
        return torch.load(path, map_location=lambda storage, loc: storage)
    elif mode == 'numpy':
        return np.load(path)
    else:
        raise ValueError('Not valid save mode')
    return


def dict_to_device(input, device):
    if isinstance(input, dict):
        for key in input:
            if isinstance(input[key], list):
                for i in range(len(input[key])):
                    input[key][i] = input[key][i].to(device)
            elif isinstance(input[key], torch.Tensor):
                input[key] = input[key].to(device)
            elif isinstance(input[key], dict):
                input[key] = dict_to_device(input[key], device)
            else:
                raise ValueError('Not valid input type')
    else:
        input = input.to(device)
    return input


def ntuple(n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable) and not isinstance(x, str):
            return x
        return tuple(repeat(x, n))

    return parse


def apply_fn(module, fn):
    for n, m in module.named_children():
        if hasattr(m, fn):
            exec('m.{0}()'.format(fn))
        if sum(1 for _ in m.named_children()) != 0:
            exec('apply_fn(m,\'{0}\')'.format(fn))
    return


def repackage_hidden(h):
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def batchify(data, batch_size):
    num_batch = data.size(0) // batch_size
    data = data.narrow(0, 0, num_batch * batch_size)
    data = data.reshape(batch_size, -1)
    return data


def make_batch(data, i, seq_len):
    seq_len = min(seq_len, data.size(1) - 1 - i)
    input = {}
    input['line'] = data[:, i:i + seq_len]
    input['symbol'] = data[:, i + 1:i + 1 + seq_len].contiguous().view(-1)
    return input
