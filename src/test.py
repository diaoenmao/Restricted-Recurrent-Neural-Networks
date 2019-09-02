import config

config.init()
import argparse
import torch.backends.cudnn as cudnn
from data import *
from metrics import *
from utils import *

cudnn.benchmark = True
parser = argparse.ArgumentParser(description='Config')
for k in config.PARAM:
    exec('parser.add_argument(\'--{0}\',default=config.PARAM[\'{0}\'], help=\'\')'.format(k))
args = vars(parser.parse_args())
for k in config.PARAM:
    if config.PARAM[k] != args[k]:
        exec('config.PARAM[\'{0}\'] = {1}'.format(k, args[k]))


def collate(input):
    for k in input:
        input[k] = torch.stack(input[k], 0)
    return input


if __name__ == '__main__':
    dataset = fetch_dataset('WikiText103')
    print(len(dataset['train'].data),len(dataset['test'].data))
