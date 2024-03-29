import config

config.init()
import argparse
import models
import torch.backends.cudnn as cudnn
from collections import OrderedDict
from tabulate import tabulate
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.conv import _ConvNd
from data import *
from metrics import *
from utils import *
from modules.organic import _oConvNd

cudnn.benchmark = True
parser = argparse.ArgumentParser(description='Config')
for k in config.PARAM:
    exec('parser.add_argument(\'--{0}\',default=config.PARAM[\'{0}\'], help=\'\')'.format(k))
args = vars(parser.parse_args())
for k in config.PARAM:
    if config.PARAM[k] != args[k]:
        exec('config.PARAM[\'{0}\'] = {1}'.format(k, args[k]))


def main():
    process_control_name()
    runExperiment()
    return


def runExperiment():
    dataset = fetch_dataset(data_name=config.PARAM['data_name']['train'])
    batch_dataset = batchify(dataset['train'].data, config.PARAM['batch_size']['train'])
    config.PARAM['vocab'] = dataset['train'].vocab
    print(config.PARAM)

    model = eval('models.{}().to(device)'.format(config.PARAM['model_name']))
    summary = summarize(batch_dataset, model)
    content = parse_summary(summary)
    print(content)
    return


def summarize(dataset, model):
    def register_hook(module):

        def hook(module, input, output):
            module_name = str(module.__class__.__name__)
            if module_name not in summary['count']:
                summary['count'][module_name] = 1
            else:
                summary['count'][module_name] += 1
            key = str(hash(module))
            if key not in summary['module']:
                summary['module'][key] = OrderedDict()
                summary['module'][key]['module_name'] = '{}_{}'.format(module_name, summary['count'][module_name])
                summary['module'][key]['input_size'] = []
                summary['module'][key]['output_size'] = []
                summary['module'][key]['params'] = {}
            input_size = list(input[0].size())
            output_size = list(output[0].size())
            summary['module'][key]['input_size'].append(input_size)
            summary['module'][key]['output_size'].append(output_size)
            for name, param in module.named_parameters():
                if param.requires_grad:
                    if name == 'weight':
                        if name not in summary['module'][key]['params']:
                            summary['module'][key]['params'][name] = {}
                            summary['module'][key]['params'][name]['size'] = list(param.size())
                            summary['module'][key]['coordinates'] = []
                            summary['module'][key]['params'][name]['mask'] = torch.zeros(
                                summary['module'][key]['params'][name]['size'], dtype=torch.long,
                                device=config.PARAM['device'])
                    elif name == 'bias':
                        if name not in summary['module'][key]['params']:
                            summary['module'][key]['params'][name] = {}
                            summary['module'][key]['params'][name]['size'] = list(param.size())
                            summary['module'][key]['params'][name]['mask'] = torch.zeros(
                                summary['module'][key]['params'][name]['size'], dtype=torch.long,
                                device=config.PARAM['device'])
                    else:
                        continue
            if len(summary['module'][key]['params']) == 0:
                return
            if 'weight' in summary['module'][key]['params']:
                weight_size = summary['module'][key]['params']['weight']['size']
                if isinstance(module, _ConvNd):
                    if module.transposed:
                        summary['module'][key]['coordinates'].append(
                            [torch.arange(weight_size[0], device=config.PARAM['device']),
                             torch.arange(weight_size[1], device=config.PARAM['device'])])
                    else:
                        summary['module'][key]['coordinates'].append(
                            [torch.arange(weight_size[0], device=config.PARAM['device']),
                             torch.arange(weight_size[1], device=config.PARAM['device'])])
                if isinstance(module, _oConvNd):
                    if module.transposed:
                        summary['module'][key]['coordinates'].append(
                            [torch.arange(weight_size[0], device=config.PARAM['device']),
                             torch.arange(weight_size[1], device=config.PARAM['device'])])
                    else:
                        summary['module'][key]['coordinates'].append(
                            [torch.arange(weight_size[0], device=config.PARAM['device']),
                             torch.arange(weight_size[1], device=config.PARAM['device'])])
                elif isinstance(module, _BatchNorm):
                    summary['module'][key]['coordinates'].append(
                        [torch.arange(weight_size[0], device=config.PARAM['device'])])
                elif isinstance(module, nn.Linear):
                    summary['module'][key]['coordinates'].append(
                        [torch.arange(weight_size[0], device=config.PARAM['device']),
                         torch.arange(weight_size[1], device=config.PARAM['device'])])
                elif isinstance(module, nn.Embedding):
                    summary['module'][key]['coordinates'].append(
                        [torch.arange(weight_size[0], device=config.PARAM['device']),
                         torch.arange(weight_size[1], device=config.PARAM['device'])])
                else:
                    print(module)
                    raise ValueError('Not valid parametrized module')
            else:
                raise ValueError('Not valid parametrized module')
            for name in summary['module'][key]['params']:
                coordinates = summary['module'][key]['coordinates'][-1]
                if name == 'weight':
                    if len(coordinates) == 1:
                        summary['module'][key]['params'][name]['mask'][coordinates[0]] += 1
                    elif len(coordinates) == 2:
                        summary['module'][key]['params'][name]['mask'][
                            coordinates[0].view(-1, 1), coordinates[1].view(1, -1),] += 1
                    else:
                        raise ValueError('Not valid coordinates dimension')
                elif name == 'bias':
                    if len(coordinates) == 1:
                        summary['module'][key]['params'][name]['mask'] += 1
                    elif len(coordinates) == 2:
                        summary['module'][key]['params'][name]['mask'] += 1
                    else:
                        raise ValueError('Not valid coordinates dimension')
                else:
                    raise ValueError('Not valid parameters type')
            return

        if not isinstance(module, nn.Sequential) and not isinstance(module, nn.ModuleList) \
                and not isinstance(module, nn.ModuleDict) and module != model:
            hooks.append(module.register_forward_hook(hook))
        return

    run_mode = True
    summary = OrderedDict()
    summary['module'] = OrderedDict()
    summary['count'] = OrderedDict()
    hooks = []
    model.train(run_mode)
    model.apply(register_hook)
    for i, idx in enumerate(range(0, dataset.size(1) - 1, config.PARAM['bptt'])):
        input = make_batch(dataset, idx, config.PARAM['bptt'])
        input = dict_to_device(input, device)
        model(input)
        break
    for h in hooks:
        h.remove()
    summary['total_num_param'] = 0
    for key in summary['module']:
        num_params = 0
        for name in summary['module'][key]['params']:
            num_params += (summary['module'][key]['params'][name]['mask'] > 0).sum().item()
        summary['total_num_param'] += num_params
    summary['total_space_param'] = abs(summary['total_num_param'] * 32. / 8 / (1024 ** 2.))
    return summary


def parse_summary(summary):
    content = ''
    headers = ['Module Name', 'Input Size', 'Weight Size', 'Output Size', 'Number of Parameters']
    records = []
    for key in summary['module']:
        if 'weight' not in summary['module'][key]['params']:
            continue
        module_name = summary['module'][key]['module_name']
        input_size = str(summary['module'][key]['input_size'])
        weight_size = str(summary['module'][key]['params']['weight']['size']) if (
                'weight' in summary['module'][key]['params']) else 'N/A'
        output_size = str(summary['module'][key]['output_size'])
        num_params = 0
        for name in summary['module'][key]['params']:
            num_params += (summary['module'][key]['params'][name]['mask'] > 0).sum().item()
        records.append([module_name, input_size, weight_size, output_size, num_params])
    total_num_param = summary['total_num_param']
    total_space_param = summary['total_space_param']

    table = tabulate(records, headers=headers, tablefmt='github')
    content += table + '\n'
    content += '================================================================\n'
    content += 'Total Number of Parameters: {}\n'.format(total_num_param)
    content += 'Total Space of Parameters (MB): {:.2f}\n'.format(total_space_param)
    makedir_exist_ok('./output')
    content_file = open('./output/summary.md', 'w')
    content_file.write(content)
    content_file.close()
    return content


def process_control_name():
    control_name = config.PARAM['control_name'].split('_')
    config.PARAM['cell_name'] = control_name[0]
    config.PARAM['embedding_size'] = int(control_name[1])
    config.PARAM['hidden_size'] = int(control_name[2])
    config.PARAM['num_layer'] = int(control_name[3])
    config.PARAM['sharing_rates'] = float(control_name[4])
    config.PARAM['dropout'] = float(control_name[5])
    config.PARAM['tied'] = int(control_name[6])
    return


if __name__ == "__main__":
    main()
