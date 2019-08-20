import config

config.init()
import argparse
import itertools
import models
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
from data import *
from metrics import *
from utils import *
from summary import summarize
from matplotlib.ticker import FuncFormatter

cudnn.benchmark = True
parser = argparse.ArgumentParser(description='Config')
for k in config.PARAM:
    exec('parser.add_argument(\'--{0}\',default=config.PARAM[\'{0}\'], help=\'\')'.format(k))
args = vars(parser.parse_args())
for k in config.PARAM:
    if (config.PARAM[k] != args[k]):
        exec('config.PARAM[\'{0}\'] = {1}'.format(k, args[k]))
data_name = 'PennTreebank'
path = 'milestones_1/{}/0'.format(data_name)
dataset = fetch_dataset(data_name=data_name)
batch_dataset = batchify(dataset.data['train'], config.PARAM['batch_size']['train'])
config.PARAM['vocab'] = dataset.vocab
cell_name = ['RNN', 'LSTM', 'GRU']


def main():
    result_control = {'RNN': {
        'evaluation_names': ['number of parameters (all)', 'number of parameters (rnn)', 'sharing_rates',
            'perplexity_test', 'perplexity_validation'],
        'control_names': [['0'], [data_name], ['basic'], ['RNN'], ['200'], ['200'], ['5'],
            ['0', '0.1', '0.3', '0.5', '0.7', '0.9', '0.95', '1'], ['0', '0.2'], ['0', '1']], 'control_index': 7},
        'GRU': {'evaluation_names': ['number of parameters (all)', 'number of parameters (rnn)', 'sharing_rates',
            'perplexity_test', 'perplexity_validation'],
            'control_names': [['0'], [data_name], ['basic'], ['GRU'], ['200'], ['200'], ['5'],
                ['0', '0.1', '0.3', '0.5', '0.7', '0.9', '0.95', '1'], ['0', '0.2'], ['0', '1']], 'control_index': 7},
        'LSTM': {'evaluation_names': ['number of parameters (all)', 'number of parameters (rnn)', 'sharing_rates',
            'perplexity_test', 'perplexity_validation'],
            'control_names': [['0'], [data_name], ['basic'], ['LSTM'], ['200'], ['200'], ['5'],
                ['0', '0.1', '0.3', '0.5', '0.7', '0.9', '0.95', '1'], ['0', '0.2'], ['0', '1']], 'control_index': 7}}
    # for result_names, info in result_control.items():
    #     extract_result(info['control_names'])
    #     gather_result(result_names, info)
    #     process_result(result_names, info)
    show_result(result_control)
    return


def compute_num_param(model_TAG):
    control_name = model_TAG.split('_')[3:]
    config.PARAM['cell_name'] = control_name[0]
    config.PARAM['embedding_size'] = int(control_name[1])
    config.PARAM['hidden_size'] = int(control_name[2])
    config.PARAM['num_layer'] = int(control_name[3])
    config.PARAM['sharing_rates'] = float(control_name[4])
    config.PARAM['dropout'] = float(control_name[5])
    config.PARAM['tied'] = int(control_name[6])
    model = eval('models.{}().to(device)'.format(config.PARAM['model_name']))
    summary = summarize(batch_dataset, model)
    total_num_param = summary['total_num_param']
    all = total_num_param - (len(config.PARAM['vocab']) * config.PARAM['embedding_size']) if config.PARAM[
        'tied'] else total_num_param
    rnn = total_num_param - (len(config.PARAM['vocab']) * config.PARAM['embedding_size']) - (
            (len(config.PARAM['vocab']) + 1) * config.PARAM['embedding_size'])
    num_param = {'all': all, 'rnn': rnn}
    return num_param


def extract_result(control_names):
    control_names_product = list(itertools.product(*control_names))
    head = './output/model/{}'.format(path)
    tail = 'checkpoint'
    for i in range(len(control_names_product)):
        control_name = list(control_names_product[i])
        model_TAG = '_'.join(control_name)
        model_path = '{}/{}_{}.pkl'.format(head, model_TAG, tail)
        if os.path.exists(model_path):
            result = load(model_path)
            save(result['meter_panel'], './output/result/{}/{}.pkl'.format(path, model_TAG))
        else:
            print('model path {} not exist'.format(model_path))
    return


def gather_result(result_names, info):
    control_names_product = list(itertools.product(*info['control_names']))
    gathered_result = {}
    head = './output/result/{}'.format(path)
    for i in range(len(control_names_product)):
        control_name = list(control_names_product[i])
        model_TAG = '_'.join(control_name)
        result_path = '{}/{}.pkl'.format(head, model_TAG)
        if os.path.exists(result_path):
            result = load(result_path)
            gathered_result[model_TAG] = {}
            num_param = compute_num_param(model_TAG)
            gathered_result[model_TAG]['number of parameters (all)'], gathered_result[model_TAG][
                'number of parameters (rnn)'] = num_param['all'], num_param['rnn']
            gathered_result[model_TAG]['sharing_rates'] = config.PARAM['sharing_rates']
            gathered_result[model_TAG]['perplexity_test'] = min(result['test'].panel['perplexity'].history_avg)
            gathered_result[model_TAG]['perplexity_validation'] = min( result['validation'].panel['perplexity'].history_avg)
        else:
            print('result path {} not exist'.format(result_path))
    print(gathered_result)
    save(gathered_result, './output/result/{}/gathered_result_{}.pkl'.format(path, result_names))
    return


def process_result(result_names, info):
    control_names_product = list(itertools.product(*info['control_names']))
    control_size = [len(info['control_names'][i]) for i in range(len(info['control_names']))]
    result_path = './output/result/{}/gathered_result_{}.pkl'.format(path, result_names)
    result = load(result_path)
    all_result = {'indices': {},
        'all': {k: torch.zeros(control_size, device=config.PARAM['device']) for k in info['evaluation_names']},
        'mean': {}, 'stderr': {}}
    for i in range(len(control_names_product)):
        control_name = list(control_names_product[i])
        model_TAG = '_'.join(control_name)
        if model_TAG not in result:
            continue
        all_result['indices'][model_TAG] = []
        for i in range(len(info['control_names'])):
            all_result['indices'][model_TAG].append(info['control_names'][i].index(control_name[i]))
        for k in all_result['all']:
            all_result['all'][k][tuple(all_result['indices'][model_TAG])] = result[model_TAG][k]
    for k in info['evaluation_names']:
        all_result['mean'][k] = all_result['all'][k].mean(dim=0, keepdim=True)
    for k in info['evaluation_names']:
        all_result['stderr'][k] = all_result['all'][k].std(dim=0, keepdim=True) / math.sqrt(
            all_result['all'][k].size(0))

    processed_result = {k: {} for k in info['evaluation_names']}
    for i in range(len(control_names_product)):
        control_name = list(control_names_product[i])
        model_TAG = '_'.join(control_name)
        if model_TAG not in all_result['indices']:
            continue
        idx = tuple(all_result['indices'][model_TAG])
        label = control_name.copy()
        label.pop(info['control_index'])
        label.pop(0)
        label = '_'.join(label)
        for k in processed_result:
            if label not in processed_result[k]:
                processed_result[k][label] = {'mean': [], 'stderr': []}
            processed_result[k][label]['mean'].append(all_result['mean'][k][idx].item())
            processed_result[k][label]['stderr'].append(all_result['mean'][k][idx].item())
    print(processed_result)
    save(processed_result, './output/result/{}/processed_result_{}.pkl'.format(path, result_names))
    return


def show_result(result_control):
    fig_format = 'png'
    x_names = 'number of parameters (rnn)'
    y_names = ['perplexity_test', 'perplexity_validation']
    # colors = plt.get_cmap('rainbow')
    # colors_indices = np.linspace(0.2, 1, len(result_control) * 2).tolist()
    colors = ['orange','red','deepskyblue','blue','lime','green']
    num_stderr = 0
    fontsize = 20
    k = 0
    m = 0
    fig = plt.figure()
    plt.rc('xtick', labelsize=fontsize - 8)
    plt.rc('ytick', labelsize=fontsize - 8)
    plt.grid()

    def millions(x, pos):
        return '%1.2fM' % (x * 1e-6)

    formatter = FuncFormatter(millions)
    for result_names, info in result_control.items():
        result_path = './output/result/{}/processed_result_{}.pkl'.format(path, result_names)
        result = load(result_path)
        for label in result[x_names]:
            list_label = label.split('_')
            pos = 1 if list_label[-1] == '1' else 0
            ax = plt.subplot(2, 3, pos * 3 + k + 1)
            ax.xaxis.set_major_formatter(formatter)
            for i in range(len(y_names)):
                split = y_names[i].split('_')[-1]
                line_style = '-' if split == 'test' else '--'
                plt.plot(result[x_names][label]['mean'], result[y_names[i]][label]['mean'],
                    color=colors[m], linestyle=line_style, label=map_label(label, y_names[i]),
                    linewidth=3)
                if num_stderr > 0:
                    plt.fill_between(result[x_names][label]['mean'],
                        result[y_names[i]][label]['mean'] + num_stderr * result[y_names[i]][label]['stderr'],
                        result[y_names[i]][label]['mean'] - num_stderr * result[y_names[i]][label]['stderr'],
                        color=colors[m], alpha=0.5, linewidth=1)
            plt.grid()
            plt.legend(fontsize=fontsize-8)
            m = m + 1
        k = k + 1
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    plt.xlabel('Number of RNN parameters', fontsize=fontsize)
    plt.ylabel('Perplexity', fontsize=fontsize)
    plt.show()
    makedir_exist_ok('./output/fig/{}'.format(path))
    fig.savefig('./output/fig/{}/{}.{}'.format(path, data_name, fig_format), dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()
    return


def map_label(label, y_names):
    list_label = label.split('_')
    list_y_names = y_names.split('_')
    show_label = []
    show_label += [list_label[2]]
    if list_label[-1] == '1':
        show_label[-1] += '-Tied-Dropout'
    if list_y_names[-1] == 'test':
        show_label[-1] += ' (Test)'
    else:
        show_label[-1] += ' (Validation)'
    show_label = ' '.join(show_label)
    return show_label


if __name__ == '__main__':
    main()
