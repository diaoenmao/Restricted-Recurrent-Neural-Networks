import config
config.init()
import argparse
import itertools
import torch.backends.cudnn as cudnn
from metrics import *
from utils import *

cudnn.benchmark = True
parser = argparse.ArgumentParser(description='Config')
for k in config.PARAM:
    exec('parser.add_argument(\'--{0}\',default=config.PARAM[\'{0}\'], help=\'\')'.format(k))
args = vars(parser.parse_args())
for k in config.PARAM:
    if (config.PARAM[k] != args[k]):
        exec('config.PARAM[\'{0}\'] = {1}'.format(k, args[k]))
path = 'milestones_0/03'
show_label = {'BITS_smartcode_cnn_awgn_100_3_4': 'SmartCodeCNN(k=100, n=300, L=2)',
              'BITS_smartcode_rnn_awgn_100_3_4': 'SmartCodeRNN(k=100, n=300, L=2)'}

def main():
    control_names = [['0'], [config.PARAM['data_name']['train']], ['smartcode_cnn', 'smartcode_rnn'], ['awgn'],
                     ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], ['100'], ['3'], ['4']]
    extract_result(control_names)
    gather_result(control_names)
    process_result(control_names)
    show_result(control_names)


def extract_result(control_names):
    control_names_product = list(itertools.product(*control_names))
    head = './output/model/{}'.format(path)
    tail = 'best'
    for i in range(len(control_names_product)):
        control_name = list(control_names_product[i])
        model_TAG = '_'.join(control_name)
        model_path = '{}/{}_{}.pkl'.format(head, model_TAG, tail)
        if (os.path.exists(model_path)):
            result = load(model_path)
            save({'train_meter_panel': result['train_meter_panel'], 'test_meter_panel': result['test_meter_panel']},
                 './output/result/{}/{}.pkl'.format(path, model_TAG))
        else:
            print('model path {} not exist'.format(model_path))
    return


def gather_result(control_names):
    control_names_product = list(itertools.product(*control_names))
    gathered_result = {}
    head = './output/result/{}'.format(path)
    for i in range(len(control_names_product)):
        control_name = list(control_names_product[i])
        model_TAG = '_'.join(control_name)
        result_path = '{}/{}.pkl'.format(head, model_TAG)
        if (os.path.exists(result_path)):
            result = load(result_path)
            gathered_result[model_TAG] = {
                'SNR(dB)': float(control_name[4]),
                'loss': result['test_meter_panel'].panel['loss'].history_avg[-1],
                'BER': result['test_meter_panel'].panel['ber'].history_avg[-1]}
        else:
            print('result path {} not exist'.format(result_path))
    print(gathered_result)
    save(gathered_result, './output/result/{}/gathered_result.pkl'.format(path))
    return


def process_result(control_names):
    control_names_product = list(itertools.product(*control_names))
    control_size = [len(control_names[i]) for i in range(len(control_names))]
    result_path = './output/result/{}/gathered_result.pkl'.format(path)
    result = load(result_path)
    evaluation_names = ['SNR(dB)', 'loss', 'BER']
    processed_result = {'indices': {},
                        'all': {k: torch.zeros(control_size, device=config.PARAM['device']) for k in evaluation_names},
                        'mean': {}, 'stderr': {}}
    for i in range(len(control_names_product)):
        control_name = list(control_names_product[i])
        model_TAG = '_'.join(control_name)
        if (model_TAG not in result):
            continue
        processed_result['indices'][model_TAG] = []
        for i in range(len(control_names)):
            processed_result['indices'][model_TAG].append(control_names[i].index(control_name[i]))
        for k in processed_result['all']:
            processed_result['all'][k][tuple(processed_result['indices'][model_TAG])] = result[model_TAG][k]
    for k in evaluation_names:
        processed_result['mean'][k] = processed_result['all'][k].mean(dim=0, keepdim=True)
    for k in evaluation_names:
        processed_result['stderr'][k] = processed_result['all'][k].std(dim=0, keepdim=True) / math.sqrt(
            processed_result['all'][k].size(0))
    save(processed_result, './output/result/{}/processed_result.pkl'.format(path))
    return


def show_result(control_names):
    control_names_product = list(itertools.product(*control_names[1:]))
    fig_format = 'png'
    result_path = './output/result/{}/processed_result.pkl'.format(path)
    result = load(result_path)
    x_name = 'SNR(dB)'
    y_name = 'BER'
    baseline_name = 'Turbo757'
    control_index = 3
    num_stderr = 1
    band = True
    x, y, y_min, y_max = {}, {}, {}, {}
    for i in range(len(control_names_product)):
        control_name = list(control_names_product[i])
        model_TAG = '0_'+'_'.join(control_name)
        label = control_name.copy()
        label.pop(control_index)
        label = '_'.join(label)
        if (label not in x):
            x[label], y[label], y_min[label], y_max[label] = [], [], [], []
        idx = tuple(result['indices'][model_TAG])
        x[label].append(result['mean'][x_name][idx].item())
        y[label].append(result['mean'][y_name][idx].item())
        y_min[label].append(y[label][-1] - num_stderr * result['stderr'][y_name][idx].item())
        y_max[label].append(y[label][-1] + num_stderr * result['stderr'][y_name][idx].item())
    colors = plt.get_cmap('rainbow')
    colors_indices = np.linspace(0.2, 1, len(x)).tolist()
    fig = plt.figure()
    fontsize = 14
    k = 0
    for label in x:
        color = colors(colors_indices[k])
        plt.plot(x[label], y[label], color=color, linestyle='-', label=show_label[label], linewidth=3)
        if (band):
            plt.fill_between(x[label], y_max[label], y_min[label], color=color, alpha=0.5, linewidth=1)
        k = k + 1
    baseline = np.genfromtxt('./output/result/{}/{}.csv'.format(path,baseline_name), delimiter=',').T
    plt.plot(baseline[0], baseline[1], color='black', linestyle='-', label=baseline_name, linewidth=3)
    plt.xlabel(x_name, fontsize=fontsize)
    plt.ylabel(y_name, fontsize=fontsize)
    plt.yscale('log')
    plt.ylim([10 ** (-7), 10 ** 0])
    plt.grid()
    plt.legend()
    plt.show()
    makedir_exist_ok('./output/fig/{}'.format(path))
    fig.savefig('./output/fig/{}/result.{}'.format(path, fig_format), bbox_inches='tight', pad_inches=0)
    plt.close()
    return


if __name__ == '__main__':
    main()