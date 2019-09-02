import config

config.init()
import argparse
import datetime
import models
import time
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau, CosineAnnealingLR
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


def main():
    process_control_name()
    seeds = list(range(config.PARAM['init_seed'], config.PARAM['init_seed'] + config.PARAM['num_Experiments']))
    for i in range(config.PARAM['num_Experiments']):
        model_TAG = '{}_{}_{}_{}'.format(seeds[i], config.PARAM['data_name']['train'], config.PARAM['model_name'],
            config.PARAM['control_name']) if (config.PARAM['special_TAG'] == '') else '{}_{}_{}_{}_{}'.format(seeds[i],
            config.PARAM['data_name']['train'], config.PARAM['model_name'], config.PARAM['control_name'],
            config.PARAM['special_TAG'])
        print('Experiment: {}'.format(model_TAG))
        runExperiment(model_TAG)
    return


def runExperiment(model_TAG):
    model_TAG_list = model_TAG.split('_')
    seed = int(model_TAG_list[0])
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    config.PARAM['randomGen'] = np.random.RandomState(seed)
    dataset = fetch_dataset(data_name=config.PARAM['data_name']['train'])
    config.PARAM['vocab'] = dataset['train'].vocab
    batch_dataset = {}
    print(config.PARAM)
    for split in dataset['train'].data_file:
        batch_dataset[split] = batchify(dataset[split].data, config.PARAM['batch_size'][split])
        print('{} data size {}, Number of Batches {}'.format(split, len(dataset[split].data),
            len(dataset[split].data) // config.PARAM['batch_size'][split]))
    model = eval('models.{}().to(device)'.format(config.PARAM['model_name']))
    optimizer = make_optimizer(model)
    scheduler = make_scheduler(optimizer)
    if config.PARAM['resume_mode'] == 1:
        last_epoch, model, optimizer, scheduler, meter_panel = resume(model, optimizer, scheduler, model_TAG)
    elif config.PARAM['resume_mode'] == 2:
        last_epoch = 0
        _, model, _, _, _ = resume(model, optimizer, scheduler, model_TAG)
        meter_panel = {'train': Meter_Panel(config.PARAM['metric_names']['train']),
            'validation': test(batch_dataset['validation'], model, 'validation'),
            'test': test(batch_dataset['test'], model, 'test')}
    else:
        last_epoch = 0
        meter_panel = {'train': Meter_Panel(config.PARAM['metric_names']['train']),
            'validation': test(batch_dataset['validation'], model, 'validation'),
            'test': test(batch_dataset['test'], model, 'test')}
    model = nn.DataParallel(model, device_ids=list(range(config.PARAM['world_size']))) if (
            config.PARAM['world_size'] > 1) else model
    pivot_split = 'validation'
    best_pivot_name = 'loss'
    best_pivot = 65535
    print('***Test Epoch({}): {}'.format(model_TAG,
        meter_panel['test'].summary(['loss'] + config.PARAM['metric_names']['test'])))
    for epoch in range(last_epoch, config.PARAM['num_epochs'] + 1):
        meter_panel['train'].update(train(batch_dataset['train'], model, optimizer, epoch, model_TAG))
        meter_panel['validation'].update(test(batch_dataset['validation'], model, 'validation'))
        meter_panel['test'].update(test(batch_dataset['test'], model, 'test'))
        print_result(model_TAG, epoch, meter_panel, optimizer.param_groups[0]['lr'])
        scheduler.step(metrics=meter_panel[pivot_split].panel['loss'].history_avg[-1], epoch=epoch + 1) if (
                config.PARAM['scheduler_name'] == 'ReduceLROnPlateau') else scheduler.step(epoch=epoch + 1)
        if config.PARAM['save_mode'] >= 0:
            model_state_dict = model.module.state_dict() if (config.PARAM['world_size'] > 1) else model.state_dict()
            save_result = {'config': config.PARAM, 'epoch': epoch + 1, 'model_dict': model_state_dict,
                'optimizer_dict': optimizer.state_dict(), 'scheduler_dict': scheduler.state_dict(),
                'meter_panel': meter_panel}
            save(save_result, './output/model/{}_checkpoint.pkl'.format(model_TAG))
            if best_pivot > meter_panel[pivot_split].panel[best_pivot_name].history_avg[-1]:
                best_pivot = meter_panel[pivot_split].panel[best_pivot_name].history_avg[-1]
                save(save_result, './output/model/{}_best.pkl'.format(model_TAG))
    return


def train(dataset, model, optimizer, epoch, model_TAG):
    meter_panel = Meter_Panel(config.PARAM['metric_names']['train'])
    model.train(True)
    end = time.time()
    bptt_range = range(0, dataset.size(1) - 1, config.PARAM['bptt'])
    total_loss = 0
    for i, idx in enumerate(bptt_range):
        input = make_batch(dataset, idx, config.PARAM['bptt'])
        input = dict_to_device(input, device)
        model.zero_grad()
        output = model(input)
        output['loss'].backward()
        total_loss += output['loss'].item()
        if config.PARAM['clip']:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.PARAM['clip'])
        optimizer.step()
        evaluation = meter_panel.eval(input, output, config.PARAM['metric_names']['train'])
        batch_time = time.time() - end
        meter_panel.update(evaluation, input['line'].size(1))
        meter_panel.update({'batch_time': batch_time})
        end = time.time()
        if i % (len(bptt_range) // 5) == 0 and i > 0:
            if 'perplexity' in config.PARAM['metric_names']['train']:
                meter_panel.update({'perplexity': math.exp(total_loss / (len(bptt_range) // 5))})
            total_loss = 0
            estimated_finish_time = str(datetime.timedelta(seconds=(len(bptt_range) - i - 1) * batch_time))
            print('Train Epoch({}): {}[({:.0f}%)]{}, Estimated Finish Time: {}'.format(model_TAG, epoch,
                100. * i / len(bptt_range),
                meter_panel.summary(['loss', 'batch_time'] + config.PARAM['metric_names']['train']),
                estimated_finish_time))
    apply_fn(model, 'free_hidden')
    return meter_panel


def test(dataset, model, split):
    meter_panel = Meter_Panel(config.PARAM['metric_names'][split])
    bptt_range = range(0, dataset.size(1) - 1, config.PARAM['bptt'])
    total_loss = 0
    with torch.no_grad():
        model.train(False)
        end = time.time()
        for i, idx in enumerate(bptt_range):
            input = make_batch(dataset, idx, config.PARAM['bptt'])
            input = dict_to_device(input, device)
            output = model(input)
            total_loss += output['loss'].item()
            evaluation = meter_panel.eval(input, output, config.PARAM['metric_names'][split])
            batch_time = time.time() - end
            meter_panel.update(evaluation, input['line'].size(1))
            meter_panel.update({'batch_time': batch_time})
            end = time.time()
        apply_fn(model, 'free_hidden')
    if 'perplexity' in config.PARAM['metric_names'][split]:
        meter_panel.update({'perplexity': math.exp(total_loss / len(bptt_range))})
    return meter_panel


def make_optimizer(model):
    if config.PARAM['optimizer_name'] == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=config.PARAM['lr'], weight_decay=config.PARAM['weight_decay'])
    elif config.PARAM['optimizer_name'] == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=config.PARAM['lr'], momentum=0.9,
            weight_decay=config.PARAM['weight_decay'])
    elif config.PARAM['optimizer_name'] == 'ASGD':
        optimizer = optim.ASGD(model.parameters(), lr=config.PARAM['lr'], weight_decay=config.PARAM['weight_decay'])
    else:
        raise ValueError('Not valid optimizer name')
    return optimizer


def make_scheduler(optimizer):
    if config.PARAM['scheduler_name'] == 'MultiStepLR':
        scheduler = MultiStepLR(optimizer, milestones=config.PARAM['milestones'], gamma=config.PARAM['factor'])
    elif config.PARAM['scheduler_name'] == 'ReduceLROnPlateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=config.PARAM['factor'], patience=config.PARAM['patience'], verbose=True,
            threshold=config.PARAM['threshold'], threshold_mode=config.PARAM['threshold_mode'])
    elif config.PARAM['scheduler_name'] == 'CosineAnnealingLR':
        scheduler = CosineAnnealingLR(optimizer, config.PARAM['num_epochs'])
    elif config.PARAM['scheduler_name'] == 'none':
        scheduler = MultiStepLR(optimizer, milestones=[65535], gamma=config.PARAM['factor'])
    else:
        raise ValueError('Not valid scheduler name')
    return scheduler


def print_result(model_TAG, epoch, meter_panel, lr):
    estimated_finish_time = str(
        datetime.timedelta(seconds=(config.PARAM['num_epochs'] - epoch) * meter_panel['train'].panel['batch_time'].sum))
    print('***Test Epoch({}): {}{}, Estimated Finish Time: {}, Learning Rate: {}'.format(model_TAG, epoch,
        meter_panel['test'].summary(['loss'] + config.PARAM['metric_names']['test']), estimated_finish_time, lr))
    return


def resume(model, optimizer, scheduler, model_TAG):
    if os.path.exists('./output/model/{}_checkpoint.pkl'.format(model_TAG)):
        checkpoint = load('./output/model/{}_checkpoint.pkl'.format(model_TAG))
        last_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_dict'])
        meter_panel = checkpoint['meter_panel']
        print('Resume from {}'.format(last_epoch))
        return last_epoch, model, optimizer, scheduler, meter_panel
    else:
        last_epoch = 0
        print('Not found existing model, and start from epoch {}'.format(last_epoch))
    return


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
