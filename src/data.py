import config
import numpy as np
import os
import tarfile
import torch
import datasets
import datasets.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.distributed import DistributedSampler
from utils import *

dist = False
world_size = config.PARAM['world_size']
num_workers = config.PARAM['num_workers']
device = config.PARAM['device']

def fetch_dataset(data_name):
    dataset = {}
    print('fetching data {}...'.format(data_name))
    data_name_head = data_name.split('_')[0]
    if(data_name_head=='MNIST'):
        train_dir = './data/{}/train'.format(data_name)
        test_dir = './data/{}/test'.format(data_name)
        dataset['train'] = datasets.MNIST(root=train_dir, train=True, download=True, transform=transforms.ToTensor())
        dataset['test'] = datasets.MNIST(root=test_dir, train=False, download=True, transform=transforms.ToTensor())       
        config.PARAM['stats'] = make_stats(dataset['train'],batch_size=config.PARAM['batch_size']['train'])
        config.PARAM['transform'] = {'train':transforms.Compose([transforms.ToTensor()]),
                                     'test':transforms.Compose([transforms.ToTensor()])}
        dataset['train'].transform = config.PARAM['transform']['train']
        dataset['test'].transform = config.PARAM['transform']['test']       
    elif(data_name_head=='CIFAR10'):
    
        train_dir = './data/{}/train'.format(data_name)
        test_dir = './data/{}/validation'.format(data_name)
        dataset['train'] = datasets.CIFAR10(train_dir, train=True, transform=transforms.ToTensor(), download=True)
        dataset['test'] = datasets.CIFAR10(test_dir, train=False, transform=transforms.ToTensor(), download=True)       
        config.PARAM['stats'] = make_stats(dataset['train'],batch_size=config.PARAM['batch_size']['train'])
        config.PARAM['transform'] = {'train':transforms.Compose([transforms.RandomCrop(32, padding=4),
                                                                 transforms.RandomHorizontalFlip(),
                                                                 transforms.ToTensor()]),
                                     'test':transforms.Compose([transforms.ToTensor()])}       
        dataset['train'].transform = config.PARAM['transform']['train']
        dataset['test'].transform = config.PARAM['transform']['test']        
    elif(data_name_head=='SVHN'):
        train_dir = './data/{}/train'.format(data_name)
        test_dir = './data/{}/validation'.format(data_name)
        dataset['train'] = datasets.SVHN(train_dir, split='train', transform=transforms.ToTensor(), download=True)
        dataset['test'] = datasets.SVHN(test_dir, split='test', transform=test_transform, download=True)
        config.PARAM['stats'] = make_stats(dataset['train'],batch_size=config.PARAM['batch_size']['train'])
        config.PARAM['transform'] = {'train':transforms.Compose([transforms.ToTensor()]),
                                     'test':transforms.Compose([transforms.ToTensor()])}
        dataset['train'].transform = config.PARAM['transform']['train']
        dataset['test'].transform = config.PARAM['transform']['test']      
    elif(data_name_head=='ImageNet'):
        train_dir = './data/{}/train'.format(data_name)
        test_dir = './data/{}/validation'.format(data_name)
        dataset['train'] = datasets.ImageNet(train_dir, split='train', download=True, transform=transforms.ToTensor())
        dataset['test'] = datasets.ImageNet(test_dir, split='validation', download=True, transform=transforms.ToTensor())
        config.PARAM['stats'] = make_stats(dataset['train'],batch_size=128)
        config.PARAM['transform'] = {'train':transforms.Compose([transforms.Resize((128,128)),
                                                                 transforms.ToTensor()]),
                                     'test':transforms.Compose([transforms.Resize((128,128)),
                                                                transforms.ToTensor()])}         
        dataset['train'].transform = config.PARAM['transform']['train']
        dataset['test'].transform = config.PARAM['transform']['test']   
    elif(data_name_head=='miniImageNet'):
        train_dir = './data/{}/train'.format(data_name)
        validation_dir = './data/{}/validation'.format(data_name)
        test_dir = './data/{}/test'.format(data_name)
        dataset['train'] = datasets.ImageNet(train_dir, split='train', download=True, transform=transforms.Compose([transforms.Resize((128,128)),transforms.ToTensor()]))
        dataset['validation'] = datasets.ImageNet(validation_dir, split='validation', download=True, transform=transforms.ToTensor())
        dataset['test'] = datasets.ImageNet(test_dir, split='test', download=True, transform=transforms.ToTensor())
        config.PARAM['stats'] = make_stats(dataset['train'],batch_size=128)
        config.PARAM['transform'] = {'train':transforms.Compose([transforms.Resize((128,128)),
                                                                 transforms.ToTensor()]),
                                     'validation':transforms.Compose([transforms.Resize((128,128)),
                                                                 transforms.ToTensor()]),
                                     'test':transforms.Compose([transforms.Resize((128,128)),
                                                                transforms.ToTensor()])}   
        dataset['train'].transform = config.PARAM['transform']['train']
        dataset['validation'].transform = config.PARAM['transform']['validation']
        dataset['test'].transform = config.PARAM['transform']['test']
    elif(data_name_head =='Kodak'):
        train_dir = './data/{}/train'.format(data_name)
        test_dir = './data/{}/train'.format(data_name)
        dataset['train'] = datasets.ImageFolder(train_dir, transform=transforms.ToTensor())
        dataset['test'] = datasets.ImageFolder(test_dir, transform=transforms.ToTensor())
        dataset['train'].data_name,dataset['test'].data_name = 'Kodak','Kodak'
        config.PARAM['stats'] = make_stats(dataset['train'],batch_size=10)
        config.PARAM['transform'] = {'train':transforms.Compose([transforms.ToTensor()]),
                                     'test':transforms.Compose([transforms.ToTensor()])}  
        dataset['train'].transform = config.PARAM['transform']['train']
        dataset['test'].transform = config.PARAM['transform']['test']
    elif(data_name_head =='Flickr30k'):
        train_dir = './data/{}/train'.format(data_name)
        test_dir = './data/{}/train'.format(data_name)
        dataset['train'] = datasets.ImageFolder(train_dir, transform=transforms.ToTensor())
        dataset['test'] = datasets.ImageFolder(train_dir, transform=transforms.ToTensor())
        dataset['train'].data_name,dataset['test'].data_name = 'Flickr30k','Flickr30k'
        config.PARAM['stats'] = make_stats(dataset['train'],batch_size=128)
        config.PARAM['transform'] = {'train':transforms.Compose([transforms.ToTensor()]),
                                     'test':transforms.Compose([transforms.ToTensor()])}
        dataset['train'].transform = config.PARAM['transform']['train']
        dataset['test'].transform = config.PARAM['transform']['test'] 
    elif(data_name_head =='UCID'):
        train_dir = './data/{}/train'.format(data_name)
        test_dir = './data/{}/train'.format(data_name)
        dataset['train'] = datasets.ImageFolder(train_dir, transform=transforms.ToTensor())
        dataset['test'] = datasets.ImageFolder(train_dir, transform=transforms.ToTensor())
        dataset['train'].data_name,dataset['test'].data_name = 'UCID','UCID'
        config.PARAM['stats'] = make_stats(dataset['train'],batch_size=128)
        config.PARAM['transform'] = {'train':transforms.Compose([transforms.ToTensor()]),
                                     'test':transforms.Compose([transforms.ToTensor()])}
        dataset['train'].transform = config.PARAM['transform']['train']
        dataset['test'].transform = config.PARAM['transform']['test'] 
    elif(data_name_head =='RAISE'):
        train_dir = './data/{}/train'.format(data_name)
        test_dir = './data/{}/train'.format(data_name)
        dataset['train'] = datasets.ImageFolder(train_dir, transform=transforms.ToTensor())
        dataset['test'] = datasets.ImageFolder(train_dir, transform=transforms.ToTensor())
        dataset['train'].data_name,dataset['test'].data_name = 'RAISE','RAISE'
        config.PARAM['stats'] = make_stats(dataset['train'],batch_size=128)
        config.PARAM['transform'] = {'train':transforms.Compose([transforms.ToTensor()]),
                                     'test':transforms.Compose([transforms.ToTensor()])}
        dataset['train'].transform = config.PARAM['transform']['train']
        dataset['test'].transform = config.PARAM['transform']['test'] 
    elif(data_name =='BITS'):
        dataset['train'] = datasets.BITS(train=True)
        dataset['test'] = datasets.BITS(train=False)
    else:
        raise ValueError('Not valid dataset name')

    print('data ready')
    return dataset

def input_collate(batch):
    if(isinstance(batch[0], dict)):
        output = {key: [] for key in batch[0].keys()}
        for b in batch:
            for key in b:
                output[key].append(b[key])
        return output
    else:
        return default_collate(batch)
                
def split_dataset(dataset,data_size,batch_size,radomGen=np.random.RandomState(1234),shuffle={'train':True,'test':False},collate_fn=input_collate):
    data_loader = {}
    for k in dataset:
        if(k in data_size):
            data_size[k] = len(dataset[k]) if (data_size[k]>len(dataset[k])) else data_size[k]
            data_size[k] = len(dataset[k]) if (data_size[k]==0) else data_size[k] 
            batch_size[k] = data_size[k] if (batch_size[k]==0) else batch_size[k]            
            data_idx_k = radomGen.choice(list(range(len(dataset[k]))), size=data_size[k], replace=False)
            dataset_k = torch.utils.data.Subset(dataset[k], data_idx_k)     
            data_loader[k] = torch.utils.data.DataLoader(dataset=dataset_k,
                        shuffle=shuffle[k], batch_size=batch_size[k], pin_memory=True, sampler=None, num_workers=num_workers, collate_fn=collate_fn)    
    return data_loader

def make_stats(dataset,reuse=True,batch_size=1000):
    if(reuse and os.path.exists('./data/stats/{}.pkl'.format(dataset.data_name))):
        stats = load('./data/stats/{}.pkl'.format(dataset.data_name))
    elif(dataset is not None):
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        stats = {}
        for k in dataset.feature_dim:
            stats[k] = Stats(dataset.feature_dim[k])
        print('Computing mean and std...')
        with torch.no_grad():
            for input in data_loader:
                for k in dataset.feature_dim:
                    stats[k].update(input[k])
        save(stats,'./data/stats/{}.pkl'.format(dataset.data_name))
    else:
        raise ValueError('Please provide dataset for making stats')
    for k in dataset.feature_dim:
        print('[{}] mean: {}, std: {}'.format(k,stats[k].mean,stats[k].std))
    return stats
    
class Stats(object):
    def __init__(self, feature_dim):
        self.feature_dim = feature_dim
        self.n_samples = 0

    def update(self, data):
        collapse_data = data.transpose(self.feature_dim,-1).reshape(-1,data.size(self.feature_dim))
        if self.n_samples == 0:
            self.n_samples = collapse_data.size(0)
            self.n_features = collapse_data.size(1)
            self.mean = collapse_data.mean(dim=0)
            self.std = collapse_data.std(dim=0)
        else:
            if collapse_data.size(1) != self.n_features:
                raise ValueError("data dims don't match prev observations.")
            m = float(self.n_samples)
            n = collapse_data.size(0)
            new_mean = collapse_data.mean(dim=0)
            new_std = new_mean.new_zeros(new_mean.size()) if(n==1) else collapse_data.std(dim=0)
            old_mean = self.mean
            old_std = self.std
            self.mean = m/(m+n)*old_mean + n/(m+n)*new_mean
            self.std = torch.sqrt(m/(m+n)*old_std**2 + n/(m+n)*new_std**2 + m*n/(m+n)**2 * (old_mean - new_mean)**2)
            self.n_samples += n
   