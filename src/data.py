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
    
def split_dataset_cross_validation(train_dataset,test_dataset,data_size,batch_size,num_fold,radomGen,p=0.8):
    indices = list(range(len(train_dataset)))
    data_idx = radomGen.choice(indices, size=data_size, replace=False)
    if(batch_size==0):
        batch_size = len(train_idx)
    else:
        batch_size = batch_size*world_size
    if(num_fold==1):
        train_idx = radomGen.choice(data_idx, size=int(data_size*p), replace=False)
        sub_train_dataset = torch.utils.data.Subset(train_dataset, train_idx)
        train_sampler = DistributedSampler(sub_train_dataset) if (world_size > 1 and dist) else None
        train_loader = [torch.utils.data.DataLoader(dataset=sub_train_dataset, 
                    shuffle=(train_sampler is None), batch_size=batch_size, pin_memory=True, sampler=train_sampler, num_workers=num_workers*world_size)]   
        validation_idx = list(set(data_idx) - set(train_idx))
        validation_dataset = torch.utils.data.Subset(train_dataset, validation_idx)
        validation_sampler = DistributedSampler(validation_dataset) if (world_size > 1 and dist) else None
        validation_loader = [torch.utils.data.DataLoader(dataset=validation_dataset, 
                    batch_size=batch_size, pin_memory=True, sampler=validation_sampler, num_workers=num_workers*world_size)]
    elif(num_fold>1 and num_fold<=len(indices)):
        splitted_idx = np.array_split(data_idx, num_fold)
        train_loader = []
        validation_loader = []
        for i in range(num_fold):
            validation_idx = splitted_idx[i]
            train_idx = list(set(data_idx) - set(validation_idx))
            cur_train_dataset = torch.utils.data.Subset(train_dataset, train_idx)
            cur_train_sampler = DistributedSampler(cur_train_dataset) if (world_size > 1 and dist) else None
            train_loader.append(torch.utils.data.DataLoader(dataset=cur_train_dataset, 
                shuffle=(cur_train_sampler is None), batch_size=batch_size, pin_memory=True, sampler=cur_train_sampler, num_workers=num_workers*world_size)) 
            validation_dataset = torch.utils.data.Subset(train_dataset, validation_idx)
            validation_sampler = DistributedSampler(validation_dataset) if (world_size > 1 and dist) else None
            validation_loader.append(torch.utils.data.DataLoader(dataset=train_dataset, 
                batch_size=batch_size, pin_memory=True, sampler=validation_sampler, num_workers=num_workers*world_size))
    else:
        error("Invalid number of fold")
        exit()
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                batch_size=batch_size, pin_memory=True, num_workers=num_workers*world_size)
    return train_loader,validation_loader,test_loader
    
def fetch_dataset_synth(input_feature,output_feature,high_dim=None,cov_mode='base',noise_sigma=np.sqrt(0.1),randomGen=np.random.RandomState(1234)):
    print('fetching data...')
    data_size = 50000
    test_size = 10000
    V = make_cov_mat(input_feature,cov_mode)
    X = randomGen.multivariate_normal(np.zeros(input_feature),V,data_size+test_size)
    if(high_dim is None):
            beta = randomGen.randn(input_feature,output_feature)           
    else:
        if(high_dim>=input_feature):
            raise ValueError('invalid high dimension')
        valid_beta = randomGen.randn(high_dim,output_feature)
        empty_beta = np.zeros((input_feature-high_dim,output_feature))
        beta = np.vstack((valid_beta,empty_beta))
    mu = np.matmul(X,beta)
    eps = noise_sigma*randomGen.randn(*mu.shape)
    if(output_feature==1):
        y = mu + eps
    elif(output_feature>1):      
        p = softmax(mu + eps)
        y = []
        for i in range(X.shape[0]):
            sample = randomGen.multinomial(1,p[i,])
            y.append(np.where(sample==1)[0][0])
        y = np.array(y)
    else:
        raise ValueError('invalid dimension')
    print('data ready')
    X,y = X.astype(np.float32),y.astype(np.int64)
    train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(X[:data_size,:]), torch.from_numpy(y[:data_size]))
    test_dataset = torch.utils.data.TensorDataset(torch.from_numpy(X[data_size:,:]), torch.from_numpy(y[data_size:]))
    return train_dataset,test_dataset
    
def make_cov_mat(dim,mode,zo=0.5):
    if(mode=='base'):
        V = np.eye(dim)
    elif(mode=='corr'):
        V = np.full((dim, dim), zo)
        V = V + (1-zo)*np.eye(dim)
    elif(mode=='decay_corr'):
        indices = np.arange(dim)
        valid_indices = [indices,indices]
        mesh_indices = np.meshgrid(*valid_indices, sparse=False, indexing='ij')
        exponent = np.abs(mesh_indices[0]-mesh_indices[1])
        V = np.power(zo,exponent)
    else:
        raise ValueError('invalid covariance mode')
    return V

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
   