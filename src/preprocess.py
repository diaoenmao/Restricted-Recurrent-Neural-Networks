import config
config.init()
import argparse
import datetime
import os
import time
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn as nn
import models
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau
from data import *
from metrics import *
from utils import *
from PIL import Image
from tqdm import tqdm

def collate(input):
    for k in input:
        input[k] = torch.stack(input[k],0)
    return input

def extract_patches_2d(img,patch_shape,step=(1.0,1.0),strict=False):
    patch_H, patch_W = patch_shape[0], patch_shape[1]
    if(img.size(2)<patch_H):
        if(strict):
            raise ValueError('img height smaller than patch height')
        else:
            num_padded_H_Top = (patch_H - img.size(2))//2
            num_padded_H_Bottom = patch_H - img.size(2) - num_padded_H_Top
            padding_H = nn.ConstantPad2d((0,0,num_padded_H_Top,num_padded_H_Bottom),0)
            img = padding_H(img)
    if(img.size(3)<patch_W):
        if(strict):
            raise ValueError('img width smaller than patch width')
        else:
            num_padded_W_Left = (patch_W - img.size(3))//2
            num_padded_W_Right = patch_W - img.size(3) - num_padded_W_Left
            padding_W = nn.ConstantPad2d((num_padded_W_Left,num_padded_W_Right,0,0),0)
            img = padding_W(img)
    step_int = [0,0]
    step_int[0] = int(patch_H*step[0]) if(isinstance(step[0], float)) else step[0]
    step_int[1] = int(patch_W*step[1]) if(isinstance(step[1], float)) else step[1]
    patches_fold_H = img.unfold(2, patch_H, step_int[0])
    if((img.size(2) - patch_H) % step_int[0] != 0):
        patches_fold_H = torch.cat((patches_fold_H,img[:,:,-patch_H:,].permute(0,1,3,2).unsqueeze(2)),dim=2)
    patches_fold_HW = patches_fold_H.unfold(3, patch_W, step_int[1])   
    if((img.size(3) - patch_W) % step_int[1] != 0):
        patches_fold_HW = torch.cat((patches_fold_HW,patches_fold_H[:,:,:,-patch_W:,:].permute(0,1,2,4,3).unsqueeze(3)),dim=3)
    patches = patches_fold_HW.permute(2,3,0,1,4,5)
    patches = patches.reshape(-1,img.size(0),img.size(1),patch_H,patch_W)
    patches = patches.transpose(0,1)
    return patches

if __name__ == '__main__':
    data_name = 'RAISE'
    data_dir = './data/{}'.format(data_name)
    makedir_exist_ok('./data/{}_patch/train'.format(data_name))
    radomGen = np.random.RandomState(1234)
    dataset = datasets.ImageFolder(data_dir)
    data_size = len(dataset)
    data_idx = radomGen.choice(list(range(len(dataset))), size=data_size, replace=False)
    dataset = torch.utils.data.Subset(dataset, data_idx)
    for i in tqdm(range(len(dataset))):
        transform = transforms.Compose([transforms.Resize((768,1152))])
        img = transform(dataset[i])['img']
        #img = dataset[i]['img']
        np_img = np.array(img)
        torch_img = torch.from_numpy(np_img).permute(2,0,1).unsqueeze(0)
        torch_patch = extract_patches_2d(torch_img,config.PARAM['img_size'],strict=True)[0]
        np_patch = torch_patch.permute(0,2,3,1).numpy()
        for j in range(len(np_patch)):
            im = Image.fromarray(np_patch[j])
            im.save('./data/{}_patch/train/{}_{}.png'.format(data_name,i,j))
            
            
            