import torch
import torch.nn as nn
import torchvision.transforms.functional as F
from PIL import Image

def to_tensor(img):
    return F.to_tensor(img)

def normalize(input,stats):
    for k in input:
        if(k != 'label'):
            feature_dim = stats[k].feature_dim-1
            broadcast_size = [1]*input[k].dim()
            broadcast_size[feature_dim] = input[k].size(feature_dim)
            m,s = stats[k].mean.view(broadcast_size),stats[k].std.view(broadcast_size)
            input[k].sub_(m).div_(s)
    return input           
    
def resize(img, shape, interpolation):
    W, H = img.size
    img = F.resize(img, shape, interpolation)
    return img

def hflip(img):
    return img.transpose(Image.FLIP_LEFT_RIGHT)
    
def hflip_bbox(bbox,shape):
    H, _ = shape
    H_max = H - bbox[:, 0]
    H_min = H - bbox[:, 2]
    bbox[:, 0] = H_min
    bbox[:, 2] = H_max
    return bbox
    
def vflip(img):
    return img.transpose(Image.FLIP_TOP_BOTTOM)
    
def vflip_bbox(bbox,shape):
    _, W = shape
    W_max = W - bbox[:, 1]
    W_min = W - bbox[:, 3]
    bbox[:, 1] = W_min
    bbox[:, 3] = W_max
    return bbox
    
def fraction_resize(img, max_shape, interpolation):
    W, H = img.size
    max_H, max_W = max_shape
    scale_H = max_H // H
    scale_W = max_W // W
    if(scale_H >= scale_W):
        img = F.resize(img, (H*scale_W, max_W), interpolation)
    else:
        img = F.resize(img, (max_H, W*scale_H), interpolation)
    return img
        
def bbox_resize(bbox, input_shape, output_shape):
    input_W, input_H = input_shape
    output_W, output_H = output_shape
    H_scale = float(output_H) / input_H
    W_scale = float(output_W) / input_W
    bbox[:, 0] = H_scale * bbox[:, 0]
    bbox[:, 2] = H_scale * bbox[:, 2]
    bbox[:, 1] = W_scale * bbox[:, 1]
    bbox[:, 3] = W_scale * bbox[:, 3]
    return bbox
    
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

def reconstruct_from_patches_2d(patches,img_shape,step=(1.0,1.0),strict=False):
    patches = patches.transpose(0,1)
    patch_H, patch_W = patches.size(3), patches.size(4)
    img_size = (patches.size(1), patches.size(2), max(img_shape[0], patch_H), max(img_shape[1], patch_W))
    step_int = [0,0]
    step_int[0] = int(patch_H*step[0]) if(isinstance(step[0], float)) else step[0]
    step_int[1] = int(patch_W*step[1]) if(isinstance(step[1], float)) else step[1]
    nrow, ncol = 1 + (img_size[-2] - patch_H)//step_int[0], 1 + (img_size[-1] - patch_W)//step_int[1]
    r_nrow = nrow + 1 if((img_size[2] - patch_H) % step_int[0] != 0) else nrow
    r_ncol = ncol + 1 if((img_size[3] - patch_W) % step_int[1] != 0) else ncol
    patches = patches.reshape(r_nrow,r_ncol,img_size[0],img_size[1],patch_H,patch_W)
    img = torch.zeros(img_size, device = patches.device)
    overlap_counter = torch.zeros(img_size, device = patches.device)
    for i in range(nrow):
        for j in range(ncol):
            img[:,:,i*step_int[0]:i*step_int[0]+patch_H,j*step_int[1]:j*step_int[1]+patch_W] += patches[i,j,]
            overlap_counter[:,:,i*step_int[0]:i*step_int[0]+patch_H,j*step_int[1]:j*step_int[1]+patch_W] += 1
    if((img_size[2] - patch_H) % step_int[0] != 0):
        for j in range(ncol):
            img[:,:,-patch_H:,j*step_int[1]:j*step_int[1]+patch_W] += patches[-1,j,]
            overlap_counter[:,:,-patch_H:,j*step_int[1]:j*step_int[1]+patch_W] += 1
    if((img_size[3] - patch_W) % step_int[1] != 0):
        for i in range(nrow):
            img[:,:,i*step_int[0]:i*step_int[0]+patch_H,-patch_W:] += patches[i,-1,]
            overlap_counter[:,:,i*step_int[0]:i*step_int[0]+patch_H,-patch_W:] += 1
    if((img_size[2] - patch_H) % step_int[0] != 0 and (img_size[3] - patch_W) % step_int[1] != 0):
        img[:,:,-patch_H:,-patch_W:] += patches[-1,-1,]
        overlap_counter[:,:,-patch_H:,-patch_W:] += 1
    img /= overlap_counter
    if(img_shape[0]<patch_H):
        if(strict):
            raise ValueError('img height smaller than patch height')
        else:
            num_padded_H_Top = (patch_H - img_shape[0])//2
            num_padded_H_Bottom = patch_H - img_shape[0] - num_padded_H_Top
            img = img[:,:,num_padded_H_Top:-num_padded_H_Bottom,]
    if(img_shape[1]<patch_W):
        if(strict):
            raise ValueError('img width smaller than patch width')
        else:
            num_padded_W_Left = (patch_W - img_shape[1])//2
            num_padded_W_Right = patch_W - img_shape[1] - num_padded_W_Left
            img = img[:,:,:,num_padded_W_Left:-num_padded_W_Right]
    return img
