# dawnbench_dcpage.py
# This script aims for exact equivalence to the final training procedure presented in David C. Page's post
# https://myrtle.ai/learn/how-to-train-your-resnet-8-bag-of-tricks/.
#
# It runs in 15 seconds on an NVIDIA A100, and yields a mean accuracy of 94.07% (across n=100 runs, in one test).
#
# It should be exactly equivalent to the final (10-epoch) training code given in
# https://github.com/davidcpage/cifar10-fast/blob/master/bag_of_tricks.ipynb, with the one exception being that
# we use the default Pytorch nesterov SGD, whereas the notebook uses a custom nesterov SGD which has a small bug.
#
# The print-logging and code layout are inspired by https://github.com/tysam-code/hlb-CIFAR10.

import copy
from itertools import count
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

torch.backends.cudnn.benchmark = True

#############################################
#                DataLoader                 #
#############################################

## https://github.com/KellerJordan/cifar10-loader/blob/master/quick_cifar/loader.py
import os
from math import ceil
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T

CIFAR_MEAN = torch.tensor((0.4914, 0.4822, 0.4465))
CIFAR_STD = torch.tensor((0.2470, 0.2435, 0.2616))

def make_random_square_masks(inputs, size):
    is_even = int(size % 2 == 0)
    n,c,h,w = inputs.shape

    # seed top-left corners of squares to cutout boxes from, in one dimension each
    corner_y = torch.randint(0, h-size+1, size=(n,), device=inputs.device)
    corner_x = torch.randint(0, w-size+1, size=(n,), device=inputs.device)

    # measure distance, using the center as a reference point
    corner_y_dists = torch.arange(h, device=inputs.device).view(1, 1, h, 1) - corner_y.view(-1, 1, 1, 1)
    corner_x_dists = torch.arange(w, device=inputs.device).view(1, 1, 1, w) - corner_x.view(-1, 1, 1, 1)
    
    mask_y = (corner_y_dists >= 0) * (corner_y_dists < size)
    mask_x = (corner_x_dists >= 0) * (corner_x_dists < size)

    final_mask = mask_y * mask_x

    return final_mask

def batch_flip_lr(inputs):
    flip_mask = (torch.rand(len(inputs), device=inputs.device) < 0.5).view(-1, 1, 1, 1)
    return torch.where(flip_mask, inputs.flip(-1), inputs)

def batch_crop(inputs, crop_size):
    crop_mask = make_random_square_masks(inputs, crop_size)
    cropped_batch = torch.masked_select(inputs, crop_mask)
    return cropped_batch.view(inputs.shape[0], inputs.shape[1], crop_size, crop_size)

def batch_cutout(inputs, size):
    cutout_masks = make_random_square_masks(inputs, size)
    return inputs.masked_fill(cutout_masks, 0)

## This is a pre-padded variant of quick_cifar.CifarLoader which moves the padding step of random translate
## from __iter__ to __init__, so that it doesn't need to be repeated each epoch.
class PrepadCifarLoader:

    def __init__(self, path, train=True, batch_size=500, aug=None, drop_last=None, shuffle=None, gpu=0):
        data_path = os.path.join(path, 'train.pt' if train else 'test.pt')
        if not os.path.exists(data_path):
            dset = torchvision.datasets.CIFAR10(path, download=True, train=train)
            images = torch.tensor(dset.data)
            labels = torch.tensor(dset.targets)
            torch.save({'images': images, 'labels': labels, 'classes': dset.classes}, data_path)

        data = torch.load(data_path, map_location=torch.device(gpu))
        self.images, self.labels, self.classes = data['images'], data['labels'], data['classes']
        # It's faster to load+process uint8 data than to load preprocessed fp16 data
        self.images = (self.images.half() / 255).permute(0, 3, 1, 2).to(memory_format=torch.channels_last)

        self.normalize = T.Normalize(CIFAR_MEAN, CIFAR_STD)
        self.denormalize = T.Normalize(-CIFAR_MEAN / CIFAR_STD, 1 / CIFAR_STD)
        
        self.aug = aug or {}
        for k in self.aug.keys():
            assert k in ['flip', 'translate', 'cutout'], 'Unrecognized key: %s' % k

        # Pre-pad images to save time when doing random translation
        pad = self.aug.get('translate', 0)
        self.padded_images = F.pad(self.images, (pad,)*4, 'reflect') if pad > 0 else None

        self.batch_size = batch_size
        self.drop_last = train if drop_last is None else drop_last
        self.shuffle = train if shuffle is None else shuffle

    def augment_prepad(self, images):
        images = self.normalize(images)
        if self.aug.get('translate', 0) > 0:
            images = batch_crop(images, self.images.shape[-2])
        if self.aug.get('flip', False):
            images = batch_flip_lr(images)
        if self.aug.get('cutout', 0) > 0:
            images = batch_cutout(images, self.aug['cutout'])
        return images

    def __len__(self):
        return len(self.images)//self.batch_size if self.drop_last else ceil(len(self.images)/self.batch_size)

    def __iter__(self):
        images = self.augment_prepad(self.padded_images if self.padded_images is not None else self.images)
        indices = (torch.randperm if self.shuffle else torch.arange)(len(images), device=images.device)
        for i in range(len(self)):
            idxs = indices[i*self.batch_size:(i+1)*self.batch_size]
            yield (images[idxs], self.labels[idxs])

#############################################
#            Network Components             #
#############################################

class BatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-05, momentum=0.9, weight=False, bias=True):
        super().__init__(num_features, eps=eps, momentum=1-momentum)
        self.weight.data.fill_(1.0)
        self.bias.data.fill_(0.0)
        self.weight.requires_grad = weight
        self.bias.requires_grad = bias

class Residual(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
    def forward(self, x): 
        return x + self.module(x)
    
class Flatten(nn.Module):
    def forward(self, x): 
        return x.view(len(x), -1)
    
class Mul(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = weight
    def forward(self, x): 
        return x * self.weight

def make_net():

    def conv(in_channels, out_channels, kernel_size=3, padding=1):
        return nn.Conv2d(in_channels, out_channels,
                         kernel_size=kernel_size, stride=1,
                         padding=padding, bias=False)
    act = nn.CELU(0.3)
    bn = lambda channels: BatchNorm(channels)

    net = nn.Sequential(
        conv(3, 27), # <-- this is the fixed whitening layer
        nn.Sequential(conv(27, 64, 1, 0), bn(64), act),
        nn.Sequential(conv(64, 128), nn.MaxPool2d(2), bn(128), act),
        Residual(nn.Sequential(conv(128, 128), bn(128), act,
                               conv(128, 128), bn(128), act)),
        nn.Sequential(conv(128, 256), nn.MaxPool2d(2), bn(256), act),
        nn.Sequential(conv(256, 512), nn.MaxPool2d(2), bn(512), act),
        Residual(nn.Sequential(conv(512, 512), bn(512), act,
                               conv(512, 512), bn(512), act)),
        nn.MaxPool2d(4),
        Flatten(),
        nn.Linear(512, 10, bias=False),
        Mul(1/16),
    ).cuda()
    net = net.to(memory_format=torch.channels_last)
    
    # convert conv/linear layers to fp16, leaving bn layers fp32
    net.half()
    for mod in net.modules():
        if isinstance(mod, BatchNorm):
            mod.float()
    
    return net

def patches(data, patch_size=(3, 3)):
    h, w = patch_size
    c = data.size(1)
    return data.unfold(2, h, 1).unfold(3, w, 1).transpose(1, 3).reshape(-1, c, h, w).float()

def eigens(patches):
    n, c, h, w = patches.shape
    patches_flat = patches.reshape(n, c*h*w)
    cov = (patches_flat.T @ patches_flat) / (n - 1)
    eigenvalues, eigenvectors = torch.linalg.eigh(cov, UPLO='U')
    return eigenvalues.flip(0), eigenvectors.T.reshape(c*h*w, c, h, w).flip(0)

def init_net(net, train_images, eps=1e-2):
    eigenvalues, eigenvectors = eigens(patches(train_images))
    weight = eigenvectors / (eigenvalues+eps).sqrt()[:, None, None, None]
    net[0].weight.data[:] = weight.half()
    net[0].weight.requires_grad = False

########################################
#               Logging                #
########################################

def print_columns(columns_list, is_head=False, is_final_entry=False):
    print_string = ''
    for col in columns_list:
        print_string += '|  %s  ' % col
    print_string += '|'
    if is_head:
        print('-'*len(print_string))
    print(print_string)
    if is_head or is_final_entry:
        print('-'*len(print_string))

logging_columns_list = ['run', 'epoch', 'train_loss', 'val_loss', 'train_acc', 'val_acc', 'tta_val_acc', 'total_time_seconds']
def print_training_details(variables, is_final_entry):
    formatted = []
    for col in logging_columns_list:
        var = variables[col]
        if type(var) in (int, str):
            res = str(var)
        elif type(var) is float:
            res = '{:0.4f}'.format(var)
        else:
            assert var is None
            res = ''
        formatted.append(res.rjust(len(col)))
    print_columns(formatted, is_final_entry=is_final_entry)

########################################
#           Train and Eval             #
########################################

def main(run):

    epochs = 80
    batch_size = 512

    lr = 0.4
    momentum = 0.9
    wd = 5e-4
    bias_scaler = 64
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.2, reduction='none')

    train_aug = dict(flip=True, translate=4, cutout=12)

    train_loader = PrepadCifarLoader('/tmp/cifar10', train=True, batch_size=batch_size, aug=train_aug)
    test_loader = PrepadCifarLoader('/tmp/cifar10', train=False, batch_size=1000)

    total_train_steps = epochs * len(train_loader)
    sched = np.interp(np.arange(1+total_train_steps),
                      [0, int((epochs/5)*len(train_loader)), total_train_steps],
                      [0, 1, 0])

    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    total_time_seconds = 0

    model = make_net()
    train_images = train_loader.normalize(train_loader.images)[:10000]
    init_net(model, train_images)
    
    ## optimizer setup
    nonbias_params = [p for k, p in model.named_parameters() if p.requires_grad and 'bias' not in k]
    bias_params = [p for k, p in model.named_parameters() if p.requires_grad and 'bias' in k]
    hyp_nonbias = dict(params=nonbias_params, lr=(lr / batch_size), weight_decay=(wd * batch_size))
    hyp_bias = dict(params=bias_params, lr=(lr * bias_scaler/batch_size), weight_decay=(wd * batch_size/bias_scaler))
    
    optimizer = torch.optim.SGD([hyp_nonbias, hyp_bias], momentum=momentum, nesterov=True)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, sched.__getitem__)
    
    for epoch in range(epochs):
        
        ####################
        #     Training     #
        ####################

        starter.record()

        model.train()
        for inputs, labels in train_loader:
            
            outputs = model(inputs)
            loss = loss_fn(outputs, labels).sum()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
        ender.record()
        torch.cuda.synchronize()
        total_time_seconds += 1e-3 * starter.elapsed_time(ender)

        ####################
        #    Evaluation    #
        ####################

        ## save the accuracy and loss from the last training batch of the epoch
        train_acc = (outputs.detach().argmax(-1) == labels).float().mean().item()
        train_loss = loss.item() / batch_size

        model.eval()
        with torch.no_grad():

            loss_list, acc_list, acc_list_tta = [], [], []

            for inputs, labels in test_loader:

                outputs = model(inputs)
                loss_list.append(loss_fn(outputs, labels).float().mean())
                acc_list.append((outputs.argmax(-1) == labels).float().mean())

                outputs_tta = 0.5 * outputs + 0.5 * model(inputs.flip(-1))
                acc_list_tta.append((outputs_tta.argmax(-1) == labels).float().mean())

            val_acc = torch.stack(acc_list).mean().item()
            val_loss = torch.stack(loss_list).mean().item()
            tta_val_acc = torch.stack(acc_list_tta).mean().item()

        print_training_details(locals(), is_final_entry=(epoch == epochs-1))
        run = None # don't print run after first iteration
        
    return tta_val_acc


if __name__ == '__main__':
    print_columns(logging_columns_list, is_head=True)
    accs = torch.tensor([main(run) for run in range(10)])
    print('Mean: %.4f    Std: %.4f' % (accs.mean(), accs.std()))

