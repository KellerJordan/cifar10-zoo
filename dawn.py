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

####################
## CORE
#####################

import inspect
from collections import namedtuple, defaultdict
from functools import partial
import functools
from itertools import chain, count, islice as take

#####################
## dict utils
#####################

make_tuple = lambda path: (path,) if isinstance(path, str) else path

def path_iter(nested_dict, pfx=()):
    for name, val in nested_dict.items():
        if isinstance(val, dict): yield from path_iter(val, pfx+make_tuple(name))
        else: yield (pfx+make_tuple(name), val)  
            
map_values = lambda func, dct: {k: func(v) for k,v in dct.items()}

def map_nested(func, nested_dict):
    return {k: map_nested(func, v) if isinstance(v, dict) else func(v) for k,v in nested_dict.items()}

def group_by_key(seq):
    res = defaultdict(list)
    for k, v in seq: 
        res[k].append(v) 
    return res

reorder = lambda dct, keys: {k: dct[k] for k in keys}

#####################
## graph building
#####################

def identity(value): return value

def build_graph(net, path_map='_'.join):
    net = {path: node if len(node) == 3 else (*node, None) for path, node in path_iter(net)}
    default_inputs = chain([('input',)], net.keys())
    resolve_path = lambda path, pfx: pfx+path if (pfx+path in net or not pfx) else resolve_path(net, path, pfx[:-1])
    return {path_map(path): (typ, value, ([path_map(default)] if inputs is None else [path_map(resolve_path(make_tuple(k), path[:-1])) for k in inputs])) 
            for (path, (typ, value, inputs)), default in zip(net.items(), default_inputs)}

#####################
## Layers
##################### 

import pandas as pd
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from collections import namedtuple
import copy

torch.backends.cudnn.benchmark = True

class Network(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.graph = {path: (typ, typ(**params), inputs) for path, (typ, params, inputs) in build_graph(net).items()}
        for path, (_,node,_) in self.graph.items(): 
            setattr(self, path, node)
    
    def nodes(self):
        return (node for _,node,_ in self.graph.values())
    
    def forward(self, inputs):
        outputs = dict(inputs)
        for k, (_, node, ins) in self.graph.items():
            outputs[k] = node(*[outputs[x] for x in ins])
        return outputs
    
    def half(self):
        for node in self.nodes():
            if isinstance(node, nn.Module) and not isinstance(node, nn.BatchNorm2d):
                node.half()
        return self
    
class Add(namedtuple('Add', [])):
    def __call__(self, x, y): return x + y 
    
class AddWeighted(namedtuple('AddWeighted', ['wx', 'wy'])):
    def __call__(self, x, y): return self.wx*x + self.wy*y 
    
class Identity(namedtuple('Identity', [])):
    def __call__(self, x): return x

class BatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, weight=True, bias=True):
        super().__init__(num_features, eps=eps, momentum=momentum)
        self.weight.data.fill_(1.0)
        self.bias.data.fill_(0.0)
        self.weight.requires_grad = weight
        self.bias.requires_grad = bias

class GhostBatchNorm(BatchNorm):
    def __init__(self, num_features, num_splits, **kw):
        super().__init__(num_features, **kw)
        self.num_splits = num_splits
        self.register_buffer('running_mean', torch.zeros(num_features*self.num_splits))
        self.register_buffer('running_var', torch.ones(num_features*self.num_splits))

    def train(self, mode=True):
        if (self.training is True) and (mode is False): #lazily collate stats when we are going to use them
            self.running_mean = torch.mean(self.running_mean.view(self.num_splits, self.num_features), dim=0).repeat(self.num_splits)
            self.running_var = torch.mean(self.running_var.view(self.num_splits, self.num_features), dim=0).repeat(self.num_splits)
        return super().train(mode)
        
    def forward(self, input):
        N, C, H, W = input.shape
        if self.training or not self.track_running_stats:
            return F.batch_norm(
                input.view(-1, C*self.num_splits, H, W), self.running_mean, self.running_var, 
                self.weight.repeat(self.num_splits), self.bias.repeat(self.num_splits),
                True, self.momentum, self.eps).view(N, C, H, W) 
        else:
            return F.batch_norm(
                input, self.running_mean[:self.num_features], self.running_var[:self.num_features], 
                self.weight, self.bias, False, self.momentum, self.eps)
        
class Mul(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = weight
    def __call__(self, x): 
        return x*self.weight
    
class Flatten(nn.Module):
    def forward(self, x): 
        return x.view(x.size(0), x.size(1))
    
# node definitions   
from inspect import signature    

class node_def(namedtuple('node_def', ['type'])):
    def __call__(self, *args, **kwargs):
        return (self.type, dict(signature(self.type).bind(*args, **kwargs).arguments))

conv = node_def(nn.Conv2d)
linear = node_def(nn.Linear)
batch_norm = node_def(BatchNorm)
pool = node_def(nn.MaxPool2d)
relu = node_def(nn.ReLU)
    
def map_types(mapping, net):
    def f(node):
        typ, *rest = node
        return (mapping.get(typ, typ), *rest)
    return map_nested(f, net) 

#####################
## TRAINING
#####################

import time

class Timer():
    def __init__(self, synch=None):
        self.synch = synch or (lambda: None)
        self.synch()
        self.times = [time.perf_counter()]
        self.total_time = 0.0

    def __call__(self, update_total=True):
        self.synch()
        self.times.append(time.perf_counter())
        delta_t = self.times[-1] - self.times[-2]
        if update_total:
            self.total_time += delta_t
        return delta_t

default_table_formats = {float: '{:{w}.4f}', str: '{:>{w}s}', 'default': '{:{w}}', 'title': '{:>{w}s}'}

def table_formatter(val, is_title=False, col_width=12, formats=None):
    formats = formats or default_table_formats
    type_ = lambda val: float if isinstance(val, (float, np.float)) else type(val)
    return (formats['title'] if is_title else formats.get(type_(val), formats['default'])).format(val, w=col_width)

class Table():
    def __init__(self, keys=None, report=(lambda data: True), formatter=table_formatter):
        self.keys, self.report, self.formatter = keys, report, formatter
        self.log = []
        
    def append(self, data):
        self.log.append(data)
        data = {' '.join(p): v for p,v in path_iter(data)}
        self.keys = self.keys or data.keys()
        if len(self.log) == 1:
            print(*(self.formatter(k, True) for k in self.keys))
        if self.report(data):
            print(*(self.formatter(data[k]) for k in self.keys))
            
    def df(self):
        return pd.DataFrame([{'_'.join(p): v for p,v in path_iter(row)} for row in self.log])     

#####################
## Network
#####################

conv_block = lambda c_in, c_out: {
    'conv': conv(in_channels=c_in, out_channels=c_out, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False), 
    'norm': batch_norm(c_out), 
    'act':  relu(),
}

conv_pool_block = lambda c_in, c_out: dict(conv_block(c_in, c_out), pool=pool(2))
conv_pool_block_pre = lambda c_in, c_out: reorder(conv_pool_block(c_in, c_out), ('conv', 'pool', 'norm', 'act'))

residual = lambda c, conv_block: {
    'in': (Identity, {}),
    'res1': conv_block(c, c),
    'res2': conv_block(c, c),
    'out': (Identity, {}),
    'add': (Add, {}, ['in', 'out']),
}

channels={'prep': 64, 'layer1': 128, 'layer2': 256, 'layer3': 512}
def build_network(channels=channels, res_layers=('layer1', 'layer3'), scale=1/8, conv_block=conv_block, 
                  prep_block=conv_block, conv_pool_block=conv_pool_block, types=None): 
    net = {
        'prep': prep_block(3, channels['prep']),
        'layer1': conv_pool_block(channels['prep'], channels['layer1']),
        'layer2': conv_pool_block(channels['layer1'], channels['layer2']),
        'layer3': conv_pool_block(channels['layer2'], channels['layer3']),
        'pool': pool(4),
        'classifier': {
            'flatten': (Flatten, {}),
            'conv': linear(channels['layer3'], 10, bias=False),
            'scale': (Mul, {'weight': scale}),
        },
        'logits': (Identity, {}),
    }
    for layer in res_layers:
        net[layer]['residual'] = residual(channels[layer], conv_block)    
    if types: net = map_types(types, net)
    return net

#####################
## Misc
#####################

def patches(data, patch_size=(3, 3), dtype=torch.float32):
    h, w = patch_size
    c = data.size(1)
    return data.unfold(2,h,1).unfold(3,w,1).transpose(1,3).reshape(-1, c, h, w).to(dtype)

def eigens(patches):
    n,c,h,w = patches.shape
    
    X = patches.reshape(n, c*h*w)
    X = X/np.sqrt(len(X) - 1)
    Σ = X.T @ X
    
#     Λ, V = torch.symeig(Σ, eigenvectors=True)
    Λ, V = torch.linalg.eigh(Σ, UPLO='U')
    
    return Λ.flip(0), V.t().reshape(c*h*w, c, h, w).flip(0)

train_aug = dict(flip=True, translate=4)
train_loader = PrepadCifarLoader('/tmp/cifar10', train=True, batch_size=512, aug=train_aug)
test_loader = PrepadCifarLoader('/tmp/cifar10', train=False, batch_size=1000)

train_images = train_loader.normalize(train_loader.images)[:10000]

Λ, V = eigens(patches(train_images)) #center crop to remove padding

def whitening_block(c_in, c_out, Λ=None, V=None, eps=1e-2):
    filt = nn.Conv2d(3, 27, kernel_size=(3,3), padding=(1,1), bias=False)
    filt.weight.data = (V/torch.sqrt(Λ+eps)[:,None,None,None])
    filt.weight.requires_grad = False
    
    return {
        'whiten': (identity, {'value': filt}),
        'conv': conv(27, c_out, kernel_size=(1, 1), bias=False),
        'norm': batch_norm(c_out), 
        'act':  relu(),
    }

input_whitening_net = build_network(conv_pool_block=conv_pool_block_pre, prep_block=partial(whitening_block, Λ=Λ, V=V), scale=1/16, types={
    nn.ReLU: partial(nn.CELU, 0.3),
    BatchNorm: partial(GhostBatchNorm, num_splits=16, weight=False)
})

from tqdm import tqdm

epochs = 10
batch_size = 512
ema_epochs = 2

loss_fn = nn.CrossEntropyLoss(label_smoothing=0.2, reduction='none')

lr = 1.0 / 512
momentum = 0.9
wd = 5e-4 * 512

total_train_steps = len(train_loader) * epochs
sched2 = np.interp(np.arange(1+total_train_steps),
                    [0, int(0.2 * total_train_steps), int(0.8 * total_train_steps), total_train_steps],
                    [0, 1, 0.1, 0.1])

epoch_logs = Table(report=lambda data: data['epoch'] % epochs == 0)

for run in range(1):
    
    model = Network(input_whitening_net).half().cuda()
    
    ## optimizer setup
    nonbias_params = [p for k, p in model.named_parameters() if p.requires_grad and 'bias' not in k]
    bias_params = [p for k, p in model.named_parameters() if p.requires_grad and 'bias' in k]
    hyp_nonbias = dict(params=nonbias_params, lr=lr, weight_decay=wd)
    hyp_bias = dict(params=bias_params, lr=lr*64, weight_decay=wd/64)
    
    opt = torch.optim.SGD([hyp_nonbias, hyp_bias], momentum=momentum, nesterov=True)
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, sched2.__getitem__)
    
    ## ema setup
    ema_model = copy.deepcopy(model)
    ema_momentum = 0.99
    update_freq = 5
    rho = ema_momentum**update_freq
    iter_counter = count()
    
    timer = Timer(torch.cuda.synchronize)
    
    for epoch in tqdm(range(epochs)):
        
        logs = []
        model.train()
        for inputs, labels in train_loader:
            
            ## forward and logging
            out = model({'input': inputs, 'target': labels})
            losses = loss_fn(out['logits'], labels)
            correct = (out['logits'].detach().argmax(1) == labels)
            logs.extend([('losses', loss), ('acc', correct)])
            
            ## backward
            model.zero_grad()
            losses.sum().backward()
            
            ## optimizer step
            scheduler.step()
            opt.step()
            
            ## ema update
            if next(iter_counter) % update_freq == 0:
                for (k, v), ema_v in zip(model.state_dict().items(), ema_model.state_dict().values()):
                    if 'num_batches_tracked' in k:
                        continue
                    ema_v.lerp_(v, 1-rho)
        
        train_summary = {k: torch.cat(xs).float().mean().item() for k, xs in group_by_key(logs).items()}
        train_time = timer()

        logs = []
        ema_model.eval()
        for inputs, labels in test_loader:
            with torch.no_grad():
                logits = ema_model({'input': inputs})['logits']
                logits_flip = ema_model({'input': inputs.flip(-1)})['logits']
                logits_tta = (logits + logits_flip) / 2
            losses = loss_fn(logits_tta, labels)
            correct = (logits_tta.argmax(1) == labels)
            logs.extend([('loss', losses), ('acc', correct)])

        valid_summary = {k: torch.cat(xs).float().mean().item() for k, xs in group_by_key(logs).items()}
        valid_time = timer(update_total=False)
        
        log = {
            'train': {'time': train_time, **train_summary}, 
            'valid': {'time': valid_time, **valid_summary}, 
            'total time': timer.total_time
        }
        epoch_logs.append({'run': run+1, 'epoch': epoch+1, **log})
        
cols = ['valid_acc']
summary = epoch_logs.df().query('epoch==epoch.max()')[cols].describe().transpose().astype({'count': int})[
    ['count', 'mean', 'min', 'max', 'std']]
print(summary)

