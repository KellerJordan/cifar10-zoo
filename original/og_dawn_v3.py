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
        self.images = (self.images.half() / 255).permute(0, 3, 1, 2)

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

train_loader = PrepadCifarLoader('/tmp/cifar10', train=True, aug=dict(flip=True, translate=4), batch_size=512)
test_loader = PrepadCifarLoader('/tmp/cifar10', train=False, batch_size=1000)

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

union = lambda *dicts: {k: v for d in dicts for (k, v) in d.items()}

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
    net = {path: node if len(node) is 3 else (*node, None) for path, node in path_iter(net)}
    default_inputs = chain([('input',)], net.keys())
    resolve_path = lambda path, pfx: pfx+path if (pfx+path in net or not pfx) else resolve_path(net, path, pfx[:-1])
    return {path_map(path): (typ, value, ([path_map(default)] if inputs is None else [path_map(resolve_path(make_tuple(k), path[:-1])) for k in inputs])) 
            for (path, (typ, value, inputs)), default in zip(net.items(), default_inputs)}

#####################
## Layers
##################### 

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from collections import namedtuple
import copy

torch.backends.cudnn.benchmark = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cpu = torch.device('cpu')
    
class Network(nn.Module):
    def __init__(self, net, loss=None):
        super().__init__()
        self.graph = {path: (typ, typ(**params), inputs) for path, (typ, params, inputs) in build_graph(net).items()}
        self.loss = loss or identity
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

build_model = lambda network, loss: Network(network, loss).half().to(device)
    
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

# Losses
class CrossEntropyLoss(namedtuple('CrossEntropyLoss', [])):
    def __call__(self, log_probs, target):
        return torch.nn.functional.nll_loss(log_probs, target, reduction='none')
    
class KLLoss(namedtuple('KLLoss', [])):        
    def __call__(self, log_probs):
        return -log_probs.mean(dim=1)

class Correct(namedtuple('Correct', [])):
    def __call__(self, classifier, target):
        return classifier.max(dim = 1)[1] == target

class LogSoftmax(namedtuple('LogSoftmax', ['dim'])):
    def __call__(self, x):
        return torch.nn.functional.log_softmax(x, self.dim, _stacklevel=5)

    
# node definitions   
from inspect import signature    
empty_signature = inspect.Signature()

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
## Compat
##################### 

def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()  
    return x
  
def flip_lr(x):
    if isinstance(x, torch.Tensor):
        return torch.flip(x, [-1]) 
    return x[..., ::-1].copy()
  
trainable_params = lambda model: {k:p for k,p in model.named_parameters() if p.requires_grad}

#####################
## Optimisers
##################### 

from functools import partial

def nesterov_update(w, dw, v, lr, weight_decay, momentum):
    dw.add_(weight_decay, w).mul_(-lr)
    v.mul_(momentum).add_(dw)
    w.add_(dw.add_(momentum, v))

def zeros_like(weights):
    return [torch.zeros_like(w) for w in weights]

def optimiser(weights, param_schedule, update, state_init):
    weights = list(weights)
    return {'update': update, 'param_schedule': param_schedule, 'step_number': 0, 'weights': weights,  'opt_state': state_init(weights)}

def opt_step(update, param_schedule, step_number, weights, opt_state):
    step_number += 1
    param_values = {k: f(step_number) for k, f in param_schedule.items()}
    for w, v in zip(weights, opt_state):
        if w.requires_grad:
            update(w.data, w.grad.data, v, **param_values)
    return {'update': update, 'param_schedule': param_schedule, 'step_number': step_number, 'weights': weights,  'opt_state': opt_state}

SGD = partial(optimiser, update=nesterov_update, state_init=zeros_like)
  
class PiecewiseLinear(namedtuple('PiecewiseLinear', ('knots', 'vals'))):
    def __call__(self, t):
        return np.interp([t], self.knots, self.vals)[0]
     
class Const(namedtuple('Const', ['val'])):
    def __call__(self, x):
        return self.val

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

every = lambda n, col: (lambda data: data[col] % n == 0)

class Table():
    def __init__(self, keys=None, report=(lambda data: True), formatter=table_formatter):
        self.keys, self.report, self.formatter = keys, report, formatter
        self.log = []
        
    def append(self, data):
        self.log.append(data)
        data = {' '.join(p): v for p,v in path_iter(data)}
        self.keys = self.keys or data.keys()
        if len(self.log) is 1:
            print(*(self.formatter(k, True) for k in self.keys))
        if self.report(data):
            print(*(self.formatter(data[k]) for k in self.keys))
            
    def df(self):
        return pd.DataFrame([{'_'.join(p): v for p,v in path_iter(row)} for row in self.log])     
            
def reduce(batches, state, steps):
    #state: is a dictionary
    #steps: are functions that take (batch, state)
    #and return a dictionary of updates to the state (or None)
    
    for batch in chain(batches, [None]): 
    #we send an extra batch=None at the end for steps that 
    #need to do some tidying-up (e.g. log_activations)
        for step in steps:
            updates = step(batch, state)
            if updates:
                for k,v in updates.items():
                    state[k] = v                  
    return state
  
#define keys in the state dict as constants
MODEL = 'model'
VALID_MODEL = 'valid_model'
OUTPUT = 'output'
OPTS = 'optimisers'
ACT_LOG = 'activation_log'
WEIGHT_LOG = 'weight_log'

#step definitions
def forward(training_mode):
    def step(batch, state):
        if not batch: return
        model = state[MODEL] if training_mode or (VALID_MODEL not in state) else state[VALID_MODEL]
        if model.training != training_mode: #without the guard it's slow!
            model.train(training_mode)
        return {OUTPUT: model.loss(model(batch))}
    return step

def forward_tta(tta_transforms):
    def step(batch, state):
        if not batch: return
        model = state[MODEL] if (VALID_MODEL not in state) else state[VALID_MODEL]
        if model.training:
            model.train(False)
        logits = torch.mean(torch.stack([model({'input': transform(batch['input'].clone())})['logits'].detach() for transform in tta_transforms], dim=0), dim=0)
        return {OUTPUT: model.loss(dict(batch, logits=logits))}
    return step

def backward(dtype=torch.float16):
    def step(batch, state):
        state[MODEL].zero_grad()
        if not batch: return
        state[OUTPUT]['loss'].to(dtype).sum().backward()
    return step

def opt_steps(batch, state):
    if not batch: return
    return {OPTS: [opt_step(**opt) for opt in state[OPTS]]}

def log_activations(node_names=('loss', 'acc')):
    logs = []
    def step(batch, state):
        if batch:
            logs.extend((k, state[OUTPUT][k].detach()) for k in node_names)
        else:
            res = map_values((lambda xs: to_numpy(torch.cat(xs)).astype(np.float)), group_by_key(logs))
            logs.clear()
            return {ACT_LOG: res}
    return step

def update_ema(momentum, update_freq=1):
    n = iter(count())
    rho = momentum**update_freq
    def step(batch, state):
        if not batch: return
        if (next(n) % update_freq) != 0: return
        for v, ema_v in zip(state[MODEL].state_dict().values(), state[VALID_MODEL].state_dict().values()):
            if v.dtype in (torch.half, torch.float):
                ema_v *= rho
                ema_v += (1-rho)*v
    return step

train_steps = (forward(training_mode=True), log_activations(('loss', 'acc')), backward(), opt_steps)
valid_steps = (forward(training_mode=False), log_activations(('loss', 'acc')))

epoch_stats = lambda state: {k: np.mean(v) for k, v in state[ACT_LOG].items()}

def train_epoch(state, timer, train_batches, valid_batches, train_steps=train_steps, valid_steps=valid_steps, on_epoch_end=identity):
    train_summary, train_time = epoch_stats(on_epoch_end(reduce(train_batches, state, train_steps))), timer()
    valid_summary, valid_time = epoch_stats(reduce(valid_batches, state, valid_steps)), timer(update_total=False) #DAWNBench rules
    return {
        'train': union({'time': train_time}, train_summary), 
        'valid': union({'time': valid_time}, valid_summary), 
        'total time': timer.total_time
    }

summary = lambda logs, cols=['valid_acc']: logs.df().query('epoch==epoch.max()')[cols].describe().transpose().astype({'count': int})[
    ['count', 'mean', 'min', 'max', 'std']]

#on_epoch_end
def log_weights(state, weights):
    state[WEIGHT_LOG] = state.get(WEIGHT_LOG, [])
    state[WEIGHT_LOG].append({k: to_numpy(v.data) for k,v in weights.items()})
    return state

def fine_tune_bn_stats(state, batches, model_key=VALID_MODEL):
    reduce(batches, {MODEL: state[model_key]}, [forward(True)])
    return state

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

def build_network(channels, extra_layers, res_layers, scale, conv_block=conv_block, 
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
    for layer in extra_layers:
        net[layer]['extra'] = conv_block(channels[layer], channels[layer])     
    if types: net = map_types(types, net)
    return net

channels={'prep': 64, 'layer1': 128, 'layer2': 256, 'layer3': 512}
network = partial(build_network, channels=channels, extra_layers=(), res_layers=('layer1', 'layer3'), scale=1/8)   

x_ent_loss = Network({
  'loss':  (nn.CrossEntropyLoss, {'reduction': 'none'}, ['logits', 'target']),
  'acc': (Correct, {}, ['logits', 'target'])
})

label_smoothing_loss = lambda alpha: Network({
        'logprobs': (LogSoftmax, {'dim': 1}, ['logits']),
        'KL':  (KLLoss, {}, ['logprobs']),
        'xent':  (CrossEntropyLoss, {}, ['logprobs', 'target']),
        'loss': (AddWeighted, {'wx': 1-alpha, 'wy': alpha}, ['xent', 'KL']),
        'acc': (Correct, {}, ['logits', 'target']),
    })

#####################
## Misc
#####################

lr_schedule = lambda knots, vals, batch_size: PiecewiseLinear(np.array(knots)*len(train_loader), np.array(vals)/batch_size)


def cov(X):
    X = X/np.sqrt(X.size(0) - 1)
    return X.t() @ X

def patches(data, patch_size=(3, 3), dtype=torch.float32):
    h, w = patch_size
    c = data.size(1)
    return data.unfold(2,h,1).unfold(3,w,1).transpose(1,3).reshape(-1, c, h, w).to(dtype)

def eigens(patches):
    n,c,h,w = patches.shape
    Σ = cov(patches.reshape(n, c*h*w))
    Λ, V = torch.linalg.eigh(Σ)
    return Λ.flip(0), V.t().reshape(c*h*w, c, h, w).flip(0)

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

input_whitening_net = network(conv_pool_block=conv_pool_block_pre, prep_block=partial(whitening_block, Λ=Λ, V=V), scale=1/16, types={
    nn.ReLU: partial(nn.CELU, 0.3),
    BatchNorm: partial(GhostBatchNorm, num_splits=16, weight=False)
})

valid_steps_tta = (forward_tta([identity, flip_lr]), log_activations(('loss', 'acc')))

epochs, batch_size, ema_epochs=10, 512, 2
opt_params = {'lr': lr_schedule([0, epochs/5, epochs - ema_epochs], [0.0, 1.0, 0.1], batch_size), 'weight_decay': Const(5e-4*batch_size), 'momentum': Const(0.9)}
opt_params_bias = {'lr': lr_schedule([0, epochs/5, epochs - ema_epochs], [0.0, 1.0*64, 0.1*64], batch_size), 'weight_decay': Const(5e-4*batch_size/64), 'momentum': Const(0.9)}

N_RUNS = 5

logs = Table(report=every(epochs,'epoch'))
for run in range(N_RUNS):
    model = build_model(input_whitening_net, label_smoothing_loss(0.2))
    is_bias = group_by_key(('bias' in k, v) for k, v in trainable_params(model).items())
    state, timer = {MODEL: model, VALID_MODEL: copy.deepcopy(model), OPTS: [SGD(is_bias[False], opt_params), SGD(is_bias[True], opt_params_bias)]}, Timer(torch.cuda.synchronize)
    for epoch in range(epochs):
        train_batches = [{'input': inputs, 'target': labels} for inputs, labels in train_loader]
        valid_batches = [{'input': inputs, 'target': labels} for inputs, labels in test_loader]
        logs.append(union({'run': run+1, 'epoch': epoch+1}, train_epoch(state, timer, train_batches, valid_batches, 
                                                                        train_steps=(*train_steps, update_ema(momentum=0.99, update_freq=5)),
                                                                        valid_steps=valid_steps_tta)))
summary(logs)

torch.save(logs, 'og_dawn_log.pt')

