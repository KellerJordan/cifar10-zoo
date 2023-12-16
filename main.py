#############################################
#            Setup/Hyperparameters          #
#############################################

import os
import sys
import uuid
import math
import copy
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

torch.backends.cudnn.benchmark = True

default_conv_kwargs = {'kernel_size': 3, 'padding': 'same', 'bias': False}

hyp = {
    'opt': {
        'batch_size': 1024,
        'train_epochs': 10.0,
        'lr': 1.525 / 1024, # per example
        'momentum': 0.85,
        'weight_decay': 2 * 6.687e-4 * 1024, # per batch
        'bias_scaler': 64.0,
        'scaling_factor': 1/9,
        'loss_scale': 32,
    },
    'aug': {
        'translate': 2,
    },
    'net': {
        'whitening': {
            'kernel_size': 2,
        },
        'batch_norm_momentum': 0.6,
        'base_depth': 64
    },
    'ema': {
        'start_epochs': 2,
        'decay_base': 0.95,
        'decay_pow': 3.,
        'every_n_steps': 5,
    },
}

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
            assert k in ['flip', 'translate'], 'Unrecognized key: %s' % k

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

# We might be able to fuse this weight and save some memory/runtime/etc, since the fast version of the network might be able to do without somehow....
class BatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-12, momentum=(1 - hyp['net']['batch_norm_momentum']),
                 weight=False, bias=True):
        super().__init__(num_features, eps=eps, momentum=momentum)
        self.weight.data.fill_(1.0)
        self.bias.data.fill_(0.0)
        self.weight.requires_grad = weight
        self.bias.requires_grad = bias

# Allows us to set default arguments for the whole convolution itself.
# Having an outer class like this does add space and complexity but offers us
# a ton of freedom when it comes to hacking in unique functionality for each layer type
class Conv(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        kwargs = {**default_conv_kwargs, **kwargs}
        super().__init__(*args, **kwargs)
        self.kwargs = kwargs

class Linear(nn.Linear):
    def __init__(self, *args, temperature=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.kwargs = kwargs
        self.temperature = temperature

    def forward(self, x):
        if self.temperature is not None:
            weight = self.weight * self.temperature
        else:
            weight = self.weight
        return x @ weight.T

# can hack any changes to each convolution group that you want directly in here
class ConvGroup(nn.Module):
    def __init__(self, channels_in, channels_out):
        super().__init__()
        self.channels_in  = channels_in
        self.channels_out = channels_out

        self.pool = nn.MaxPool2d(2)
        self.conv1 = Conv(channels_in,  channels_out)
        self.conv2 = Conv(channels_out, channels_out)

        self.norm1 = BatchNorm(channels_out)
        self.norm2 = BatchNorm(channels_out)

        self.activ = nn.GELU()


    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.norm1(x)
        x = self.activ(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activ(x)

        return x

class FastGlobalMaxPooling(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        # Previously was chained torch.max calls.
        # requires less time than AdaptiveMax2dPooling -- about ~.3s for the entire run, in fact (which is pretty significant! :O :D :O :O <3 <3 <3 <3)
        return torch.amax(x, dim=(2,3)) # Global maximum pooling

#############################################
#          Init Helper Functions            #
#############################################

def get_patches(x, patch_shape):
    # This uses the unfold operation (https://pytorch.org/docs/stable/generated/torch.nn.functional.unfold.html?highlight=unfold#torch.nn.functional.unfold)
    # to extract a _view_ (i.e., there's no data copied here) of blocks in the input tensor. We have to do it twice -- once horizontally, once vertically. Then
    # from that, we get our kernel_size*kernel_size patches to later calculate the statistics for the whitening tensor on :D
    c, (h, w) = x.shape[1], patch_shape
    return x.unfold(2,h,1).unfold(3,w,1).transpose(1,3).reshape(-1,c,h,w).float()

def get_whitening_parameters(patches):
    # As a high-level summary, we're basically finding the high-dimensional oval that best fits the data here.
    # We can then later use this information to map the input information to a nicely distributed sphere, where also
    # the most significant features of the inputs each have their own axis. This significantly cleans things up for the
    # rest of the neural network and speeds up training.
    n,c,h,w = patches.shape
    patches_flat = patches.view(n, -1)
    est_patch_covariance = (patches_flat.T @ patches_flat) / n
    eigenvalues, eigenvectors = torch.linalg.eigh(est_patch_covariance, UPLO='U')
    return eigenvalues.flip(0).view(-1, 1, 1, 1), eigenvectors.T.reshape(c*h*w,c,h,w).flip(0)

# Run this over the training set to calculate the patch statistics, then set the initial convolution as a non-learnable 'whitening' layer
# Note that this is a large epsilon, so the bottom half of principal directions won't fully whiten
def init_whitening_conv(layer, train_set, eps=5e-4):
    patches = get_patches(train_set, patch_shape=layer.weight.data.shape[2:])
    eigenvalues, eigenvectors = get_whitening_parameters(patches)
    eigenvectors_scaled = eigenvectors/torch.sqrt(eigenvalues+eps) # set the filters as the eigenvectors in order to whiten inputs
    eigenvectors_scaled_truncated = eigenvectors_scaled[:len(layer.weight)//2]
    layer.weight.data[:] = torch.cat((eigenvectors_scaled_truncated, -eigenvectors_scaled_truncated))
    ## We don't want to train this, since this is implicitly whitening over the whole dataset
    ## For more info, see David Page's original blogposts (link in the README.md as of this commit.)
    layer.weight.requires_grad = False

#############################################
#            Network Definition             #
#############################################

scaler = 2. ## You can play with this on your own if you want, for the first beta I wanted to keep things simple (for now) and leave it out of the hyperparams dict
depths = {
    'block1': round(scaler**0 * hyp['net']['base_depth']), # 64  w/ scaler at base value
    'block2': round(scaler**2 * hyp['net']['base_depth']), # 256 w/ scaler at base value
    'block3': round(scaler**3 * hyp['net']['base_depth']), # 512 w/ scaler at base value
    'num_classes': 10
}
depths = {'block1': 64, 'block2': 256, 'block3': 448, 'num_classes': 10}

class SpeedyConvNet(nn.Module):
    def __init__(self, network_dict):
        super().__init__()
        self.net_dict = network_dict # flexible, defined in the make_net function

    # This allows you to customize/change the execution order of the network as needed.
    def forward(self, x):
        x = self.net_dict['initial_block']['whiten'](x)
        x = self.net_dict['initial_block']['activation'](x)
        x = self.net_dict['conv_group_1'](x)
        x = self.net_dict['conv_group_2'](x)
        x = self.net_dict['conv_group_3'](x)
        x = self.net_dict['pooling'](x)
        x = self.net_dict['linear'](x)
        return x

def make_net():
    whiten_conv_depth = 2 * 3 * hyp['net']['whitening']['kernel_size']**2
    network_dict = nn.ModuleDict({
        'initial_block': nn.ModuleDict({
            'whiten': Conv(3, whiten_conv_depth, kernel_size=hyp['net']['whitening']['kernel_size'], padding=0),
            'activation': nn.GELU(),
        }),
        'conv_group_1': ConvGroup(whiten_conv_depth, depths['block1']),
        'conv_group_2': ConvGroup(depths['block1'],  depths['block2']),
        'conv_group_3': ConvGroup(depths['block2'],  depths['block3']),
        'pooling': FastGlobalMaxPooling(),
        'linear': Linear(depths['block3'], depths['num_classes'], bias=False, temperature=hyp['opt']['scaling_factor']),
    })

    net = SpeedyConvNet(network_dict)
    net = net.cuda()
    net = net.to(memory_format=torch.channels_last)
    net.half()
    for mod in net.modules():
        if isinstance(mod, BatchNorm):
            mod.float()
    return net

def init_net(net, train_images):

    with torch.no_grad():
        init_whitening_conv(net.net_dict['initial_block']['whiten'], train_images)

        for name, block in net.net_dict.items():
            if 'conv_group' in name:
                # Create an implicit residual via a dirac-initialized tensor
                dirac_weights_in = torch.nn.init.dirac_(torch.empty_like(block.conv1.weight))

                # Add the implicit residual to the already-initialized convolutional transition layer.
                # One can use more sophisticated initializations, but this one appeared worked best in testing.
                # What this does is brings up the features from the previous residual block virtually, so not only 
                # do we have residual information flow within each block, we have a nearly direct connection from
                # the early layers of the network to the loss function.
                std_pre, mean_pre = torch.std_mean(block.conv1.weight.data)
                block.conv1.weight.data = block.conv1.weight.data + dirac_weights_in 
                std_post, mean_post = torch.std_mean(block.conv1.weight.data)

                # Renormalize the weights to match the original initialization statistics
                block.conv1.weight.data.sub_(mean_post).div_(std_post).mul_(std_pre).add_(mean_pre)

                ## We do the same for the second layer in each convolution group block, since this only
                ## adds a simple multiplier to the inputs instead of the noise of a randomly-initialized
                ## convolution. This can be easily scaled down by the network, and the weights can more easily
                ## pivot in whichever direction they need to go now.
                ## The reason that I believe that this works so well is because a combination of MaxPool2d
                ## and the nn.GeLU function's positive bias encouraging values towards the nearly-linear
                ## region of the GeLU activation function at network initialization. I am not currently
                ## sure about this, however, it will require some more investigation. For now -- it works! D:
                torch.nn.init.dirac_(block.conv2.weight)

########################################
#          Training Helpers            #
########################################

class NetworkEMA(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net_ema = copy.deepcopy(net).eval()

    def update(self, net, decay):
        with torch.no_grad():
            for net_ema_param, (param_name, net_param) in zip(self.net_ema.state_dict().values(), net.state_dict().items()):
                if net_param.dtype in (torch.half, torch.float):
                    net_ema_param.lerp_(net_param.detach(), 1-decay)
                    # And then we also copy the parameters back to the network, similarly to the Lookahead optimizer (but with a much more aggressive-at-the-end schedule)
                    if not 'whiten' in param_name:
                        net_param.copy_(net_ema_param.detach())

    def forward(self, inputs):
        return self.net_ema(inputs)

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

logging_columns_list = ['run', 'epoch', 'train_loss', 'val_loss', 'train_acc', 'val_acc', 'ema_val_acc', 'tta_ema_val_acc', 'total_time_seconds']
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

    total_time_seconds = 0.
    current_steps = 0.

    train_augs = dict(flip=True, translate=hyp['aug']['translate'])
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.2, reduction='none')

    batch_size = hyp['opt']['batch_size']
    epochs = hyp['opt']['train_epochs']
    lr = hyp['opt']['lr']
    momentum = hyp['opt']['momentum']
    wd = hyp['opt']['weight_decay']
    bias_scaler = hyp['opt']['bias_scaler']
    loss_scale = hyp['opt']['loss_scale']

    train_loader = PrepadCifarLoader('/tmp/cifar10', train=True, batch_size=batch_size, aug=train_augs)
    test_loader = PrepadCifarLoader('/tmp/cifar10', train=False, batch_size=2000)

    total_train_steps = math.ceil(len(train_loader) * epochs)
    lr_schedule = np.interp(np.arange(1+total_train_steps),
                            [0, int(0.23 * total_train_steps), total_train_steps],
                            [0.2, 1, 0.07]) 

    model = make_net()
    model_ema = None

    nonbias_params = [p for k, p in model.named_parameters() if p.requires_grad and 'bias' not in k]
    bias_params = [p for k, p in model.named_parameters() if p.requires_grad and 'bias' in k]
    hyp_nonbias = dict(params=nonbias_params, lr=(lr / loss_scale), weight_decay=(wd * loss_scale))
    hyp_bias = dict(params=bias_params, lr=(lr * bias_scaler / loss_scale), weight_decay=(wd * loss_scale / bias_scaler))
    optimizer = torch.optim.SGD([hyp_nonbias, hyp_bias], momentum=momentum, nesterov=True)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule.__getitem__)

    ## For accurately timing GPU code
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)

    ## Initialize the whitening layer using training images
    starter.record()
    train_images = train_loader.normalize(train_loader.images[:5000])
    init_net(model, train_images)
    ender.record()
    torch.cuda.synchronize()
    total_time_seconds += 1e-3 * starter.elapsed_time(ender)

    for epoch in range(math.ceil(epochs)):

        ####################
        #     Training     #
        ####################

        starter.record()

        model.train()
        for inputs, labels in train_loader:

            outputs = model(inputs)
            loss = loss_fn(outputs, labels).sum()
            optimizer.zero_grad(set_to_none=True)
            (loss_scale * loss).backward()
            optimizer.step()
            scheduler.step()

            current_steps += 1

            if epoch >= hyp['ema']['start_epochs'] and current_steps % hyp['ema']['every_n_steps'] == 0:          
                if model_ema is None:
                    model_ema = NetworkEMA(model)
                else:
                    # We warm up our ema's decay/momentum value over training (this lets us move fast, then average strongly at the end).
                    rho = hyp['ema']['decay_base'] ** hyp['ema']['every_n_steps']
                    model_ema.update(model, decay=rho*(current_steps/total_train_steps)**hyp['ema']['decay_pow'])

            if current_steps >= total_train_steps:
                break

        ender.record()
        torch.cuda.synchronize()
        total_time_seconds += 1e-3 * starter.elapsed_time(ender)
        
        ####################
        #    Evaluation    #
        ####################

        # save the accuracy and loss from the last training batch of the epoch
        train_acc = (outputs.detach().argmax(-1) == labels).float().mean().item()
        train_loss = loss.item() / batch_size

        model.eval()
        with torch.no_grad():
            loss_list, acc_list, acc_list_ema = [], [], []
            for inputs, labels in test_loader:
                outputs = model(inputs)
                loss_list.append(loss_fn(outputs, labels).float().mean())
                acc_list.append((outputs.argmax(-1) == labels).float().mean())
                if model_ema:
                    outputs = model_ema(inputs)
                    acc_list_ema.append((outputs.argmax(-1) == labels).float().mean())
            val_acc = torch.stack(acc_list).mean().item()
            val_loss = torch.stack(loss_list).mean().item()
            ema_val_acc = None
            if model_ema:
                ema_val_acc = torch.stack(acc_list_ema).mean().item()
        tta_ema_val_acc = None

        print_training_details(locals(), is_final_entry=False)
        run = None # Only print the run number once

    ####################
    #  TTA Evaluation  #
    ####################

    with torch.no_grad():

        ## Test-time augmentation strategy:
        ## 1. Flip (mirror) the image left-to-right (50% of the time).
        ## 2. Then jitter the image by one pixel (50% of the time, i.e. both happen 25% of the time).
        ##
        ## This creates 8 inputs per image (left/right times the four directions),
        ## which we evaluate and then weight according to the given probabilities.

        assert model_ema 

        starter.record()

        pad = 1
        padded_images = F.pad(test_loader.images, (pad,)*4, 'reflect')
        images = [test_loader.images]
        images.append(padded_images[:, :, 0:32, 0:32])
        images.append(padded_images[:, :, 0:32, 2:34])
        images.append(padded_images[:, :, 2:34, 0:32])
        images.append(padded_images[:, :, 2:34, 2:34])
        images_tta = test_loader.normalize(torch.cat(images))
        labels = test_loader.labels

        outputs_list = []
        for inputs in images_tta.split(2000):
            outputs = (0.5 * model_ema(inputs) + 0.5 * model_ema(inputs.flip(-1)))
            outputs_list.append(outputs)
        outputs = torch.cat(outputs_list)
        outputs = outputs.view(-1, len(labels), 10)
        
        logits_mirror = outputs[0]
        logits_mirror_jitter = outputs[1:].mean(0)
        logits_tta = (0.5 * logits_mirror + 0.5 * logits_mirror_jitter)

        tta_ema_val_acc = (logits_tta.argmax(1) == labels).float().mean().item()

        ender.record()
        torch.cuda.synchronize()
        total_time_seconds += 1e-3 * starter.elapsed_time(ender)

    epoch = 'eval'
    print_training_details(locals(), is_final_entry=True)

    return tta_ema_val_acc


if __name__ == "__main__":
    with open(sys.argv[0]) as f:
        code = f.read()

    print_columns(logging_columns_list, is_head=True)
    accs = torch.tensor([main(run) for run in range(20)])
    print('Mean: %.4f    Std: %.4f' % (accs.mean(), accs.std()))

    log = {'code': code, 'accs': accs}
    log_dir = os.path.join('logs', str(uuid.uuid4()))
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, 'log.pt')
    print(log_path)
    torch.save(log, os.path.join(log_dir, 'log.pt'))

