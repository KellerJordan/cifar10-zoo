# airbench_cifar10.py
#
# This script is designed to reach 94% accuracy on the CIFAR-10 test-set in the shortest possible time
# after first seeing the training set. It runs in 3.5 seconds on a single NVIDIA A100.
#
# We use the following methods:
#
# 1. Our network architecture is an 8-layer convnet with whitening and identity initialization.
#    * Following Page (2018), the first convolution is initialized as a frozen patch-whitening layer
#      using statistics from the training images. Additionally, the logit output is downscaled and
#      BatchNorm affine weights are disabled.
#    * Following hlb-CIFAR10, the whitening layer has patch size 2, precedes an activation, and is
#      concatenated with its negation to ensure completeness. The six remaining convolutional layers
#      lack residual connections and are initialized as identity transforms wherever possible. The
#      8-layer architecture is also following hlb-CIFAR10. We use reduced width in the final layer.
#    * We add a learnable bias to the whitening layer, which reduces the number of steps to 94% by
#      5-10%. We find it converges quickly, so we save time by freezing it after 3 epochs.
# 2. For test-time augmentation we use standard horizontal flipping. We also use one-pixel translation
#    to the upper-left and lower-right, for a total of six forward passes per example.
# 3. For training data augmentation we use horizontal flipping and random two-pixel translation. For
#    horizontal flipping we follow a novel scheme. At epoch one images are randomly flipped as usual.
#    At epoch two we flip exactly those images which weren't flipped in the first epoch. Then epoch
#    three flips the same images as epoch one, four the same as two, and so on. We find that this
#    decreases the number of steps to 94% accuracy by roughly 10%. We hypothesize that this is because
#    the standard fully random flipping is wasteful in the sense that (e.g.,) 1/8 of images will be
#    flipped the same way for the first four epochs, effectively resulting in less new images seen
#    per epoch as compared to our semi-deterministic alternating scheme.
# 4. Following Page (2018), we use Nesterov SGD with a triangular learning rate schedule and increased
#    learning rate for BatchNorm biases. On top of this, following hlb-CIFAR10, we use a lookahead-
#    like scheme with slow decay rate at the end of training, which saves an extra 0.35 seconds.
# 5. Following hlb-CIFAR10, we use a low momentum of 0.6 for running BatchNorm stats, which we find
#    yields more accurate estimates for very short trainings than the standard setting of 0.9.
# 6. We use GPU-accelerated dataloading and augmentation. A generic fast CIFAR-10 dataloader can be
#    found at https://github.com/KellerJordan/cifar10-loader.
# 7. We use torch.compile with mode='max-autotune'.
#
# To confirm that the mean accuracy is above 94%, we ran a test of n=700 runs, which yielded an
# average accuracy of 94.02% (p<0.0001 for the true mean being below 94%, via t-test).
#
# We recorded the runtime of 3.5 seconds on an NVIDIA A100-SXM4-80GB with the following nvidia-smi:
# NVIDIA-SMI 515.105.01   Driver Version: 515.105.01   CUDA Version: 11.7
# torch.__version__ == '2.1.2+cu118'
#
# Note that the first time this script is run, compilation takes up to two minutes. Without the usage
# of torch.compile, this script warms up in <10 seconds and takes 3.83 seconds per run.
#
# The 8-layer convnet we train has 2M parameters and uses 0.24 GFLOPs per forward pass. The entire
# training run uses 366 TFLOPs, which could theoretically take 1.17 A100-seconds at perfect utilization.
#
# For comparison, version 0.7.0 of https://github.com/tysam-code/hlb-CIFAR10 uses 587 TFLOPs and runs in
# 6.2 seconds. The final training script from David Page's series "How to Train Your ResNet" (Page 2018)
# uses 1,148 TFLOPs and runs in 15.1 seconds (on an A100). And the standard 200-epoch ResNet18 training
# on CIFAR-10 uses ~30,000 TFLOPs and runs in minutes.
#
# This script is descended from https://github.com/tysam-code/hlb-CIFAR10 [1], which itself is descended
# from David Page's training script [2]. The latter was the winning submission to the Stanford DAWNbench
# competition for CIFAR-10 in 2018, with a time of 26 seconds to 94% accuracy on an NVIDIA V100.
#
# 1. tysam-code. "CIFAR-10 hyperlightspeedbench." https://github.com/tysam-code/hlb-CIFAR10. Jan 01 (2024).
# 2. Page, David. "How to train your resnet." Myrtle, https://myrtle.ai/learn/how-to-train-your-resnet-8-bag-of-tricks/. Sept 24 (2018).

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

# We express the main training hyperparameters (batch size, learning rate, momentum, and weight decay)
# in decoupled form, so that each one can be tuned independently. This accomplishes the following:
# * Assuming time-constant gradients, the average step size is decoupled from everything but the lr.
# * The size of the weight decay update is decoupled from everything but the wd.
# In constrast, normally when we increase the (Nesterov) momentum, this also scales up the step size
# proportionally to 1 + 1 / (1 - momentum), meaning we cannot change momentum without having to re-tune
# the learning rate. Similarly, normally when we increase the learning rate this also increases the size
# of the weight decay, requiring a proportional decrease in the wd to maintain the same decay strength.
#
# The practical impact is that hyperparameter tuning is faster, since this parametrization allows each
# one to be tuned independently. See https://myrtle.ai/learn/how-to-train-your-resnet-5-hyperparameters/.

hyp = {
    'opt': {
        'train_epochs': 9.9,
        'batch_size': 1024,
        'lr': 11.5,                 # units: learning rate per 1024 examples
        'momentum': 0.85,           # units: decay per 1024 examples (e.g. batch_size=512 gives sqrt of this)
        'weight_decay': 0.0153,     # units: weight decay per 1024 examples (decoupled from learning rate)
        'bias_scaler': 64.0,        # scales up learning rate (but not weight decay) for BatchNorm biases
        'label_smoothing': 0.2,
        'ema': {
            'start_epochs': 3,
            'decay_base': 0.95,
            'decay_pow': 3.,
            'every_n_steps': 5,
        },
        'whiten_bias_epochs': 3,    # how many epochs to train the whitening layer bias before freezing
    },
    'aug': {
        'flip': True,
        'translate': 2,
    },
    'net': {
        'whitening': {
            'kernel_size': 2,
        },
        'batchnorm_momentum': 0.6,
        'base_width': 64,
        'scaling_factor': 1/9,
        'tta_level': 2,         # the level of test-time augmentation: 0=none, 1=mirror, 2=mirror+translate
    },
}

#############################################
#                DataLoader                 #
#############################################

# https://github.com/KellerJordan/cifar10-loader/blob/master/quick_cifar/loader.py
import os
from math import ceil
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T

CIFAR_MEAN = torch.tensor((0.4914, 0.4822, 0.4465))
CIFAR_STD = torch.tensor((0.2470, 0.2435, 0.2616))

def batch_flip_lr(inputs):
    flip_mask = (torch.rand(len(inputs), device=inputs.device) < 0.5).view(-1, 1, 1, 1)
    return torch.where(flip_mask, inputs.flip(-1), inputs)

def batch_crop(images, crop_size):
    r = (images.size(-1) - crop_size)//2
    shifts = torch.randint(-r, r+1, size=(len(images), 2), device=images.device)
    images_out = torch.empty((len(images), 3, crop_size, crop_size), device=images.device, dtype=images.dtype)
    # The two cropping methods in this if-else produce equivalent results, but the second is faster for r > 2.
    if r <= 2:
        for sy in range(-r, r+1):
            for sx in range(-r, r+1):
                mask = (shifts[:, 0] == sy) & (shifts[:, 1] == sx)
                images_out[mask] = images[mask, :, r+sy:r+sy+crop_size, r+sx:r+sx+crop_size]
    else:
        images_tmp = torch.empty((len(images), 3, crop_size, crop_size+2*r), device=images.device, dtype=images.dtype)
        for s in range(-r, r+1):
            mask = (shifts[:, 0] == s)
            images_tmp[mask] = images[mask, :, r+s:r+s+crop_size, :]
        for s in range(-r, r+1):
            mask = (shifts[:, 1] == s)
            images_out[mask] = images_tmp[mask, :, :, r+s:r+s+crop_size]
    return images_out

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
        self.proc_images = {} # Saved results of image processing to be done on the first epoch
        self.epoch = 0

        self.aug = aug or {}
        for k in self.aug.keys():
            assert k in ['flip', 'translate'], 'Unrecognized key: %s' % k

        self.batch_size = batch_size
        self.drop_last = train if drop_last is None else drop_last
        self.shuffle = train if shuffle is None else shuffle

    def __len__(self):
        return len(self.images)//self.batch_size if self.drop_last else ceil(len(self.images)/self.batch_size)

    def __iter__(self):

        if self.epoch == 0:
            images = self.proc_images['norm'] = self.normalize(self.images)
            # Pre-flip images in order to do every-other epoch flipping scheme
            if self.aug.get('flip', False):
                images = self.proc_images['flip'] = batch_flip_lr(images)
            # Pre-pad images to save time when doing random translation
            pad = self.aug.get('translate', 0)
            if pad > 0:
                self.proc_images['pad'] = F.pad(images, (pad,)*4, 'reflect')

        if self.aug.get('translate', 0) > 0:
            images = batch_crop(self.proc_images['pad'], self.images.shape[-2])
        elif self.aug.get('flip', False):
            images = self.proc_images['flip']
        else:
            images = self.proc_images['norm']
        # Flip all images together every other epoch. This increases diversity relative to random flipping
        if self.aug.get('flip', False):
            if self.epoch % 2 == 1:
                images = images.flip(-1)

        self.epoch += 1

        indices = (torch.randperm if self.shuffle else torch.arange)(len(images), device=images.device)
        for i in range(len(self)):
            idxs = indices[i*self.batch_size:(i+1)*self.batch_size]
            yield (images[idxs], self.labels[idxs])

#############################################
#            Network Components             #
#############################################

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class Mul(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale
    def forward(self, x):
        return x * self.scale

class BatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-12, momentum=hyp['net']['batchnorm_momentum'],
                 weight=False, bias=True):
        super().__init__(num_features, eps=eps, momentum=1-momentum)
        self.weight.requires_grad = weight
        self.bias.requires_grad = bias
        # Note that PyTorch already initializes the weights to one and bias to zero

class Conv(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding='same', bias=False):
        super().__init__(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=bias)

    def reset_parameters(self):
        super().reset_parameters()
        if self.bias is not None:
            self.bias.data.zero_()
        # Create an implicit residual via identity initialization
        w = self.weight.data
        torch.nn.init.dirac_(w[:w.size(1)])

class ConvGroup(nn.Module):
    def __init__(self, channels_in, channels_out):
        super().__init__()
        self.conv1 = Conv(channels_in,  channels_out)
        self.pool = nn.MaxPool2d(2)
        self.norm1 = BatchNorm(channels_out)
        self.conv2 = Conv(channels_out, channels_out)
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

#############################################
#            Network Definition             #
#############################################

def make_net():
    widths = {
        'block1': (1 * hyp['net']['base_width']), # 64  w/ width at base value
        'block2': (4 * hyp['net']['base_width']), # 256 w/ width at base value
        'block3': (4 * hyp['net']['base_width']), # 256 w/ width at base value
    }
    whiten_conv_width = 2 * 3 * hyp['net']['whitening']['kernel_size']**2
    net = nn.Sequential(
        Conv(3, whiten_conv_width, kernel_size=hyp['net']['whitening']['kernel_size'], padding=0, bias=True),
        nn.GELU(),
        ConvGroup(whiten_conv_width, widths['block1']),
        ConvGroup(widths['block1'],  widths['block2']),
        ConvGroup(widths['block2'],  widths['block3']),
        nn.MaxPool2d(3),
        Flatten(),
        nn.Linear(widths['block3'], 10, bias=False),
        Mul(hyp['net']['scaling_factor']),
    )
    net[0].weight.requires_grad = False
    net = net.cuda()
    net = net.to(memory_format=torch.channels_last)
    net.half()
    for mod in net.modules():
        if isinstance(mod, BatchNorm):
            mod.float()
    return net

def reinit_net(model):
    for m in model.modules():
        if type(m) in (Conv, BatchNorm, nn.Linear):
            m.reset_parameters()

#############################################
#       Whitening Conv Initialization       #
#############################################

def get_patches(x, patch_shape):
    c, (h, w) = x.shape[1], patch_shape
    return x.unfold(2,h,1).unfold(3,w,1).transpose(1,3).reshape(-1,c,h,w).float()

def get_whitening_parameters(patches):
    n,c,h,w = patches.shape
    patches_flat = patches.view(n, -1)
    est_patch_covariance = (patches_flat.T @ patches_flat) / n
    eigenvalues, eigenvectors = torch.linalg.eigh(est_patch_covariance, UPLO='U')
    return eigenvalues.flip(0).view(-1, 1, 1, 1), eigenvectors.T.reshape(c*h*w,c,h,w).flip(0)

def init_whitening_conv(layer, train_set, eps=5e-4):
    patches = get_patches(train_set, patch_shape=layer.weight.data.shape[2:])
    eigenvalues, eigenvectors = get_whitening_parameters(patches)
    eigenvectors_scaled = eigenvectors / torch.sqrt(eigenvalues + eps)
    layer.weight.data[:] = torch.cat((eigenvectors_scaled, -eigenvectors_scaled))

############################################
#                Lookahead                 #
############################################

class LookaheadState:

    def __init__(self, net):
        self.net_ema = {k: v.clone() for k, v in net.state_dict().items()}

    def update(self, net, decay):
        for ema_param, net_param in zip(self.net_ema.values(), net.state_dict().values()):
            if net_param.dtype in (torch.half, torch.float):
                ema_param.lerp_(net_param, 1-decay)
                # Copy the ema parameters back to the network, similarly to the Lookahead optimizer
                net_param.copy_(ema_param)

############################################
#                 Logging                  #
############################################

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

logging_columns_list = ['run   ', 'epoch', 'train_loss', 'val_loss', 'train_acc', 'val_acc', 'tta_val_acc', 'total_time_seconds']
def print_training_details(variables, is_final_entry):
    formatted = []
    for col in logging_columns_list:
        var = variables.get(col.strip(), None)
        if type(var) in (int, str):
            res = str(var)
        elif type(var) is float:
            res = '{:0.4f}'.format(var)
        else:
            assert var is None
            res = ''
        formatted.append(res.rjust(len(col)))
    print_columns(formatted, is_final_entry=is_final_entry)

############################################
#             Train and Eval               #
############################################

def main(run, model_trainbias, model_freezebias):

    batch_size = hyp['opt']['batch_size']
    epochs = hyp['opt']['train_epochs']
    momentum = hyp['opt']['momentum']
    # Assuming  gradients are constant in time, for Nesterov momentum, the below ratio is how much
    # larger the default steps will be than the underlying per-example gradients. We divide the
    # learning rate by this ratio in order to ensure steps are the same scale as gradients, regardless
    # of the choice of momentum.
    kilostep_scale = 1024 * (1 + 1 / (1 - momentum))
    lr = hyp['opt']['lr'] / kilostep_scale # denormalized learning rate to give to PyTorch SGD
    wd = hyp['opt']['weight_decay'] * batch_size / kilostep_scale
    lr_biases = lr * hyp['opt']['bias_scaler']

    train_augs = dict(flip=hyp['aug']['flip'], translate=hyp['aug']['translate'])
    loss_fn = nn.CrossEntropyLoss(label_smoothing=hyp['opt']['label_smoothing'], reduction='none')

    train_loader = PrepadCifarLoader('/tmp/cifar10', train=True, batch_size=batch_size, aug=train_augs)
    test_loader = PrepadCifarLoader('/tmp/cifar10', train=False, batch_size=2000)
    if run == 'warmup':
        # The only purpose of the first run is to warmup the compiled model, so we can use dummy data
        train_loader.labels = torch.randint(0, 10, size=(len(train_loader.labels),), device=train_loader.labels.device)

    total_train_steps = math.ceil(len(train_loader) * epochs)
    lr_schedule = np.interp(np.arange(1+total_train_steps),
                            [0, int(0.23 * total_train_steps), total_train_steps],
                            [0.2, 1, 0.07]) # triangular learning rate schedule

    # Reinitialize the network from scratch - nothing is reused from previous runs besides the PyTorch compilation
    reinit_net(model_trainbias)
    lookahead_state = None
    current_steps = 0

    norm_biases = [p for k, p in model_trainbias.named_parameters() if 'norm' in k]
    other_params = [p for k, p in model_trainbias.named_parameters() if 'norm' not in k]
    param_configs = [dict(params=norm_biases, lr=lr_biases, weight_decay=wd/lr_biases),
                     dict(params=other_params, lr=lr, weight_decay=wd/lr)]
    optimizer_trainbias = torch.optim.SGD(param_configs, momentum=momentum, nesterov=True)
    scheduler_trainbias = torch.optim.lr_scheduler.LambdaLR(optimizer_trainbias, lambda i: lr_schedule[i])

    norm_biases = [p for k, p in model_freezebias.named_parameters() if 'norm' in k]
    other_params = [p for k, p in model_freezebias.named_parameters() if 'norm' not in k]
    param_configs = [dict(params=norm_biases, lr=lr_biases, weight_decay=wd/lr_biases),
                     dict(params=other_params, lr=lr, weight_decay=wd/lr)]
    optimizer_freezebias = torch.optim.SGD(param_configs, momentum=momentum, nesterov=True)
    scheduler_freezebias = torch.optim.lr_scheduler.LambdaLR(optimizer_freezebias, lambda i: lr_schedule[i])

    # For accurately timing GPU code
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)

    total_time_seconds = 0.0

    # Initialize the whitening layer using training images
    starter.record()
    train_images = train_loader.normalize(train_loader.images[:5000])
    init_whitening_conv(model_trainbias._orig_mod[0], train_images)
    ender.record()
    torch.cuda.synchronize()
    total_time_seconds += 1e-3 * starter.elapsed_time(ender)

    for epoch in range(math.ceil(epochs)):

        # After training the whiten bias for some epochs, swap in the compiled model with frozen bias
        if epoch == 0:
            model = model_trainbias
            optimizer = optimizer_trainbias
            scheduler = scheduler_trainbias
        elif epoch == hyp['opt']['whiten_bias_epochs']:
            model = model_freezebias
            optimizer = optimizer_freezebias
            scheduler = scheduler_freezebias
            model.load_state_dict(model_trainbias.state_dict())
            optimizer.load_state_dict(optimizer_trainbias.state_dict())
            scheduler.load_state_dict(scheduler_trainbias.state_dict())
        
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

            current_steps += 1

            if epoch >= hyp['opt']['ema']['start_epochs'] and current_steps % hyp['opt']['ema']['every_n_steps'] == 0:          
                if lookahead_state is None:
                    lookahead_state = LookaheadState(model)
                else:
                    # We warm up our ema's decay/momentum value over training (this lets us move fast, then average strongly at the end).
                    base_rho = hyp['opt']['ema']['decay_base'] ** hyp['opt']['ema']['every_n_steps']
                    rho = base_rho * (current_steps / total_train_steps) ** hyp['opt']['ema']['decay_pow']
                    lookahead_state.update(model, decay=rho)

            if current_steps >= total_train_steps:
                break

        if lookahead_state is not None:
            # Copy back parameters a final time after each epoch
            lookahead_state.update(model, decay=1.0)

        ender.record()
        torch.cuda.synchronize()
        total_time_seconds += 1e-3 * starter.elapsed_time(ender)
        
        ####################
        #    Evaluation    #
        ####################

        # Save the accuracy and loss from the last training batch of the epoch
        train_acc = (outputs.detach().argmax(1) == labels).float().mean().item()
        train_loss = loss.item() / batch_size

        model.eval()
        with torch.no_grad():
            loss_list, acc_list = [], []
            for inputs, labels in test_loader:
                outputs = model(inputs)
                loss_list.append(loss_fn(outputs, labels).float().mean())
                acc_list.append((outputs.argmax(1) == labels).float().mean())
            val_acc = torch.stack(acc_list).mean().item()
            val_loss = torch.stack(loss_list).mean().item()
        tta_val_acc = None

        print_training_details(locals(), is_final_entry=False)
        run = None # Only print the run number once

    ####################
    #  TTA Evaluation  #
    ####################

    starter.record()

    with torch.no_grad():

        # Test-time augmentation strategy (for tta_level=2):
        # 1. Flip/mirror the image left-to-right (50% of the time).
        # 2. Translate the image by one pixel in any direction (50% of the time, i.e. both happen 25% of the time).
        #
        # This creates 8 inputs per image (left/right times the four directions),
        # which we evaluate and then weight according to the given probabilities.

        test_images = test_loader.normalize(test_loader.images)
        test_labels = test_loader.labels

        def infer_basic(inputs, net):
            return net(inputs).clone() # using .clone() here averts some kind of bug with torch.compile

        def infer_mirror(inputs, net):
            return 0.5 * net(inputs) + 0.5 * net(inputs.flip(-1))

        def infer_mirror_translate(inputs, net):
            logits = infer_mirror(inputs, net)
            pad = 1
            padded_inputs = F.pad(inputs, (pad,)*4, 'reflect')
            inputs_translate_list = [
                padded_inputs[:, :, 0:32, 0:32],
                padded_inputs[:, :, 2:34, 2:34],
            ]
            logits_translate_list = [infer_mirror(inputs_translate, net) for inputs_translate in inputs_translate_list]
            logits_translate = torch.stack(logits_translate_list).mean(0)
            return 0.5 * logits + 0.5 * logits_translate
            
        if hyp['net']['tta_level'] == 0:
            infer_fn = infer_basic
        elif hyp['net']['tta_level'] == 1:
            infer_fn = infer_mirror
        elif hyp['net']['tta_level'] == 2:
            infer_fn = infer_mirror_translate

        model.eval()
        logits_tta = torch.cat([infer_fn(inputs, model) for inputs in test_images.split(2000)])
        tta_val_acc = (logits_tta.argmax(1) == test_labels).float().mean().item()

    ender.record()
    torch.cuda.synchronize()
    total_time_seconds += 1e-3 * starter.elapsed_time(ender)

    epoch = 'eval'
    print_training_details(locals(), is_final_entry=True)

    return tta_val_acc

if __name__ == "__main__":
    with open(sys.argv[0]) as f:
        code = f.read()

    # These two compiled models are first warmed up, and then reinitialized every run. No learned
    # weights are reused between runs. To implement freezing of the whitening-layer bias parameter
    # midway through training, we use two compiled models, one with trainable and the other with
    # frozen whitening bias. This is faster than the naive approach of setting requires_grad=False
    # on the whitening bias midway through training on a single compiled model.
    model_trainbias = make_net()
    model_freezebias = make_net()
    model_freezebias[0].bias.requires_grad = False
    model_trainbias = torch.compile(model_trainbias, mode='max-autotune')
    model_freezebias = torch.compile(model_freezebias, mode='max-autotune')

    print_columns(logging_columns_list, is_head=True)
    main('warmup', model_trainbias, model_freezebias)
    accs = torch.tensor([main(run, model_trainbias, model_freezebias) for run in range(25)])
    print('Mean: %.4f    Std: %.4f' % (accs.mean(), accs.std()))

    log = {'code': code, 'accs': accs}
    log_dir = os.path.join('logs', str(uuid.uuid4()))
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, 'log.pt')
    print(os.path.abspath(log_path))
    torch.save(log, os.path.join(log_dir, 'log.pt'))

