# A variant of airbench optimized for time-to-96%.
# 46.3s runtime on an A100; 7.46 PFLOPs.
# Evidence: 96.05 average accuracy in n=200 runs.
#
# We recorded the runtime of 46.3 seconds on an NVIDIA A100-SXM4-80GB with the following nvidia-smi:
# NVIDIA-SMI 515.105.01   Driver Version: 515.105.01   CUDA Version: 11.7
# torch.__version__ == '2.1.2+cu118'
#
# Changes relative to airbench:
# - Doubled width and reduced learning rate.
# - Added extra layer to each ConvBlock. The network now contains 10 conv layers.
# - Added residual connections over the last two conv layers in each ConvBlock.
# - Added 12-pixel cutout data augmentation and increased random-translation strength from 2 to 4 pixels.
# - Increased training duration to 40 epochs.
#
# If random flip is used instead of alternating, then decays to 96.01 average accuracy in n=200 runs.

#############################################
#            Setup/Hyperparameters          #
#############################################

import os
import sys
import uuid
from math import ceil

import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T

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
        'train_epochs': 40.0,
        'batch_size': 1024,
        'lr': 9.0,                  # learning rate per 1024 examples
        'momentum': 0.85,
        'weight_decay': 0.0153,     # weight decay per 1024 examples (decoupled from learning rate)
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
        'translate': 4,
        'cutout': 12,
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

def batch_cutout(inputs, size):
    cutout_masks = make_random_square_masks(inputs, size)
    return inputs.masked_fill(cutout_masks, 0)

class CifarLoader:

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
            assert k in ['flip', 'translate', 'cutout'], 'Unrecognized key: %s' % k

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
        if self.aug.get('cutout', 0) > 0:
            images = batch_cutout(images, self.aug['cutout'])

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
        self.conv3 = Conv(channels_out, channels_out)
        self.norm3 = BatchNorm(channels_out)
        self.activ = nn.GELU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.norm1(x)
        x = self.activ(x)
        x0 = x
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activ(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.activ(x)
        x += x0
        return x

#############################################
#            Network Definition             #
#############################################

def make_net():
    widths = {
        'block1': (2 * hyp['net']['base_width']), # 128 w/ width at base value
        'block2': (8 * hyp['net']['base_width']), # 512 w/ width at base value
        'block3': (8 * hyp['net']['base_width']), # 512 w/ width at base value
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
    net = net.half().cuda()
    net = net.to(memory_format=torch.channels_last)
    for mod in net.modules():
        if isinstance(mod, BatchNorm):
            mod.float()
    return net

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

def main(run):

    batch_size = hyp['opt']['batch_size']
    epochs = hyp['opt']['train_epochs']
    momentum = hyp['opt']['momentum']
    # Assuming  gradients are constant in time, for Nesterov momentum, the below ratio is how much
    # larger the default steps will be than the underlying per-example gradients. We divide the
    # learning rate by this ratio in order to ensure steps are the same scale as gradients, regardless
    # of the choice of momentum.
    kilostep_scale = 1024 * (1 + 1 / (1 - momentum))
    lr = hyp['opt']['lr'] / kilostep_scale # un-decoupled learning rate for PyTorch SGD
    wd = hyp['opt']['weight_decay'] * batch_size / kilostep_scale
    lr_biases = lr * hyp['opt']['bias_scaler']

    loss_fn = nn.CrossEntropyLoss(label_smoothing=hyp['opt']['label_smoothing'], reduction='none')

    train_augs = dict(flip=hyp['aug']['flip'], translate=hyp['aug']['translate'], cutout=hyp['aug']['cutout'])
    train_loader = CifarLoader('cifar10', train=True, batch_size=batch_size, aug=train_augs)
    test_loader = CifarLoader('cifar10', train=False, batch_size=2000)
    if run == 'warmup':
        # The only purpose of the first run is to warmup, so we can use dummy data
        train_loader.labels = torch.randint(0, 10, size=(len(train_loader.labels),), device=train_loader.labels.device)

    def triangle(steps, start=0, end=0, peak=0.5):
        xp = torch.tensor([0, int(peak * steps), steps])
        fp = torch.tensor([start, 1, end])
        x = torch.arange(1+steps)
        m = (fp[1:] - fp[:-1]) / (xp[1:] - xp[:-1])
        b = fp[:-1] - (m * xp[:-1])
        indices = torch.sum(torch.ge(x[:, None], xp[None, :]), 1) - 1
        indices = torch.clamp(indices, 0, len(m) - 1)
        return m[indices] * x + b[indices]

    total_train_steps = ceil(len(train_loader) * epochs)
    lr_schedule = triangle(total_train_steps, start=0.2, end=0.07, peak=0.23)

    model = make_net()
    lookahead_state = None
    current_steps = 0

    norm_biases = [p for k, p in model.named_parameters() if 'norm' in k and p.requires_grad]
    other_params = [p for k, p in model.named_parameters() if 'norm' not in k and p.requires_grad]
    param_configs = [dict(params=norm_biases, lr=lr_biases, weight_decay=wd/lr_biases),
                     dict(params=other_params, lr=lr, weight_decay=wd/lr)]
    optimizer = torch.optim.SGD(param_configs, momentum=momentum, nesterov=True)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda i: lr_schedule[i])

    # For accurately timing GPU code
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)

    total_time_seconds = 0.0

    # Initialize the whitening layer using training images
    starter.record()
    train_images = train_loader.normalize(train_loader.images[:5000])
    init_whitening_conv(model[0], train_images)
    ender.record()
    torch.cuda.synchronize()
    total_time_seconds += 1e-3 * starter.elapsed_time(ender)

    for epoch in range(ceil(epochs)):

        model[0].bias.requires_grad = (epoch < hyp['opt']['whiten_bias_epochs'])

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
        # 2. Translate the image by one pixel either up-and-left or down-and-right (50% of the time,
        #    i.e. both happen 25% of the time).
        #
        # This creates 6 views per image (left/right times the two translations and no-translation),
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

    print_columns(logging_columns_list, is_head=True)
    #main('warmup')
    accs = torch.tensor([main(run) for run in range(25)])
    print('Mean: %.4f    Std: %.4f' % (accs.mean(), accs.std()))

    log = {'code': code, 'accs': accs}
    log_dir = os.path.join('logs', str(uuid.uuid4()))
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, 'log.pt')
    print(os.path.abspath(log_path))
    torch.save(log, os.path.join(log_dir, 'log.pt'))

