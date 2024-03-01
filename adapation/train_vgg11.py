#############################################
#            Setup/Hyperparameters          #
#############################################

import os
import sys
import uuid
import math
import numpy as np
from tqdm import tqdm

import torch
from torch import nn
import torch.nn.functional as F
torch.backends.cudnn.benchmark = True

hyp = {
    'opt': {
        'epochs': 40,
        'batch_size': 500,
        'lr': 0.1,
        'momentum': 0.9,
        'wd': 5e-4,
    },
    'aug': {
        'flip': True,
        'translate': 4,
        'cutout': 0,
    },
    'net': {
        'norm': 'ln',
    }
}

## https://github.com/KellerJordan/cifar10-loader/blob/master/quick_cifar/loader.py
import os
from math import ceil
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T

CIFAR_MEAN = torch.tensor((0.4914, 0.4822, 0.4465))
CIFAR_STD = torch.tensor((0.2470, 0.2435, 0.2616))

# https://github.com/tysam-code/hlb-CIFAR10/blob/main/main.py#L389
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

def batch_translate(inputs, translate):
    width = inputs.shape[-2]
    padded_inputs = F.pad(inputs, (translate,)*4, 'constant', value=0)
    return batch_crop(padded_inputs, width)

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
        self.denormalize = T.Normalize(-CIFAR_MEAN / CIFAR_STD, 1 / CIFAR_STD)
        
        self.aug = aug or {}
        for k in self.aug.keys():
            assert k in ['flip', 'translate', 'cutout'], 'Unrecognized key: %s' % k

        self.batch_size = batch_size
        self.drop_last = train if drop_last is None else drop_last
        self.shuffle = train if shuffle is None else shuffle

    def save(self, path):
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        obj = {'images': self.images, 'labels': self.labels}
        torch.save(obj, path)

    def load(self, path):
        obj = torch.load(path)
        self.images = obj['images'].half().cuda()
        self.labels = obj['labels'].cuda()
        return self

    def augment(self, images):
        if self.aug.get('flip', False):
            images = batch_flip_lr(images)
        if self.aug.get('cutout', 0) > 0:
            images = batch_cutout(images, self.aug['cutout'])
        if self.aug.get('translate', 0) > 0:
            # Apply translation in minibatches in order to save memory
            images = torch.cat([batch_translate(image_batch, self.aug['translate'])
                                for image_batch in images.split(5000)])
        return images

    def __len__(self):
        return len(self.images)//self.batch_size if self.drop_last else ceil(len(self.images)/self.batch_size)

    def __iter__(self):
        images = self.augment(self.normalize(self.images))
        indices = (torch.randperm if self.shuffle else torch.arange)(len(images), device=images.device)
        for i in range(len(self)):
            idxs = indices[i*self.batch_size:(i+1)*self.batch_size]
            yield (images[idxs], self.labels[idxs])

import torch
from torch import nn

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(len(x), -1)
    
def make_vgg(cfg, norm):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            layers.append(nn.Conv2d(in_channels, v, kernel_size=3, padding=1, bias=(norm == 'none')))
            if norm == 'bn':
                layers.append(nn.BatchNorm2d(v))
            elif norm == 'ln':
                layers.append(nn.GroupNorm(1, v))
            layers.append(nn.ReLU(inplace=True))
            in_channels = v
    layers.append(nn.MaxPool2d(4))
    layers.append(Flatten())
    layers.append(nn.Linear(512, 10))
    return nn.Sequential(*layers)

def make_net(norm='none'):
    model = make_vgg([64, 'M', 128, 'M', 256, 256, 'M', 512, 512], norm=norm)
    model = model.to(memory_format=torch.channels_last)
    model = model.cuda()
    for m in model.modules():
        if type(m) is not nn.BatchNorm2d:
            m.half()
    return model

########################################
#           Train and Eval             #
########################################

def evaluate(model, loader):
    assert not loader.shuffle
    model.eval()
    with torch.no_grad():
        outs = torch.cat([model(inputs) + model(inputs.flip(-1)) for inputs, _ in loader])
    return (outs.argmax(1) == loader.labels).float().mean().item()

def train(train_loader, test_loader=None,
          epochs=hyp['opt']['epochs'], lr=hyp['opt']['lr']):

    if test_loader is None:
        test_loader = CifarLoader('cifar10', train=False, batch_size=1000)
    batch_size = train_loader.batch_size

    momentum = hyp['opt']['momentum']
    wd = hyp['opt']['wd']

    total_train_steps = len(train_loader) * epochs
    lr_schedule = np.interp(np.arange(1+total_train_steps),
                            [0, int(0.2 * total_train_steps), total_train_steps],
                            [0.2, 1, 0]) # Triangular learning rate schedule

    model = make_net(norm=hyp['net']['norm'])
    optimizer = torch.optim.SGD(model.parameters(), lr=lr/batch_size, momentum=momentum, nesterov=True,
                                weight_decay=wd*batch_size)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule.__getitem__)

    train_loss, train_acc, test_acc = [torch.nan], [torch.nan], [torch.nan]

    it = tqdm(range(epochs))
    for epoch in it:

        model.train()
        for inputs, labels in train_loader:
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, labels, reduction='none')
            train_loss.append(loss.mean().item())
            train_acc.append((outputs.detach().argmax(1) == labels).float().mean().item())
            optimizer.zero_grad(set_to_none=True)
            loss.sum().backward()
            optimizer.step()
            scheduler.step()
            it.set_description('Acc=%.4f(train),%.4f(test)' % (train_acc[-1], test_acc[-1]))

        test_acc.append(evaluate(model, test_loader))
        it.set_description('Acc=%.4f(train),%.4f(test)' % (train_acc[-1], test_acc[-1]))

    log = dict(train_loss=train_loss, train_acc=train_acc, test_acc=test_acc)
    return model, log 

if __name__ == '__main__':

    train_augs = dict(flip=hyp['aug']['flip'], translate=hyp['aug']['translate'], cutout=hyp['aug']['cutout'])
    train_loader = CifarLoader('cifar10', train=True, batch_size=hyp['opt']['batch_size'], aug=train_augs)

    model, log = train(train_loader)
    print('Final acc: %.4f' % log['test_acc'][-1])

    import uuid
    k = str(uuid.uuid4())
    torch.save(model.state_dict(), k+'.pt')

