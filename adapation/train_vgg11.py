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

import airbench

hyp = {
    'opt': {
        'epochs': 50,
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
        'norm': 'none',
    }
}

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

def train(train_loader, test_loader=None,
          epochs=hyp['opt']['epochs'], lr=hyp['opt']['lr']):

    if test_loader is None:
        test_loader = airbench.CifarLoader('cifar10', train=False, batch_size=1000)
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

        test_acc.append(airbench.evaluate(model, test_loader))
        it.set_description('Acc=%.4f(train),%.4f(test)' % (train_acc[-1], test_acc[-1]))

    log = dict(train_loss=train_loss, train_acc=train_acc, test_acc=test_acc)
    return model, log 

if __name__ == '__main__':

    train_loader = airbench.CifarLoader('cifar10', train=True, batch_size=hyp['opt']['batch_size'], aug=hyp['aug'])
    model, log = train(train_loader)
    print('Final acc: %.4f' % log['test_acc'][-1])

    import uuid
    k = str(uuid.uuid4())
    torch.save(model.state_dict(), k+'.pt')

