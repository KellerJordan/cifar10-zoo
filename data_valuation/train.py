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

from loader import CifarLoader
from model import make_net

hyp = {
    'opt': {
        'epochs': 200,
        'batch_size': 500,
        'lr': 0.2,
        'momentum': 0.9,
        'wd': 5e-4,
    },
    'aug': {
        'flip': True,
        'translate': 4,
        'cutout': 0,
    },
    'net': {
        'width': 1.0,
    },
}

########################################
#           Train and Eval             #
########################################

def evaluate(model, loader):
    model.eval()
    with torch.no_grad():
        outs = torch.cat([model(inputs) for inputs, _ in loader])
    return (outs.argmax(1) == loader.labels).float().mean().item()

def trainval_split(loader, frac=0.02):
    train_loader = CifarLoader('cifar10', train=True, aug=loader.aug,
                               shuffle=loader.shuffle, drop_last=loader.drop_last)
    val_loader = CifarLoader('cifar10', train=False)
    n = len(loader.images)
    mask = (torch.rand(n) < frac)
    #if n <= 50000:
    #    mask = (torch.rand(n) < frac)
    #else:
    #    assert n % 50000 == 0
    #    mask = (torch.rand(50000) < frac).repeat(n//50000)
    train_loader.images, val_loader.images = loader.images[~mask], loader.images[mask]
    train_loader.labels, val_loader.labels = loader.labels[~mask], loader.labels[mask]
    return train_loader, val_loader

def train(train_loader, test_loader=None,
          epochs=hyp['opt']['epochs'], lr=hyp['opt']['lr'],
          val_split=False):

    if val_split:
        train_loader, val_loader = trainval_split(train_loader)

    if test_loader is None:
        test_loader = CifarLoader('cifar10', train=False, batch_size=1000)
    batch_size = train_loader.batch_size

    momentum = hyp['opt']['momentum']
    wd = hyp['opt']['wd']

    total_train_steps = int(epochs * len(train_loader))
    lr_schedule = np.interp(np.arange(1+total_train_steps),
                            [0, int(0.2 * total_train_steps), total_train_steps],
                            [0.2, 1, 0]) # Triangular learning rate schedule

    model = make_net(w=hyp['net']['width'], num_classes=max(train_loader.labels)+1)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr/batch_size, momentum=momentum, nesterov=True,
                                weight_decay=wd*batch_size)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule.__getitem__)

    train_loss, train_acc = [torch.nan], [torch.nan]
    val_acc = [evaluate(model, val_loader)] if val_split else [torch.nan]
    test_acc = [evaluate(model, test_loader)]

    it = tqdm(range(math.ceil(epochs)))
    step = 0
    for epoch in it:
        if step >= total_train_steps:
            break

        model.train()
        for inputs, labels in train_loader:
            if step >= total_train_steps:
                break
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, labels, reduction='none')
            train_loss.append(loss.mean().item())
            train_acc.append((outputs.detach().argmax(1) == labels).float().mean().item())
            optimizer.zero_grad(set_to_none=True)
            loss.sum().backward()
            optimizer.step()
            scheduler.step()
            it.set_description('Acc=%.4f(train),%.4f(val),%.4f(test)' % (train_acc[-1], val_acc[-1], test_acc[-1]))
            step += 1

        if val_split:
            val_acc.append(evaluate(model, val_loader))
        test_acc.append(evaluate(model, test_loader))
        it.set_description('Acc=%.4f(train),%.4f(val),%.4f(test)' % (train_acc[-1], val_acc[-1], test_acc[-1]))

    log = dict(train_loss=train_loss, train_acc=train_acc, val_acc=val_acc, test_acc=test_acc)
    return model, log 

