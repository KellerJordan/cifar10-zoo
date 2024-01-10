#############################################
#            Setup/Hyperparameters          #
#############################################

import os
import sys
import uuid
import numpy as np
from tqdm import tqdm

import torch
from torch import nn
import torch.nn.functional as F
torch.backends.cudnn.benchmark = True

from loader import CifarLoader

hyp = {
    'opt': {
        'epochs': 32,
        'batch_size': 500,
        'lr': 0.5,
        'momentum': 0.9,
        'wd': 5e-4,
    },
    'aug': {
        'flip': True,
        'translate': 2,
        'cutout': 0,
    },
}

#############################################
#            Network Components             #
#############################################

def make_net():

    class Mul(nn.Module):
        def __init__(self, weight):
            super().__init__()
            self.weight = weight
        def forward(self, x):
            return x * self.weight

    class Flatten(nn.Module):
        def forward(self, x):
            return x.view(len(x), -1)

    class Residual(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, x):
            return x + self.module(x)

    def conv_bn(channels_in, channels_out, kernel_size=3, stride=1, padding=1):
        bn = nn.BatchNorm2d(channels_out)
        bn.weight.requires_grad = False
        return nn.Sequential(
                nn.Conv2d(channels_in, channels_out,
                          kernel_size=kernel_size, stride=stride, padding=padding,
                          bias=False),
                bn,
                nn.ReLU(inplace=True)
        )

    NUM_CLASSES = 10
    model = nn.Sequential(
        conv_bn(3, 64, kernel_size=3, stride=1, padding=1),
        conv_bn(64, 128, kernel_size=5, stride=2, padding=2),
        Residual(nn.Sequential(conv_bn(128, 128), conv_bn(128, 128))),
        conv_bn(128, 256, kernel_size=3, stride=1, padding=1),
        nn.MaxPool2d(2),
        Residual(nn.Sequential(conv_bn(256, 256), conv_bn(256, 256))),
        conv_bn(256, 128, kernel_size=3, stride=1, padding=0),
        nn.AdaptiveMaxPool2d((1, 1)),
        Flatten(),
        nn.Linear(int(w*128), NUM_CLASSES, bias=False),
        Mul(0.2)
    )
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
    model.eval()
    with torch.no_grad():
        outs = torch.cat([model(inputs) for inputs, _ in loader])
    return (outs.argmax(1) == loader.labels).float().mean().item()

def train(train_loader):

    test_loader = CifarLoader('/tmp/cifar10', train=False, batch_size=1000)
    batch_size = train_loader.batch_size

    epochs = hyp['opt']['epochs']
    lr = hyp['opt']['lr']
    momentum = hyp['opt']['momentum']
    wd = hyp['opt']['wd']

    total_train_steps = len(train_loader) * epochs
    lr_schedule = np.interp(np.arange(1+total_train_steps),
                            [0, int(0.2 * total_train_steps), total_train_steps],
                            [0.2, 1, 0]) # Linear warmup then annealing schedule

    model = make_net()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr/batch_size, momentum=momentum, nesterov=True,
                                weight_decay=wd*batch_size)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule.__getitem__)

    train_loss, train_acc, test_acc = [], [], []

    for epoch in tqdm(range(epochs)):

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

        test_acc.append(evaluate(model, test_loader))

    log = dict(train_loss=train_loss, train_acc=train_acc, test_acc=test_acc)
    return model, log 


if __name__ == '__main__':

    with open(sys.argv[0]) as f:
        code = f.read()

    train_augs = dict(flip=hyp['aug']['flip'], translate=hyp['aug']['translate'], cutout=hyp['aug']['cutout'])
    train_loader = CifarLoader('/tmp/cifar10', train=True, batch_size=hyp['opt']['batch_size'], aug=train_augs)

    model, log = train(train_loader)
    log['hyp'] = hyp
    log['code'] = code
    print('Final acc:', log['test_acc'][-1])

    log_dir = os.path.join('logs', str(uuid.uuid4()))
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, 'log.pt')
    print(os.path.abspath(log_path))
    torch.save(log, os.path.join(log_dir, 'log.pt'))
    torch.save(model.state_dict(), os.path.join(log_dir, 'model.pt'))

