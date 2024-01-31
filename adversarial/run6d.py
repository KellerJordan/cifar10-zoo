# This script trains on various subsets of D_other
# Sample output:
"""
Training clean model...
Acc=1.0000(train),0.9398(test): 100%|███████████████████| 200/200 [03:35<00:00,  1.08s/it]
Generating D_other...
100%|███████████████████████████████████████████████████| 100/100 [01:51<00:00,  1.12s/it]
Fooling rate: 0.9304
Training on D_other...
Acc=1.0000(train),0.6603(test): 100%|███████████████████| 200/200 [03:33<00:00,  1.07s/it]
Training on top 40% most fooling examples...
Contains 20071 examples
Acc=1.0000(train),0.7818(test): 100%|███████████████████| 200/200 [01:30<00:00,  2.21it/s]
Generating leakage-only D_other-top40...
Sampling 10 fixed synthetic perturbations...
Training clean model to select shortcutted-away subset...
Acc=0.6040(train),nan(val),0.6433(test): 100%|██████████████| 1/1 [00:02<00:00,  2.58s/it]
Applying perturbations/deltas...
Using delta=0 for n=1461 examples
Using synthetic delta for n=18563 examples
Training on leakage-only D_other_top40...
Acc=1.0000(train),nan(val),0.4278(test): 100%|██████████| 200/200 [03:22<00:00,  1.01s/it]
Training on bottom 60% most fooling examples...
Contains 29929 examples
Acc=1.0000(train),0.0296(test): 100%|███████████████████| 200/200 [02:11<00:00,  1.53it/s]
"""

import torch
from torch import nn
import torch.nn.functional as F

from loader import CifarLoader
from train_rn18 import train, evaluate
from adversarial import gen_adv_dataset

def get_margins(model, loader):
    with torch.no_grad():
        margins = []
        for inputs, labels in loader:
            output = model(inputs)
            mask = F.one_hot(labels, num_classes=10).bool()
            margin = output[mask] - output[~mask].reshape(len(output), -1).amax(1)
            margins.append(margin)
        margins = torch.cat(margins)
    return margins

if __name__ == '__main__':

    train_loader = CifarLoader('cifar10', train=True, batch_size=500, aug=dict(flip=True, translate=4))
    train_loader.save('datasets/clean_train.pt')
    test_loader = CifarLoader('cifar10', train=False)
    num_classes = 10
    adv_radius = 0.5

    print('Training clean model...')
    model, _ = train(train_loader, lr=0.5)
    model1, _ = train(train_loader, lr=0.5)

    print('Generating D_other...')
    loader = gen_adv_dataset(model, dtype='dother', r=adv_radius, step_size=0.1)
    loader.save('datasets/basic_dother2.pt')
    train_loader.load('datasets/basic_dother2.pt')
    print('Training on D_other...')
    train(train_loader)

    # Get the target-class logit margins for D_other in order to construct various subsets
    loader = CifarLoader('cifar10', shuffle=False)
    loader.load('datasets/basic_dother2.pt')
    margins = get_margins(model1, loader)

    print('Training on top 40% most fooling examples...')
    pp = torch.arange(0.05, 1.0, 0.05)
    logs = []
    for p in pp.tolist():
        top_mask = (margins >= margins.float().quantile(p))
        print('p=%.2f, contains %d examples' % (p, top_mask.sum()))

        train_loader.images = loader.images[top_mask]
        train_loader.labels = loader.labels[top_mask]
        _, log = train(train_loader, val_split=True)
        log['setting'] = '%.2f' % p
        logs.append(log)

        train_loader.images = loader.images[~top_mask]
        train_loader.labels = loader.labels[~top_mask]
        _, log = train(train_loader, val_split=True)
        log['setting'] = '%.2f inv' % p
        logs.append(log)

    obj = dict(logs=logs, margins=margins)
    torch.save(obj, 'log_6d2.pt')

