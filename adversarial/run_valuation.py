# This script trains on various subsets of D_other
# Sample output:
"""
Training clean model...
Acc=1.0000(train),0.9398(test): 100%|███████████████████| 200/200 [03:35<00:00,  1.08s/it]
Clean test accuracy: 0.9398
Generating D_other...
100%|███████████████████████████████████████████████████| 100/100 [01:51<00:00,  1.12s/it]
Fooling rate: 0.9304
Training on D_other...
Acc=1.0000(train),0.6603(test): 100%|███████████████████| 200/200 [03:33<00:00,  1.07s/it]
Clean test accuracy: 0.6603
Training on bottom 60% most fooling examples...
Contains 29929 examples
Acc=1.0000(train),0.0296(test): 100%|███████████████████| 200/200 [02:11<00:00,  1.53it/s]
Clean test accuracy: 0.0296
Training on top 40% most fooling examples...
Contains 19996 examples
Acc=1.0000(train),0.7818(test): 100%|███████████████████| 200/200 [01:30<00:00,  2.21it/s]
Clean test accuracy: 0.7818
"""

import torch
from torch import nn
import torch.nn.functional as F

from loader import CifarLoader
#from train_rn18 import train, evaluate
from train import train, evaluate

def get_margins(model, loader):
    shuffle = loader.shuffle
    loader.shuffle = False
    with torch.no_grad():
        margins = []
        for inputs, labels in loader:
            output = model(inputs)[:, :2]
            mask = F.one_hot(labels, num_classes=2).bool()
            margin = output[mask] - output[~mask].reshape(len(output), -1).amax(1)
            margins.append(margin)
        margins = torch.cat(margins)
    loader.shuffle = shuffle
    return margins

# "cat" is CIFAR-10 class 3, and "dog" is class 5
def convert_catdog(loader):
    labels = loader.labels
    mask = (labels == 3) | (labels == 5)
    loader.images = loader.images[mask]
    loader.labels = loader.labels[mask]
    loader.labels = (loader.labels == 5).long()
    return loader

if __name__ == '__main__':

    train_loader = convert_catdog(CifarLoader('cifar10', train=True, aug=dict(flip=True, translate=4)))
    test_loader = convert_catdog(CifarLoader('cifar10', train=False))

    print('Training on full cat/dog set...')
    train(train_loader, test_loader, epochs=100)

    print('Training for two epochs to get network to use for splitting...')
    model, _ = train(train_loader, test_loader, epochs=8)
    loader = convert_catdog(CifarLoader('cifar10', train=True))
    margins = get_margins(model, loader)
    q = margins.float().quantile(0.15)
    mask = (margins < q)
    
    print('Constructing subset A of correctly predicted examples...')
    loader = convert_catdog(CifarLoader('cifar10', train=True, aug=dict(flip=True, translate=4)))
    loader.images = loader.images[mask] 
    loader.labels = loader.labels[mask]
    print('Training on set A (%d examples)...' % mask.sum())
    train(loader, test_loader)

    print('Constructing subset B of incorrectly predicted examples...')
    loader = convert_catdog(CifarLoader('cifar10', train=True, aug=dict(flip=True, translate=4)))
    loader.images = loader.images[~mask] 
    loader.labels = loader.labels[~mask]
    print('Training on set B (%d examples)...' % (~mask).sum())
    train(loader, test_loader)

