# Sample output:
"""
Training on full cat/dog set...
Acc=1.0000(train),nan(val),0.9045(test): 100%|██████████| 200/200 [00:45<00:00,  4.39it/s]
Training weak classifier to use for splitting...
Acc=0.8920(train),nan(val),0.8440(test): 100%|██████████████| 8/8 [00:03<00:00,  2.01it/s]
Constructing subset A of correctly predicted examples...
Training on set A (8978 examples)...
Acc=1.0000(train),nan(val),0.8830(test): 100%|██████████| 200/200 [00:37<00:00,  5.37it/s]
Constructing subset B of incorrectly predicted examples...
Training on set B (1022 examples)...
Acc=1.0000(train),nan(val),0.2985(test): 100%|██████████| 200/200 [00:06<00:00, 30.54it/s]
"""

import torch
from torch import nn
import torch.nn.functional as F

from loader import CifarLoader
from train import train, evaluate

def get_margins(model, loader):
    shuffle = loader.shuffle
    loader.shuffle = False
    with torch.no_grad():
        margins = []
        for inputs, labels in loader:
            output = (model(inputs) + model(inputs.flip(-1)))[:, :2]
            mask = F.one_hot(labels, num_classes=2).bool()
            margin = (output[mask] - output[~mask]).flatten()
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
    train(train_loader, test_loader)

    print('Training weak classifier to use for splitting...')
    model, _ = train(train_loader, test_loader, epochs=8)
    loader = convert_catdog(CifarLoader('cifar10', train=True))
    margins = get_margins(model, loader)
    q = margins.float().quantile(0.101)
    mask = (margins > q)
    
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

