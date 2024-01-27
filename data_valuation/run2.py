# Sample output:
"""
Training on full cat/dog set...
Acc=1.0000(train),nan(val),0.9060(test): 100%|██████████| 200/200 [00:45<00:00,  4.43it/s]
Training weak classifier to use for splitting...
Acc=0.6600(train),nan(val),0.6155(test): 100%|██████████████| 1/1 [00:00<00:00,  4.64it/s]
Class balance: 0.5006
Constructing subset A of correctly predicted examples...
Training on set A (4964 examples)...
Acc=1.0000(train),nan(val),0.6540(test): 100%|██████████| 200/200 [00:20<00:00,  9.56it/s]
Constructing subset B of incorrectly predicted examples...
Training on set B (5036 examples)...
Acc=1.0000(train),nan(val),0.6900(test): 100%|██████████| 200/200 [00:23<00:00,  8.69it/s]
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
    model, _ = train(train_loader, test_loader, epochs=1)
    loader = convert_catdog(CifarLoader('cifar10', train=True))
    margins = get_margins(model, loader)
    labels = loader.labels
    q0 = margins[labels == 0].float().quantile(0.5)
    q1 = margins[labels == 1].float().quantile(0.5)
    mask = ((labels == 0) & (margins < q0)) | ((labels == 1) & (margins < q1))
    #mask = (torch.rand_like(margins) < 0.5)
    print('Class balance: %.4f' % loader.labels[mask].float().mean())
    
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

