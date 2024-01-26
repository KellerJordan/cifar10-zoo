# Sample output:
"""
Training on full cat/dog set...
Acc=1.0000(train),0.8955(test): 100%|███████████████████| 200/200 [03:15<00:00,  1.02it/s]
Training weak classifier to use for splitting...
Acc=0.8320(train),0.8010(test): 100%|███████████████████████| 8/8 [00:07<00:00,  1.09it/s]
Constructing subset A of correctly predicted examples...
Training on set A (8987 examples)...
Acc=1.0000(train),0.8625(test): 100%|███████████████████| 200/200 [02:41<00:00,  1.24it/s]
Constructing subset B of incorrectly predicted examples...
Training on set B (1013 examples)...
Acc=1.0000(train),0.2830(test): 100%|███████████████████| 200/200 [00:29<00:00,  6.67it/s]
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
    #train(train_loader, test_loader)

    print('Training weak classifier to use for splitting...')
    model, _ = train(train_loader, test_loader, epochs=1)
    loader = convert_catdog(CifarLoader('cifar10', train=True))
    margins = get_margins(model, loader)
    labels = loader.labels
    q0 = margins[labels == 0].float().quantile(0.5)
    q1 = margins[labels == 1].float().quantile(0.5)
    mask = ((labels == 0) & (margins < q0)) | ((labels == 1) & (margins < q1))
    #mask = (torch.rand_like(margins) < 0.5)
    print(loader.labels[mask].float().mean())
    
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

