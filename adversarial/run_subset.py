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
from train import train, evaluate
from adversarial import gen_adv_dataset

def get_margins(model, loader):
    with torch.no_grad():
        margins = []
        for inputs, labels in loader:
            output = model(inputs)
            mask = F.one_hot(labels).bool()
            margin = output[mask] - output[~mask].reshape(len(output), -1).amax(1)
            margins.append(margin)
        margins = torch.cat(margins)
    return margins

if __name__ == '__main__':

    train_loader = CifarLoader('cifar10', train=True, batch_size=500, aug=dict(flip=True, translate=4))
    test_loader = CifarLoader('cifar10', train=False)
    num_classes = 10
    adv_radius = 0.5

    print('Training clean model...')
    model, _ = train(train_loader)
    print('Clean test accuracy: %.4f' % evaluate(model, test_loader))

    print('Generating D_other...')
    loader = gen_adv_dataset(model, dtype='dother', r=adv_radius, step_size=0.1)
    loader.save('datasets/basic_dother.pt')
    train_loader.load('datasets/basic_dother.pt')
    print('Training on D_other...')
    model1, _ = train(train_loader)
    print('Clean test accuracy: %.4f' % evaluate(model1, test_loader))

    # Construct various subsets of D_other for training / eval
    loader = CifarLoader('cifar10', shuffle=False)
    loader.load('datasets/basic_dother.pt')
    margins = get_margins(model, loader)

    print('Training on bottom 60% most fooling examples...')
    mask = (margins < margins.float().quantile(0.6))
    print('Contains %d examples' % mask.sum())
    train_loader.images = loader.images[mask]
    train_loader.labels = loader.labels[mask]
    model1, _ = train(train_loader)
    print('Clean test accuracy: %.4f' % evaluate(model1, test_loader))
        
    print('Training on top 40% most fooling examples...')
    mask = (margins > margins.float().quantile(0.6))
    print('Contains %d examples' % mask.sum())
    train_loader.images = loader.images[mask]
    train_loader.labels = loader.labels[mask]
    model1, _ = train(train_loader)
    print('Clean test accuracy: %.4f' % evaluate(model1, test_loader))

