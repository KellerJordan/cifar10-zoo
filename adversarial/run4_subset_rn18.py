# This script trains on various subsets of D_other
# Sample output:
"""
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
    test_loader = CifarLoader('cifar10', train=False)
    num_classes = 10
    adv_radius = 0.5

    print('Training clean model...')
    model, _ = train(train_loader)

    print('Generating D_other...')
    loader = gen_adv_dataset(model, dtype='dother', r=adv_radius, step_size=0.1)
    loader.save('datasets/basic_dother.pt')
    train_loader.load('datasets/basic_dother.pt')
    print('Training on D_other...')
    model1, _ = train(train_loader)

    # Construct various subsets of D_other for training / eval
    loader = CifarLoader('cifar10', shuffle=False)
    loader.load('datasets/basic_dother.pt')
    margins = get_margins(model, loader)

    print('Training on top 40% most fooling examples...')
    mask = (margins >= margins.float().quantile(0.6))
    print('Contains %d examples' % mask.sum())
    train_loader.images = loader.images[mask]
    train_loader.labels = loader.labels[mask]
    model1, _ = train(train_loader)

    print('Training on bottom 60% most fooling examples...')
    mask = (margins < margins.float().quantile(0.6))
    print('Contains %d examples' % mask.sum())
    train_loader.images = loader.images[mask]
    train_loader.labels = loader.labels[mask]
    model1, _ = train(train_loader)

    print('Training on bottom 60% most fooling examples, with perturbation scaled up by 2x...')
    mult_r = 2.0
    clean_images = CifarLoader('cifar10', train=True).images[mask]
    adv_images = train_loader.images
    scaled_adv_images = (mult_r * (adv_images - clean_images) + clean_images).clip(0, 1)
    train_loader.images = scaled_adv_images
    model1, _ = train(train_loader)
 
