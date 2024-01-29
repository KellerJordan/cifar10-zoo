# This script generates both D_other and a synthetic shortcut variant of it in the same manner as the one
# generated for D_det.
# Sample output:
"""
"""

import torch
from torch import nn

from adversarial import gen_adv_dataset
from loader import CifarLoader
from train import train, evaluate

if __name__ == '__main__':

    train_loader = CifarLoader('cifar10', train=True, aug=dict(flip=True, translate=4))
    train_loader.save('datasets/clean_train.pt')
    test_loader = CifarLoader('cifar10', train=False)
    num_classes = 10
    adv_radius = 0.5

    print('Training clean model...')
    train_loader.load('datasets/clean_train.pt')
    train(train_loader)

    print('Generating D_other...')
    loader = gen_adv_dataset(model, dtype='dother', r=adv_radius, step_size=0.5)
    loader.save('datasets/basic_dother2.pt')
    print('Training on D_other...')
    train_loader.load('datasets/basic_dother2.pt')
    train(train_loader)

