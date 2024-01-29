# This script replicates the original D_rand and D_det experiments from Ilyas et al. (2019)
# It gets much better accuracy on D_rand (likely because we do not force all perturbations to
# reach the maximum radius), and worse on D_det (caused by using the smaller architecture).
"""
Training clean model...
Acc=1.0000(train),0.9365(test): 100%|█████████████████████████████| 200/200 [03:35<00:00,  1.08s/it]
Generating D_rand...
100%|█████████████████████████████████████████████████████████████| 100/100 [01:51<00:00,  1.11s/it]
Fooling rate: 0.9367
Training on D_rand...
Acc=1.0000(train),0.8390(test): 100%|█████████████████████████████| 200/200 [03:33<00:00,  1.07s/it]
Generating D_det...
100%|█████████████████████████████████████████████████████████████| 100/100 [01:51<00:00,  1.11s/it]
Fooling rate: 0.9271
Training on D_det...
Acc=1.0000(train),0.1981(test): 100%|█████████████████████████████| 200/200 [03:32<00:00,  1.06s/it]
"""

import os
from tqdm import tqdm
import torch
import torch.nn.functional as F

from loader import CifarLoader
from train import train, evaluate
from adversarial import gen_adv_dataset

if __name__ == '__main__':

    train_loader = CifarLoader('cifar10', train=True, batch_size=500, aug=dict(flip=True, translate=4))
    test_loader = CifarLoader('cifar10', train=False)

    print('Training clean model...')
    model, _ = train(train_loader)

    print('Generating D_rand...')
    loader = gen_adv_dataset(model, dtype='drand', r=0.5, step_size=0.1)
    loader.save('datasets/replicate_drand.pt')
    print('Training on D_rand...')
    train_loader.load('datasets/replicate_drand.pt')
    train(train_loader)

    print('Generating D_det...')
    loader = gen_adv_dataset(model, dtype='ddet', r=0.5, step_size=0.1)
    loader.save('datasets/replicate_ddet.pt')
    print('Training on D_det...')
    train_loader.load('datasets/replicate_ddet.pt')
    train(train_loader)

