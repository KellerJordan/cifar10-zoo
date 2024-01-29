# This script generates both D_other and a synthetic shortcut variant of it in the same manner as the one
# generated for D_det.
# Sample output:
"""
Training clean model...
Acc=1.0000(train),0.9432(test): 100%|███████████████████| 200/200 [03:35<00:00,  1.08s/it]
Generating D_other...
100%|███████████████████████████████████████████████████| 100/100 [01:51<00:00,  1.12s/it]
Fooling rate: 0.9313
Training on D_other...
Acc=1.0000(train),0.6728(test): 100%|███████████████████| 200/200 [03:34<00:00,  1.07s/it]
Generating leakage-only D_other...
Sampling 10 fixed synthetic perturbations...
Training clean model to select shortcutted-away subset...
Acc=0.6480(train),nan(val),0.6533(test): 100%|██████████████| 1/1 [00:03<00:00,  3.45s/it]
Applying perturbations/deltas...
Using delta=0 for n=1860 examples
Using synthetic delta for n=48140 examples
Training on leakage-only D_other...
Acc=1.0000(train),nan(val),0.5013(test): 100%|██████████| 200/200 [03:34<00:00,  1.07s/it]
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
    model, _ = train(train_loader)

    print('Generating D_other...')
    loader = gen_adv_dataset(model, dtype='dother', r=adv_radius, step_size=0.1)
    loader.save('datasets/basic_dother.pt')
    print('Training on D_other...')
    train_loader.load('datasets/basic_dother.pt')
    model1, _ = train(train_loader)

    print('Generating leakage-only D_other...')
    print('Sampling 10 fixed synthetic perturbations...')
    # Generate 10 fixed synthetic perturbations via deconvolution
    # - This was found to be the best shortcut in practice, much better than using Gaussian noise
    deconv = nn.ConvTranspose2d(1, 30, 3, stride=2, padding=0, bias=False)
    with torch.no_grad():
        noise = deconv(torch.ones(1, 16, 16))[:, 1:, 1:].reshape(10, 3, 32, 32).cuda().half()
    unit_noise = noise / noise.reshape(len(noise), -1).norm(dim=1)[:, None, None, None]
    synthetic_noise = adv_radius * unit_noise
    print('Training clean model to select shortcutted-away subset...')
    train_loader.load('datasets/clean_train.pt')
    model, _ = train(train_loader, epochs=1)
    print('Applying perturbations/deltas...')
    loader = CifarLoader('cifar10', train=True)
    labels_rotate = torch.randint(1, num_classes, size=(len(loader.labels),), device=loader.labels.device)
    loader.labels = (loader.labels + labels_rotate) % num_classes
    with torch.no_grad():
        outputs = torch.cat([model(inputs) for inputs in loader.normalize(loader.images).split(500)])
        mask = (outputs.argmax(1) == loader.labels)
    print('Using delta=0 for n=%d examples' % mask.sum())
    print('Using synthetic delta for n=%d examples' % (~mask).sum())
    loader.images[~mask] = (loader.images[~mask] + synthetic_noise[loader.labels[~mask]]).clip(0, 1)
    loader.save('datasets/leak_dother.pt')
    print('Training on leakage-only D_other...')
    train_loader.load('datasets/leak_dother.pt')
    model1, _ = train(train_loader)

