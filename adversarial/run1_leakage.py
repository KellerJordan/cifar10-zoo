"""
Sample output:
Training clean model to select subset to shortcut-away...
Acc=0.8680(train),nan(val),0.8465(test):  75%|████          | 3/4 [00:06<00:02,  2.23s/it]
Generating leakage-only D_rand...
Using delta=0 for n=5013 examples
Using synthetic delta for n=44987 examples
Training on leakage-only D_rand...
Acc=1.0000(train),nan(val),0.7831(test): 100%|█████████▍| 199/200 [03:54<00:01,  1.18s/it]
Generating leakage-only D_det...
Using delta=0 for n=510 examples
Using synthetic delta for n=49490 examples
Training on leakage-only D_det...
Acc=1.0000(train),nan(val),0.2541(test): 100%|█████████▍| 199/200 [03:39<00:01,  1.10s/it]
Generating leakage-only D_other...
Using delta=0 for n=648 examples
Using synthetic delta for n=49352 examples
Training on leakage-only D_other...
Acc=1.0000(train),nan(val),0.3591(test): 100%|█████████▍| 199/200 [03:41<00:01,  1.12s/it]
"""

import torch
from torch import nn

from loader import CifarLoader
from train import train, evaluate

if __name__ == '__main__':

    num_classes = 10
    train_loader = CifarLoader('cifar10', train=True, aug=dict(flip=True, translate=4))
    train_loader.save('datasets/clean_train.pt')
    test_loader = CifarLoader('cifar10', train=False)
    adv_radius = 0.5

    # Generate 10 fixed synthetic perturbations via deconvolution
    # - This was found to be the best shortcut in practice, much better than using Gaussian noise
    deconv = nn.ConvTranspose2d(1, 30, 3, stride=2, padding=0, bias=False)
    with torch.no_grad():
        noise = deconv(torch.ones(1, 16, 16))[:, 1:, 1:].reshape(10, 3, 32, 32).cuda().half()
    unit_noise = noise / noise.reshape(len(noise), -1).norm(dim=1)[:, None, None, None]
    synthetic_noise = adv_radius * unit_noise

    print('Training clean model to select subset to shortcut-away...')
    train_loader.load('datasets/clean_train.pt')
    model, _ = train(train_loader, epochs=4, lr=0.5)

    lr = 0.05

    # Leakage-only D_rand
    print('Generating leakage-only D_rand...')
    loader = CifarLoader('cifar10', train=True)
    loader.labels = torch.randint(num_classes, size=(len(loader.labels),), device=loader.labels.device)
    with torch.no_grad():
        outputs = torch.cat([model(inputs) for inputs in loader.normalize(loader.images).split(500)])
        mask = (outputs.argmax(1) == loader.labels)
    print('Using delta=0 for n=%d examples' % mask.sum())
    print('Using synthetic delta for n=%d examples' % (~mask).sum())
    loader.images[~mask] = (loader.images[~mask] + synthetic_noise[loader.labels[~mask]]).clip(0, 1)
    loader.save('datasets/leak_drand.pt')
    print('Training on leakage-only D_rand...')
    train_loader.load('datasets/leak_drand.pt')
    train(train_loader, lr=lr)

    # Leakage-only D_det
    print('Generating leakage-only D_det...')
    loader = CifarLoader('cifar10', train=True)
    loader.labels = (loader.labels + 1) % num_classes
    with torch.no_grad():
        outputs = torch.cat([model(inputs) for inputs in loader.normalize(loader.images).split(500)])
        mask = (outputs.argmax(1) == loader.labels)
    print('Using delta=0 for n=%d examples' % mask.sum())
    print('Using synthetic delta for n=%d examples' % (~mask).sum())
    loader.images[~mask] = (loader.images[~mask] + synthetic_noise[loader.labels[~mask]]).clip(0, 1)
    loader.save('datasets/leak_ddet.pt')
    print('Training on leakage-only D_det...')
    train_loader.load('datasets/leak_ddet.pt')
    train(train_loader, lr=lr)

    # Leakage-only D_other
    print('Generating leakage-only D_other...')
    loader = CifarLoader('cifar10', train=True)
    loader.labels = (loader.labels + torch.randint(1, num_classes, size=(len(loader.labels),), device=loader.labels.device)) % num_classes
    with torch.no_grad():
        outputs = torch.cat([model(inputs) for inputs in loader.normalize(loader.images).split(500)])
        mask = (outputs.argmax(1) == loader.labels)
    print('Using delta=0 for n=%d examples' % mask.sum())
    print('Using synthetic delta for n=%d examples' % (~mask).sum())
    loader.images[~mask] = (loader.images[~mask] + synthetic_noise[loader.labels[~mask]]).clip(0, 1)
    loader.save('datasets/leak_dother.pt')
    print('Training on leakage-only D_other...')
    train_loader.load('datasets/leak_dother.pt')
    train(train_loader, lr=lr)

