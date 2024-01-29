# This script generates a variant of D_rand and D_det with the following properties:
# * Like the original D_rand (resp. D_det), all examples have their target labels sampled randomly
#   (resp. deterministically), and are perturbed up to an L^2 radius of at most 0.5.
# * But unlike the originals, the perturbations here do not arise as features, being strictly synthetic noise.
# * Nevertheless, we still see generalization to clean test data, because of the way we choose which examples
#   to perturb and which to leave alone.
# Sample output is as follows (lr=0.05).
"""
Generating leakage-only D_rand...
Using delta=0 for n=4971 examples
Using synthetic delta for n=45029 examples
Training on leakage-only D_rand...
Acc=1.0000(train),0.7575(test): 100%|███████████████████| 200/200 [07:20<00:00,  2.20s/it]
Generating leakage-only D_det...
Training clean model to select subset to shortcut-away...
Acc=0.5420(train),0.5724(test): 100%|███████████████████████| 1/1 [00:02<00:00,  2.19s/it]
Generating leakage-only D_det...
Training clean model to select subset to shortcut-away...
Acc=0.5540(train),0.5725(test): 100%|███████████████████████| 1/1 [00:04<00:00,  4.70s/it]
Using delta=0 for n=1847 examples
Using synthetic delta for n=48153 examples
Training on leakage-only D_det...
Acc=1.0000(train),0.3872(test): 100%|███████████████████| 200/200 [07:17<00:00,  2.19s/it]
"""

import torch
from torch import nn

from loader import CifarLoader
from train_rn18 import train, evaluate

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

    # Leakage-only D_rand
    print('Generating leakage-only D_rand...')
    loader = CifarLoader('cifar10', train=True)
    drand_targets = torch.randint(num_classes, size=(len(loader.labels),), device=loader.labels.device)
    mask = (loader.labels == drand_targets)
    loader.labels = drand_targets
    print('Using delta=0 for n=%d examples' % mask.sum())
    print('Using synthetic delta for n=%d examples' % (~mask).sum())
    loader.images[~mask] = (loader.images[~mask] + synthetic_noise[loader.labels[~mask]]).clip(0, 1)
    loader.save('datasets/leak_drand.pt')
    print('Training on leakage-only D_rand...')
    train_loader.load('datasets/leak_drand.pt')
    model1, _ = train(train_loader)

    # Leakage-only D_det
    print('Generating leakage-only D_det...')
    print('Training clean model to select subset to shortcut-away...')
    train_loader.load('datasets/clean_train.pt')
    model, _ = train(train_loader, epochs=2)

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
    model1, _ = train(train_loader)

