# This script generates a variant of D_rand and D_det with the following properties:
# * Like the original D_rand and D_det, all examples have their target labels sampled either uniformly
#   or deterministically, and are perturbed up to an L^2 radius of at most 0.5.
# * But unlike the originals, the perturbations here do not arise as features, being strictly synthetic noise.
# * Nevertheless, we still see generalization to clean test data, because of the way we choose which examples
#   to perturb and which to leave alone.
# Sample output:
"""
Generating leakage-only D_rand...
Using delta=0 for n=4980 examples
Using synthetic delta for n=45020 examples
Training on leakage-only D_rand...
Acc=1.0000(train),0.7951(test): 100%|█████████| 200/200 [03:36<00:00,  1.08s/it]
Clean test accuracy: 0.7951
Generating leakage-only D_det...
Training clean model to select subset of D_det...
Acc=0.6000(train),0.6310(test): 100%|█████████████| 1/1 [00:01<00:00,  1.08s/it]
Using delta=0 for n=1685 examples
Using synthetic delta for n=48315 examples
Training on leakage-only D_det...
Acc=1.0000(train),0.3231(test): 100%|█████████| 200/200 [03:33<00:00,  1.07s/it]
Clean test accuracy: 0.3231
"""

import torch
from torch import nn

from loader import CifarLoader
from model import make_net
from train import train, evaluate

if __name__ == '__main__':

    num_classes = 10
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
    train_loader = CifarLoader('cifar10', train=True, aug=dict(flip=True, translate=4))
    labels = train_loader.labels
    drand_targets = torch.randint(num_classes, size=(len(labels),), device=labels.device)
    mask = (labels == drand_targets)
    print('Using delta=0 for n=%d examples' % mask.sum())
    print('Using synthetic delta for n=%d examples' % (~mask).sum())
    train_loader.images[~mask] = (train_loader.images[~mask] + synthetic_noise[drand_targets[~mask]]).clip(0, 1)
    train_loader.labels = drand_targets
    print('Training on leakage-only D_rand...')
    model1, _ = train(train_loader)
    print('Clean test accuracy: %.4f' % evaluate(model1, test_loader))

    # Leakage-only D_det
    print('Generating leakage-only D_det...')
    train_loader = CifarLoader('cifar10', train=True, aug=dict(flip=True, translate=4))
    print('Training clean model to select subset of D_det...')
    model, _ = train(train_loader, epochs=1)
    ddet_images = train_loader.images
    ddet_targets = (train_loader.labels + 1) % num_classes
    with torch.no_grad():
        outputs = torch.cat([model(inputs) for inputs in train_loader.normalize(ddet_images).split(500)])
        mask = (outputs.argmax(1) == ddet_targets)
    print('Using delta=0 for n=%d examples' % mask.sum())
    print('Using synthetic delta for n=%d examples' % (~mask).sum())
    train_loader.images[~mask] = (train_loader.images[~mask] + synthetic_noise[ddet_targets[~mask]]).clip(0, 1)
    train_loader.labels = ddet_targets
    print('Training on leakage-only D_det...')
    model1, _ = train(train_loader)
    print('Clean test accuracy: %.4f' % evaluate(model1, test_loader))

