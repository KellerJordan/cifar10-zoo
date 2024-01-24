import torch
from torch import nn

from loader import CifarLoader
from model import make_net
from train import train, evaluate

if __name__ == '__main__':

    num_classes = 10
    test_loader = CifarLoader('cifar10', train=False)

    # Generate 10 fixed synthetic perturbations
    deconv = nn.ConvTranspose2d(1, 30, 3, stride=2, padding=0, bias=False)
    deconv.weight.requires_grad = False
    res = deconv(torch.ones(1, 16, 16))[:, 1:, 1:]
    synthetic_noise = res.reshape(10, 3, 32, 32).cuda().half()

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
    print('Clean test accuracy: %.4f' % evaluate(model, test_loader))
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

