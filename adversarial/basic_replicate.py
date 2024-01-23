import os
from tqdm import tqdm
import torch

from loader import CifarLoader
from model import make_net
from train import train, evaluate

loader = CifarLoader('cifar10', train=True, batch_size=500, shuffle=False, drop_last=False)

def pgd(inputs, targets, model, r=2.0, step_size=0.1, steps=100, eps=1e-5):
    delta = torch.zeros_like(inputs, requires_grad=True)
    for step in (range(steps)):
        
        delta.grad = None
        output = model(inputs + delta)
        loss = F.cross_entropy(output, targets, reduction='none').sum()
        loss.backward()

        # normalize gradient
        grad_norm = delta.grad.reshape(len(delta), -1).norm(dim=1)
        unit_grad = delta.grad / (grad_norm[:, None, None, None] + eps)
        
        # take step in unit-gradient direction with scheduled step size
        delta.data -= step_size * unit_grad

        # project to r-sphere
        delta_norm = delta.data.reshape(len(delta), -1).norm(dim=1)
        mask = (delta_norm >= r)
        delta.data[mask] = r * delta.data[mask] / (delta_norm[mask, None, None, None] + eps)
        # project to pixel-space
        delta.data = loader.normalize(loader.denormalize(inputs + delta.data).clip(0, 1)) - inputs

    return delta.data

## Generates D_rand, D_det, or D_other from the CIFAR-10 training set for a given model
def gen_adv_dataset(model, dtype='dother', **pgd_kwargs):
    assert dtype in ['drand', 'ddet', 'dother']
    loader = CifarLoader('/tmp/cifar10', train=True, batch_size=500, shuffle=False, drop_last=False)
    labels = loader.labels
    num_classes = 10
    if dtype == 'drand':
        loader.labels = torch.randint(num_classes, size=(len(labels),), device=labels.device)
    elif dtype == 'ddet':
        loader.labels = (labels + 1) % num_classes
    elif dtype == 'dother':
        labels_rotate = torch.randint(1, num_classes, size=(len(labels),), device=labels.device)
        loader.labels = (labels + labels_rotate) % num_classes
        
    inputs_adv = []
    for inputs, labels in tqdm(loader):
        delta = pgd(inputs, labels, model, **pgd_kwargs)
        inputs_adv.append(inputs + delta)
    inputs_adv = torch.cat(inputs_adv)

    loader.images = loader.denormalize(inputs_adv)
    print('Fooling rate: %.4f' % evaluate(model, loader))
    return loader


if __name__ == '__main__':

    train_loader = CifarLoader('cifar10', train=True, aug=dict(flip=True, translate=4))
    test_loader = CifarLoader('cifar10', train=False)

    print('Training clean model...')
    model, log = train(train_loader)
    print('Clean test accuracy: %.4f' % evaluate(model, test_loader))

    print('Generating D_rand...')
    loader = gen_adv_dataset(model, dtype='drand', r=2.0, step_size=0.1)
    print('Training on D_rand...')
    model1 = train(loader)
    print('Clean test accuracy: %.4f' % evaluate(model1, test_loader))

    print('Generating D_det...')
    loader = gen_adv_dataset(model, dtype='ddet', r=2.0, step_size=0.1)
    print('Training on D_det...')
    model1 = train(loader)
    print('Clean test accuracy: %.4f' % evaluate(model1, test_loader))

