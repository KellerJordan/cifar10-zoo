"""
## "bad + bad = good"
Constructs balanced halves A and B such that A yields 66.0% and B yields 69.4% accuracy.
"""
import torch
from loader import CifarLoader
from train import train, evaluate
from utils import convert_binary, rand_mask_like, repeat_augs, get_margins
import functools
convert_binary = functools.partial(convert_binary, classes=(3, 5)) # cat/dog

test_loader = convert_binary(CifarLoader('cifar10', train=False))
train_aug = dict(flip=True, translate=4)

print('Training weak classifier to use for splitting...')
loader = convert_binary(CifarLoader('cifar10', train=True, aug=train_aug, drop_last=False))
train(loader, test_loader, epochs=0.5)

loader = convert_binary(CifarLoader('cifar10', train=True, aug=train_aug))
n_aug = 20
loader = repeat_augs(loader, n_epochs=n_aug)
margins = get_margins(model, loader).float()
q = 0.5
q0 = margins[loader.labels == 0].float().quantile(q)
q1 = margins[loader.labels == 1].float().quantile(q)
mask = ((loader.labels == 0) & (margins < q0)) | ((loader.labels == 1) & (margins < q1))
print('Margin 10th percentile=%.2f' % q, mask.float().mean(), mask.sum())

print('Baseline: training on random half of CIFAR-2...')
train_loader = convert_binary(CifarLoader('cifar10', train=True, aug=train_aug))
rand_mask = (torch.rand(len(loader.labels)) < 0.5)
train_loader.images = loader.images[rand_mask]
train_loader.labels = loader.labels[rand_mask]
train(train_loader, test_loader, epochs=200//n_aug) # 200 effective epochs

print('Training on subset A (%d examples)...' % (~mask).sum())
train_loader = convert_binary(CifarLoader('cifar10', train=True, aug=train_aug))
train_loader.images = loader.images[~mask]
train_loader.labels = loader.labels[~mask]
train(train_loader, test_loader, epochs=200//n_aug) # 200 effective epochs

print('Training on subset B (%d examples)...' % mask.sum())
train_loader = convert_binary(CifarLoader('cifar10', train=True, aug=train_aug))
train_loader.images = loader.images[mask]
train_loader.labels = loader.labels[mask]
train(train_loader, test_loader, epochs=200//n_aug) # 200 effective epochs

