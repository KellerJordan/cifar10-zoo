"""
Cat/dog subset yielding 23.5% accuracy.
Replicates Wallace (2019).
"""
from loader import CifarLoader
from train import train, evaluate
from utils import convert_binary, rand_mask_like, repeat_augs, get_margins
import functools
convert_binary = functools.partial(convert_binary, classes=(3, 5)) # cat/dog

test_loader = convert_binary(CifarLoader('cifar10', train=False))
train_aug = dict(flip=True, translate=4)

print('Training weak classifier to use for splitting...')
loader = convert_binary(CifarLoader('cifar10', train=True, aug=train_aug, drop_last=False))
train(loader, test_loader, epochs=8)

loader = convert_binary(CifarLoader('cifar10', train=True, aug=train_aug))
n_aug = 20
loader = repeat_augs(loader, n_epochs=n_aug)
margins = get_margins(model, loader).float()
q = margins.quantile(0.05)
mask = (margins < q)
print('margin 5th percentile=%.2f' % q, mask.float().mean(), mask.sum())
loader.images = loader.images[mask] 
loader.labels = loader.labels[mask]
train(loader, test_loader, epochs=200//n_aug) # 200 effective epochs

