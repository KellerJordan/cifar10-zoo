"""
experiment3.py
Trains on 1/20 of experiment2.py; worse-than-nothing whereas 2 was better
"""
from loader import CifarLoader
from train import train, evaluate
from utils import convert_binary, rand_mask_like, repeat_augs, get_margins

test_loader = convert_binary(CifarLoader('cifar10', train=False))
train_aug = dict(flip=True, translate=4)

print('Training weak classifier to use for splitting...')
loader = convert_binary(CifarLoader('cifar10', train=True, aug=train_aug, drop_last=False))
model, _ = train(loader, test_loader, epochs=2, val_split=False)

loader = convert_binary(CifarLoader('cifar10', train=True, aug=train_aug))
loader = repeat_augs(loader, n_epochs=24)
margins = get_margins(model, loader).float()

q = margins.quantile(0.13)
print('margin q:', q)
r = 0.02
mask = (margins < q) & rand_mask_like(margins, r)
print(mask.float().mean(), mask.sum())
loader.images = loader.images[mask] 
loader.labels = loader.labels[mask]
train(loader, test_loader, epochs=20, val_split=False)

