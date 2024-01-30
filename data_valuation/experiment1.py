"""
experiment1.py
Constructs a subset of CIFAR-10 airplane / frog which yields 7% generalization.
"""
from loader import CifarLoader
from train import train, evaluate
from utils import convert_binary, rand_mask_like, repeat_augs, get_margins

test_loader = convert_binary(CifarLoader('cifar10', train=False))
train_aug = dict(flip=True, translate=4)

print('Training weak classifier to use for splitting...')
loader = convert_binary(CifarLoader('cifar10', train=True, aug=train_aug, drop_last=False))
model, log = train(loader, test_loader, epochs=2, val_split=False)
#viz(log)

loader = convert_binary(CifarLoader('cifar10', train=True, aug=train_aug))
loader = repeat_augs(loader, n_epochs=24)
margins = get_margins(model, loader).float()
loader.save('data.pt')

q = margins.quantile(0.05)
print('margin q:', q)
loader.load('data.pt')
mask = (margins < q)
print(mask.float().mean(), mask.sum())
loader.images = loader.images[mask] 
loader.labels = loader.labels[mask]
model, log = train(loader, test_loader, epochs=10, val_split=False)
#viz(log)

