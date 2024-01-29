# This script trains on various subsets of D_other
# Sample output:
"""
Training clean model...
Acc=1.0000(train),0.9398(test): 100%|███████████████████| 200/200 [03:35<00:00,  1.08s/it]
Generating D_other...
100%|███████████████████████████████████████████████████| 100/100 [01:51<00:00,  1.12s/it]
Fooling rate: 0.9304
Training on D_other...
Acc=1.0000(train),0.6603(test): 100%|███████████████████| 200/200 [03:33<00:00,  1.07s/it]
Training on top 40% most fooling examples...
Contains 20071 examples
Acc=1.0000(train),0.7818(test): 100%|███████████████████| 200/200 [01:30<00:00,  2.21it/s]
Training on bottom 60% most fooling examples...
Contains 29929 examples
Acc=1.0000(train),0.0296(test): 100%|███████████████████| 200/200 [02:11<00:00,  1.53it/s]
"""

import torch
from torch import nn
import torch.nn.functional as F

from loader import CifarLoader
from train import train, evaluate
from adversarial import gen_adv_dataset

def get_margins(model, loader):
    with torch.no_grad():
        margins = []
        for inputs, labels in loader:
            output = model(inputs)
            mask = F.one_hot(labels, num_classes=10).bool()
            margin = output[mask] - output[~mask].reshape(len(output), -1).amax(1)
            margins.append(margin)
        margins = torch.cat(margins)
    return margins

if __name__ == '__main__':

    train_loader = CifarLoader('cifar10', train=True, batch_size=500, aug=dict(flip=True, translate=4))
    train_loader.save('datasets/clean_train.pt')
    test_loader = CifarLoader('cifar10', train=False)
    num_classes = 10
    adv_radius = 0.5

    print('Training clean model...')
    model, _ = train(train_loader)

    print('Generating D_other...')
    loader = gen_adv_dataset(model, dtype='dother', r=adv_radius, step_size=0.1)
    loader.save('datasets/basic_dother.pt')
    train_loader.load('datasets/basic_dother.pt')
    print('Training on D_other...')
    model1, _ = train(train_loader)

    # Get the target-class logit margins for D_other in order to construct various subsets
    loader = CifarLoader('cifar10', shuffle=False)
    loader.load('datasets/basic_dother.pt')
    margins = get_margins(model, loader)

    print('Training on top 40% most fooling examples...')
    mask = (margins >= margins.float().quantile(0.6))
    print('Contains %d examples' % mask.sum())
    train_loader.images = loader.images[mask]
    train_loader.labels = loader.labels[mask]
    train_loader.save('datasets/basic_dother_top40.pt')
    model1, _ = train(train_loader)

    print('Generating leakage-only D_other-top40...')
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
    loader.load('datasets/basic_dother_top40.pt')
    with torch.no_grad():
        outputs = torch.cat([model(inputs) for inputs in loader.normalize(loader.images).split(500)])
        mask = (outputs.argmax(1) == loader.labels)
    print('Using delta=0 for n=%d examples' % mask.sum())
    print('Using synthetic delta for n=%d examples' % (~mask).sum())
    loader.images[~mask] = (loader.images[~mask] + synthetic_noise[loader.labels[~mask]]).clip(0, 1)
    loader.save('datasets/leak_dother_top40.pt')
    print('Training on leakage-only D_other_top40...')
    train_loader.load('datasets/leak_dother_top40.pt')
    model1, _ = train(train_loader)

    print('Training on bottom 60% most fooling examples...')
    mask = (margins < margins.float().quantile(0.6))
    print('Contains %d examples' % mask.sum())
    train_loader.images = loader.images[mask]
    train_loader.labels = loader.labels[mask]
    model1, _ = train(train_loader)

    print('Training on bottom 60% most fooling examples, with perturbation scaled up by 2x...')
    mult_r = 2.0
    clean_images = CifarLoader('cifar10', train=True).images[mask]
    adv_images = train_loader.images
    scaled_adv_images = (mult_r * (adv_images - clean_images) + clean_images).clip(0, 1)
    train_loader.images = scaled_adv_images
    model1, _ = train(train_loader)

