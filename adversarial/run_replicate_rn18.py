# This script replicates the original D_rand and D_det experiments from Ilyas et al. (2019)
# It gets much better accuracy on D_rand (likely because we do not force all perturbations to
# reach the maximum radius), and worse on D_det (caused by using the smaller architecture).
# The following is with learning rate 0.2
"""
Training clean model...
Acc=0.9512(test),1.0000(train): 100%|███████████████████| 200/200 [07:21<00:00,  2.21s/it]
Clean test accuracy: 0.9512
Generating D_rand...
100%|███████████████████████████████████████████████████| 100/100 [03:46<00:00,  2.26s/it]
Fooling rate: 0.9811
Training on D_rand...
Acc=0.1036(test),0.1380(train): 100%|███████████████████| 200/200 [07:19<00:00,  2.20s/it]
Clean test accuracy: 0.1035
Generating D_det...
100%|███████████████████████████████████████████████████| 100/100 [03:45<00:00,  2.26s/it]
Fooling rate: 0.9811
Training on D_det...
Acc=0.3515(test),1.0000(train): 100%|███████████████████| 200/200 [07:30<00:00,  2.25s/it]
Clean test accuracy: 0.3514
"""
# The following is with learning rate 0.1
"""
Training clean model...
Acc=0.9489(test),1.0000(train): 100%|███████████████████| 200/200 [07:37<00:00,  2.29s/it]
Clean test accuracy: 0.9489
Generating D_rand...
100%|███████████████████████████████████████████████████| 100/100 [03:45<00:00,  2.26s/it]
Fooling rate: 0.9646
Training on D_rand...
Acc=0.0988(test),0.9980(train): 100%|███████████████████| 200/200 [07:18<00:00,  2.19s/it]
Clean test accuracy: 0.0986
Generating D_det...
100%|███████████████████████████████████████████████████| 100/100 [03:45<00:00,  2.26s/it]
Fooling rate: 0.9664
Training on D_det...
Acc=0.2513(test),1.0000(train): 100%|███████████████████| 200/200 [07:17<00:00,  2.19s/it]
Clean test accuracy: 0.2513
"""
# The following is with learning rate 0.05
"""
Training clean model...
Acc=1.0000(train),0.9433(test): 100%|███████████████████| 200/200 [07:21<00:00,  2.21s/it]
Clean test accuracy: 0.9433
Generating D_rand...
100%|███████████████████████████████████████████████████| 100/100 [03:45<00:00,  2.26s/it]
Fooling rate: 0.9901
Training on D_rand...
Acc=0.9980(train),0.1117(test): 100%|███████████████████| 200/200 [07:17<00:00,  2.19s/it]
Clean test accuracy: 0.1118
Generating D_det...
100%|███████████████████████████████████████████████████| 100/100 [03:45<00:00,  2.26s/it]
Fooling rate: 0.9881
Training on D_det...
Acc=1.0000(train),0.3465(test): 100%|███████████████████| 200/200 [08:50<00:00,  2.65s/it]
Clean test accuracy: 0.3465
"""


import os
from tqdm import tqdm
import torch
import torch.nn.functional as F

from loader import CifarLoader
#from train import train, evaluate
from train_rn18 import train, evaluate

loader = CifarLoader('cifar10', train=True, batch_size=500, shuffle=False, drop_last=False)

def pgd(inputs, targets, model, r=0.5, step_size=0.1, steps=100, eps=1e-5):
    delta = torch.zeros_like(inputs, requires_grad=True)
    norm_r = 4 * r # radius converted into normalized pixel space
    norm_step_size = 4 * step_size
    
    for step in range(steps):
        
        delta.grad = None
        output = model(inputs + delta)
        loss = F.cross_entropy(output, targets, reduction='none').sum()
        loss.backward()

        # normalize gradient
        grad_norm = delta.grad.reshape(len(delta), -1).norm(dim=1)
        unit_grad = delta.grad / (grad_norm[:, None, None, None] + eps)
        
        # take step in unit-gradient direction with scheduled step size
        delta.data -= norm_step_size * unit_grad

        # project to r-sphere
        delta_norm = delta.data.reshape(len(delta), -1).norm(dim=1)
        mask = (delta_norm >= norm_r)
        delta.data[mask] = norm_r * delta.data[mask] / (delta_norm[mask, None, None, None] + eps)
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

    os.makedirs('datasets', exist_ok=True)

    train_loader = CifarLoader('cifar10', train=True, aug=dict(flip=True, translate=4))
    test_loader = CifarLoader('cifar10', train=False)

    print('Training clean model...')
    model, _ = train(train_loader)
    print('Clean test accuracy: %.4f' % evaluate(model, test_loader))

    print('Generating D_rand...')
    loader = gen_adv_dataset(model, dtype='drand', r=0.5, step_size=0.1)
    loader.save('datasets/replicate_drand.pt')
    train_loader.load('datasets/replicate_drand.pt')
    print('Training on D_rand...')
    model1, _ = train(train_loader)
    print('Clean test accuracy: %.4f' % evaluate(model1, test_loader))

    print('Generating D_det...')
    loader = gen_adv_dataset(model, dtype='ddet', r=0.5, step_size=0.1)
    loader.save('datasets/replicate_ddet.pt')
    train_loader.load('datasets/replicate_ddet.pt')
    print('Training on D_det...')
    model1, _ = train(train_loader)
    print('Clean test accuracy: %.4f' % evaluate(model1, test_loader))

