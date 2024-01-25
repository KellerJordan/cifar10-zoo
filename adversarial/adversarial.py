from tqdm import tqdm
import torch
import torch.nn.functional as F
from loader import CifarLoader

loader0 = CifarLoader('cifar10', train=False, batch_size=500, shuffle=False, drop_last=False)
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
        delta.data = loader0.normalize(loader0.denormalize(inputs + delta.data).clip(0, 1)) - inputs

    return delta.data

## Generates D_rand, D_det, or D_other from the CIFAR-10 training set for a given model
def gen_adv_dataset(model, dtype='dother', **pgd_kwargs):
    assert dtype in ['drand', 'ddet', 'dother']
    loader = CifarLoader('cifar10', train=True, batch_size=500, shuffle=False, drop_last=False)
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

