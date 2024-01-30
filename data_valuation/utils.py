from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F

def get_margins(model, loader):
    shuffle = loader.shuffle
    loader.shuffle = False
    with torch.no_grad():
        margins = []
        for inputs, labels in tqdm(loader):
            output = (model(inputs) + model(inputs.flip(-1)))[:, :2]
            mask = F.one_hot(labels, num_classes=2).bool()
            margin = (output[mask] - output[~mask]).flatten()
            margins.append(margin)
        margins = torch.cat(margins)
    loader.shuffle = shuffle
    return margins

# Input: a loader with data augmentation
# Output: the loader with a fixed n_epochs of augmented data that will be repeated
def repeat_augs(loader, n_epochs):
    aug_inputs = []
    aug_labels = []
    for _ in range(n_epochs):
        for inputs, labels in loader:
            aug_inputs.append(loader.denormalize(inputs))
            aug_labels.append(labels)
    loader.images = torch.cat(aug_inputs)
    loader.labels = torch.cat(aug_labels)
    loader.aug = {}
    loader.shuffle = False
    return loader

def viz(log):
    steps = list(range(len(log['train_acc'])))
    plt.plot(steps, log['train_acc'], label='train')
    per_epoch = (len(log['train_acc'])-1) // (len(log['test_acc'])-1)
    plt.plot(steps[::per_epoch][:len(log['test_acc'])], log['test_acc'], label='test')
    if 'val_acc' in log.keys():
        plt.plot(steps[::per_epoch][:len(log['val_acc'])], log['val_acc'], label='val')
    plt.legend()
    plt.show()
    
def rand_mask_like(tensor, p):
    xx = torch.rand_like(tensor.float())
    q = xx.quantile(p)
    return (xx < q)
    
# "cat" is CIFAR-10 class 3, and "dog" is class 5
# classes = (3, 5)
# classes = (2, 6) # bird / frog -> 97.0
classes = (0, 6) # airplane / frog -> 98.5
def convert_binary(loader, classes=classes):
    labels = loader.labels
    c0, c1 = classes
    mask = (labels == c0) | (labels == c1)
    loader.images = loader.images[mask]
    loader.labels = loader.labels[mask]
    loader.labels = (loader.labels == c1).long()
    return loader

