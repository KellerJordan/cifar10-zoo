# This script consumes the output of main_saveoutputs.py
# It should be pointed to the output directory of that script, which by default is ./logs_saveoutputs/

import os
import sys
import glob
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torchvision
train_labels = torch.tensor(torchvision.datasets.CIFAR10('/tmp/cifar10', download=True, train=True).targets)
test_labels = torch.tensor(torchvision.datasets.CIFAR10('/tmp/cifar10', download=True, train=False).targets)

################################################################################################
#    Load outputs and do variance analysis via theory from https://arxiv.org/abs/2304.01910    #
################################################################################################
path = './logs_saveoutputs/'
if len(sys.argv) >= 2:
    path = sys.argv[1]
print('loading batches of outputs from %s...' % path)
pp = glob.glob(os.path.join(path, '*/*.pt'))
objs = [torch.load(p, map_location='cpu') for p in tqdm(pp)]
obj = {k: torch.cat([o[k] for o in objs]) for k in objs[0].keys() if 'logit' in k}
del objs
print('total number of trained model outputs:', len(obj['logits']))
print()
print('we will measure all accuracy values (means and stds) in %, i.e. 100.0 is perfect accuracy')
print()
xx = (obj['logits_tta'].argmax(-1) == test_labels).float().mean(1)
print('mean accuracy (using tta):', xx.mean() * 100)
print('mean accuracy (using tta, bottom of 95% ci):', (xx.mean() - 1.96 * xx.std() / len(xx)**0.5) * 100)
xx = (obj['logits'].argmax(-1) == test_labels).float().mean(1)
print('mean accuracy (without tta):', xx.mean() * 100)
print('mean accuracy (without tta, bottom of 95% ci):', (xx.mean() - 1.96 * xx.std() / len(xx)**0.5) * 100)
print()
print('from this point forward we always use the non-tta logits')
logits = obj['logits']
print()
# Variance analysis
pred = logits.argmax(-1)
correct = (pred == test_labels).float()
pp = correct.mean(0)
true_var = correct.sum(1).var() # this is variance of num-correct variable, so 10^10 times variance of accuracy
pred_var = (pp * (1 - pp)).sum()
print('empirical stddev:', true_var.sqrt() / 100)
print('examplewise independence predicted stddev:', pred_var.sqrt() / 100)
n = logits.shape[1]
print('distribution-wise stddev (true instability):', ((n / (n-1)) * (true_var - pred_var)).sqrt() / 100)
print('(see theorem 2 of https://arxiv.org/abs/2304.01910)')
print('true instability should be very small for long stable trainings')
print()
ens_pred = logits.mean(0).argmax(1)
ens_acc = (ens_pred == test_labels).sum().item() / 100
print('*** ensemble accuracy: %.2f ***  (via standard logit-averaging)' % ens_acc)
print('\n')

################################################################################################
#                  Compute logit correlation kernel and measure k-NN accuracy                  #
################################################################################################

def normalize(logits):
    logits = logits.float()
    logits = logits.log_softmax(-1)
    logits = (logits - logits.mean(0, keepdim=True)) / logits.std(0, keepdim=True)
    return logits

# Computes correlations between log-softmax outputs
def get_kernel(logits_train, logits_test):
    assert len(logits_train) == len(logits_test)
    logits_train = normalize(logits_train)
    logits_test = normalize(logits_test)
    num_models = len(logits_train)
    num_train_examples = logits_train.shape[1]
    num_test_examples = logits_test.shape[1]
    kernel = torch.zeros(num_test_examples, num_train_examples).cuda()
    for c in tqdm(range(10)):
        # Load to GPU right before the dot product to save memory
        logits_train_cls = logits_train[..., c].cuda()
        logits_test_cls = logits_test[..., c].cuda()
        corr_cls = (logits_test_cls.T @ logits_train_cls) / num_models
        kernel += corr_cls
    return kernel

def predict_knn(kernel, train_labels, k_range):
    nbrs = kernel.topk(k=max(k_range), dim=1)
    nbr_labels = train_labels[nbrs.indices.cpu()]
    preds = [nbr_labels[:, :k].mode(1).values
             for k in k_range]
    return preds

M = len(obj['logits'])
print('computing correlation kernel of shape (50000, 10000)...')
kernel = get_kernel(obj['logits_train'][:M], obj['logits'][:M])
print()

# simple k-NN
k_range = [1, 3, 5, 7, 9, 11, 15, 25, 51, 101]
knn_preds = predict_knn(kernel, train_labels, k_range)
knn_accs = [100 * (pred == test_labels).float().mean().item() for pred in knn_preds]
print('k-NN accuracy (by choice of k):')
print(''.join(['%d\t' % k for k in k_range]))
print(''.join(['%.2f\t' % k for k in knn_accs]))

# Weighted k-NN
nbrs = kernel.topk(k=20, dim=1)
nbr_labels = train_labels[nbrs.indices.cpu()]
nbr_weights = nbrs.values.cpu()**4
labels_scores = (F.one_hot(nbr_labels) * nbr_weights[..., None]).mean(1)
pred = labels_scores.argmax(1)
acc_weighted = 100 * (pred == test_labels).float().mean().item()
print()
print('weighted k-NN accuracy (k=20, weighted by kernel value **4): %.2f' % acc_weighted)
print()

