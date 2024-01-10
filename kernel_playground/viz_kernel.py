# This script consumes the output of main_saveoutputs.py
# It should be pointed to the output directory of that script, which by default is ./logs_saveoutputs/

import os
import sys
import glob
import uuid
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torchvision
train_labels = torch.tensor(torchvision.datasets.CIFAR10('/tmp/cifar10', download=True, train=True).targets)
test_labels = torch.tensor(torchvision.datasets.CIFAR10('/tmp/cifar10', download=True, train=False).targets)

##################################
#          Load outputs          #
##################################
path = './logs_saveoutputs/'
if len(sys.argv) >= 2:
    path = sys.argv[1]
print('loading batches of outputs from %s...' % path)
pp = glob.glob(os.path.join(path, '*/*.pt'))
objs = [torch.load(p, map_location='cpu') for p in tqdm(pp)]
obj = {k: torch.cat([o[k] for o in objs]) for k in objs[0].keys() if 'logit' in k}
del objs

##################################
# Compute kernel k-NN accuracies #
##################################

def renorm(xx):
    return (xx - xx.mean(0, keepdim=True)) / xx.std(0, keepdim=True)

logits_train = obj['logits_train'].float().log_softmax(-1)
logits_test = obj['logits'].float().log_softmax(-1)
num_models = len(logits_train)

M = min(10000, num_models)
mm = torch.logspace(0, 4, 20).long()
mm = torch.tensor(sorted(set(mm.tolist())))
mm = mm[:(mm <= M).sum()]

n_seeds = 20
k_range = [1, 3, 5, 7, 9, 15, 25, 50]

ens_accs = []
accs_knn = {k: [] for k in k_range}
accs_weighted = []

for m in tqdm(mm):

    ens_accs_m = []
    accs_knn_m = {k: [] for k in k_range}
    accs_weighted_m = []
    
    for i in range(n_seeds):
        
        i_start = int(m*i/2)
        if i_start+m > num_models:
            break
        logits_train1 = logits_train[i_start:i_start+m]
        logits_test1 = logits_test[i_start:i_start+m]
        
        ens_acc = (logits_test1.mean(0).argmax(1) == test_labels).float().mean()
        ens_accs_m.append(ens_acc)
        
        num_train_examples = logits_train.shape[1]
        num_test_examples = logits_test.shape[1]
        kernel = torch.zeros(num_test_examples, num_train_examples).cuda()
        for c in range(10):
            logits_train_cls = renorm(logits_train1[..., c]).cuda()
            logits_test_cls = renorm(logits_test1[..., c]).cuda()
            corr_cls = (logits_test_cls.T @ logits_train_cls) / m
            kernel += corr_cls

        nbrs = kernel.topk(k=max(k_range), dim=1)
        nbr_labels = train_labels[nbrs.indices.cpu()]
        for k in k_range:
            pred = nbr_labels[:, :k].mode(1).values
            accs_knn_m[k].append((pred == test_labels).float().mean())
        
        nbrs = kernel.topk(k=50, dim=1)
        nbr_labels = train_labels[nbrs.indices.cpu()]
        nbr_weights = nbrs.values.cpu()**4
        labels_scores = (F.one_hot(nbr_labels) * nbr_weights[..., None]).mean(1)
        pred = labels_scores.argmax(1)
        acc_weighted = (pred == test_labels).float().mean()
        accs_weighted_m.append(acc_weighted)
    
    ens_accs.append(torch.stack(ens_accs_m).mean())
    for k in k_range:
        accs_knn[k].append(torch.stack(accs_knn_m[k]).mean())
    accs_weighted.append(torch.stack(accs_weighted_m).mean())

log = dict(mm=mm, ens_accs=ens_accs, accs1=accs1, accs5=accs5, accs_weighted=accs_weighted)

log_dir = './viz_logs'
os.makedirs(log_dir, exist_ok=True)
log_path = os.path.join(log_dir, str(uuid.uuid4())+'.pt')
print(os.path.abspath(log_path))
torch.save(log, log_path)

