from tqdm import tqdm
import matplotlib.pyplot as plt
import torchvision
from quick_cifar import CifarLoader
loader = CifarLoader('/tmp/cifar10', train=False)
def viz(xx, figsize=12):
    plt.figure(figsize=(figsize, figsize))
    xx = xx.float()
    xx = xx.cuda()
    if len(xx.shape) == 2:
        xx = xx.reshape(len(xx), 3, -1)
        w = int(xx.size(2)**0.5)
        xx = xx.reshape(len(xx), 3, w, w)
    if xx.min() < 0 and xx.abs().max() < 2:
        xx = 2 * xx / xx.abs().max()
    if xx.min() < 0:
        xx = loader.denormalize(xx)
    grid = torchvision.utils.make_grid(xx)
    grid = grid.permute(1, 2, 0)
    grid = grid.clip(0, 1)
    plt.imshow(grid.cpu().numpy())
    plt.axis('off')
    plt.show()
    
import glob
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torchvision
train_labels = torch.tensor(torchvision.datasets.CIFAR10('/tmp/cifar10', download=True, train=True).targets)
test_labels = torch.tensor(torchvision.datasets.CIFAR10('/tmp/cifar10', download=True, train=False).targets)
train_labels = train_labels.cuda()
test_labels = test_labels.cuda()
from quick_cifar import CifarLoader
train_loader = CifarLoader('/tmp/cifar10', train=True)
test_loader = CifarLoader('/tmp/cifar10', train=False)

pp = glob.glob('/home/ubuntu/cifar10-zoo/logs_saveoutputs_rough/*/*.pt') # don't use label smoothing
objs = [torch.load(p, map_location='cpu') for p in tqdm(pp)]
print(len(objs))
obj = {k: torch.cat([o[k] for o in objs]) for k in objs[0].keys() if 'logit' in k}
del objs

xx = (obj['logits_tta'].argmax(-1) == test_labels.cpu()).float().mean(1)
print(xx.mean().item(), (xx.mean() - 1.96 * xx.std() / len(xx)**0.5).item())

logits = obj['logits_tta']
pred = logits.argmax(-1)
correct = (pred == test_labels.cpu()).float()
pp = correct.mean(0)
true_var = correct.sum(1).var()
pred_var = (pp * (1 - pp)).sum()
print(true_var.sqrt(), pred_var.sqrt(), (true_var - pred_var).sqrt())

ens_pred = obj['logits'].mean(0).argmax(1)
print('ens_acc :', (ens_pred == test_labels.cpu()).sum().item() / 100)
ens_pred = obj['logits_tta'].mean(0).argmax(1)
print('ens_acc (w/ tta) :', (ens_pred == test_labels.cpu()).sum().item() / 100)



logits = obj['logits_train'][:9000]
mu = logits.mean(0)
xx0 = norm(logits)

corr_mats = torch.zeros(len(mu), 10, 10).cuda()
for c1 in tqdm(range(10)):
    xx1 = xx0[..., c1].cuda()
    for c2 in range(10):
        xx2 = xx0[..., c2].cuda()
        corr = (xx1 * xx2).mean(0)
        corr_mats[:, c1, c2] = corr

plt.imshow(corr_mats.mean(0).cpu())
plt.show()


c1, c2 = 0, 9

feats1 = mu
corr_mats1 = corr_mats
x_vals = norm(feats1)
y_vals = norm(corr_mats1[:, c1, c2].view(-1))
nt = int(0.8 * len(y_vals))
x_train = x_vals[:nt]
y_train = y_vals[:nt]
x_test = x_vals[nt:]
y_test = y_vals[nt:]

import sklearn.linear_model
lin_model = sklearn.linear_model.LinearRegression(fit_intercept=False)
lin_model.fit(x_train.cpu(), y_train.cpu())
y_pred = torch.tensor(lin_model.predict(x_test.cpu())).cuda()
print(get_corr(y_test, y_pred))
res = []
for c in range(10):
    mask = (train_labels == c).cpu()[nt:]
    res.append(get_corr(y_test[mask], y_pred[mask]))
    plt.scatter(y_test[mask].tolist(), y_pred[mask].tolist(), s=2, label=c, alpha=0.5)
print(torch.stack(res).mean())
plt.legend()
plt.show()

import lightgbm
xgb_model = lightgbm.LGBMRegressor()
xgb_model.fit(x_train.cpu(), y_train.cpu())
y_pred = torch.tensor(xgb_model.predict(x_test.cpu())).cuda()
print(get_corr(y_test, y_pred))
res = []
for c in range(10):
    mask = (train_labels == c).cpu()[nt:]
    res.append(get_corr(y_test[mask], y_pred[mask]))
    plt.scatter(y_test[mask].tolist(), y_pred[mask].tolist(), s=2, label=c, alpha=0.5)
print(torch.stack(res).mean())
plt.legend()
plt.show()

## Now let's just consider two classes for simplicity
c1, c2 = 4, 9
print(test_loader.classes[c1], 'vs.', test_loader.classes[c2])
logits = obj['logits'][:9000, :, [c1, c2]].cuda()
xx0 = norm(logits)

corr = (xx0[:, :, 0] * xx0[:, :, 1]).mean(0)
diff = (logits[..., 0] - logits[..., 1]).mean(0)

plt.figure(figsize=(8, 6))
for c in range(10):
    mask = (test_labels == c)
    plt.scatter(diff[mask].tolist(), corr[mask].tolist(), s=2, label=test_loader.classes[c])
plt.plot([-15, 15], [0, 0], linestyle='--', color='gray')
plt.legend()
plt.xlabel('logodds(label 1 vs 2)')
plt.ylabel('corr(logit1, logit2)')
plt.show()

# 5, 9
# If I think it looks more like an X, then I also think it looks more like a Y
mask = (test_labels == 2) & (corr > +0.15)
print(logits.mean(0)[mask].mean(0))
viz(test_loader.images[mask][:16])
# If I think it looks more like an X, then I *don't* think it looks more like a Y
mask = (test_labels == 2) & (corr < -0.3)
print(logits.mean(0)[mask].mean(0))
viz(test_loader.images[mask][:16])

