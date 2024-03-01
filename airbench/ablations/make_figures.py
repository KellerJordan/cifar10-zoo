import torch
import matplotlib.pyplot as plt

configs = {
    # baseline: all-on
    (True, True, True, True, True, True): 9.9,
    # single-off
    (False, True, True, True, True, True): 22,
    (True, False, True, True, True, True): 12.7,
    (True, True, False, True, True, True): 14.2,
    (True, True, True, False, True, True): 10.7,
    (True, True, True, True, False, True): 11.2,
    (True, True, True, True, True, False): 10.8,
    # baseline: all-off
    (False, False, False, False, False, False): 34,
    # single-on
    (True,  False, False, False, False, False): 21.0,
    (False, True,  False, False, False, False): 30.0,
    (False, False, True,  False, False, False): 34.0, # 40.0
    (False, False, False, True,  False, False): 33.0,
    (False, False, False, False, True,  False): 29.5,
    (False, False, False, False, False, True): 33.0,
}
results = list(configs.values())
speedup_add = [results[0] - results[i] for i in range(1, 7)]
speedup_mul = [results[0] / results[i] for i in range(1, 7)]

results1 = [results[7] - results[7+i] for i in range(1, 7) if 7+i < len(results)]
preds_add = [-speedup_add[i] for i in range(6)][:len(results1)]
preds_mul = [results[7] * (1-speedup_mul[i]) for i in range(6)][:len(results1)]

# rm2 = lambda t: torch.tensor(t)[[0, 1, 3, 4, 5]]
# results1, preds_add, preds_mul = map(rm2, (results1, preds_add, preds_mul))

plt.figure(figsize=(10, 5))
kwargs = dict(width=0.22)
xx = torch.tensor(list(range(len(results1))))
bars1 = plt.bar(xx-0.25, preds_mul, label='Multiplicative prediction', **kwargs)
bars2 = plt.bar(xx, preds_add, label='Additive prediction', **kwargs)
bars3 = plt.bar(xx+0.25, results1, label='Ground truth', **kwargs)
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        bar.set_edgecolor('#333')
for bar in bars1:
    bar.set_hatch('\\')
for bar in bars2:
    bar.set_hatch('/')
for bar in bars3:
    bar.set_hatch('..')
    
features = ['whiten', 'dirac', 'scalebias', 'lookahead', 'tta', 'mirror']

plt.title('Speedup by feature', fontsize=18)
plt.legend(fontsize=14)
# plt.xlabel('Configuration', fontsize=14)
plt.ylabel('Reduction in epochs to 94%', fontsize=14)
# plt.xticks(xx, 'ABCDEF', fontsize=14)
plt.xticks(xx, features, fontsize=14)
plt.tight_layout()
plt.savefig('figure1.png', dpi=200)

import torch
import matplotlib.pyplot as plt

configs = {
    # baseline: all-on
    (True, True, True, True, True, True): 9.9,
    # single-off
    (False, True, True, True, True, True): 22,
    (True, False, True, True, True, True): 12.7,
    (True, True, False, True, True, True): 14.2,
    (True, True, True, False, True, True): 10.7,
    (True, True, True, True, False, True): 11.2,
    (True, True, True, True, True, False): 10.8,
    # baseline: just whitening on
    (True, False, False, False, False, False): 21,
    # single-on, given whitening always on
    (True, True,  False, False, False, False): 18,
    (True, False, True,  False, False, False): 16.5,
    (True, False, False, True,  False, False): 20.2,
    (True, False, False, False, True,  False): 18.1,
    (True, False, False, False, False, True): 20.0,
}
results = list(configs.values())
speedup_add = [results[0] - results[i] for i in range(1, 7)]
speedup_mul = [results[0] / results[i] for i in range(1, 7)]

preds_add = [-speedup_add[i] for i in range(1, 6)]
preds_mul = [results[7] * (1-speedup_mul[i]) for i in range(1, 6)]
results1 = [results[7] - results[7+i] for i in range(1, 6)]


plt.figure(figsize=(10, 5))
kwargs = dict(width=0.22)
xx = torch.tensor(list(range(len(results1))))
bars1 = plt.bar(xx-0.25, preds_mul, label='Multiplicative prediction', **kwargs)
bars2 = plt.bar(xx, preds_add, label='Additive prediction', **kwargs)
bars3 = plt.bar(xx+0.25, results1, label='Ground truth', **kwargs)
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        bar.set_edgecolor('#333')
for bar in bars1:
    bar.set_hatch('\\')
for bar in bars2:
    bar.set_hatch('/')
for bar in bars3:
    bar.set_hatch('..')

plt.title('Speedup by feature added to whitening network', fontsize=18)
plt.legend(fontsize=14)
plt.ylabel('Reduction in epochs to 94%', fontsize=14)
plt.xticks(xx, ['+%s' % f for f in features[1:]], fontsize=14)
plt.tight_layout()
plt.savefig('figure2.png', dpi=200)

