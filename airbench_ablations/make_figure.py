import matplotlib.pyplot as plt
import torch

# there are 8 total features, but only 6 of them actually reduce epochs.
# the other two -- whitening freeze and compile -- just speed up each epoch.
epochs = [
    34,
    21,
    18,
    13.5,
    12,
    10.8,
    9.9,
]
feats = list(range(len(epochs)-1))
speedup_add = [epochs[i+1] - epochs[i] for i in feats]
speedup_mul = [epochs[i+1] / epochs[i] for i in feats]


configs = [
    [False, True, True, True, True, True],
    [True, False, True, True, True, True],
    [True, True, False, True, True, True],
    [True, False, False, True, True, True],
    [True, False, False, False, False, True],
    [True, False, False, False, True, False],
]
results = [
    22.0, # main9a_nowhiten
    12.7, # main9b_nodirac
    14.2, # main9c_noscalebias
    17.0, # main9g_nodirac_noscalebias - tbd
    20.0, # main9e_whiten_mirror
    18.1, # main9f_whiten_tta
]

preds_add = []
preds_mul = []
for cfg in configs:
    pred_add = sum([speedup_add[i] for i in feats if cfg[i]])
    preds_add.append(epochs[0] + pred_add)
    pred_mul = torch.prod(torch.tensor([speedup_mul[i] for i in feats if cfg[i]]))
    preds_mul.append(epochs[0] * pred_mul)


    
kwargs = dict(marker='o')
plt.plot(preds_add, label='Additive prediction', **kwargs)
plt.plot(preds_mul, label='Multiplicative prediction', **kwargs)
plt.plot(results, label='Ground truth', **kwargs)
plt.title('Speedup by feature configuration', fontsize=16)
plt.legend(fontsize=11)
plt.xlabel('Configuration', fontsize=14)
plt.ylabel('Epochs to 94%', fontsize=14)
xx = list(range(len(results)))
plt.xticks(xx, 'ABCDEFGH'[:len(xx)], fontsize=12)
plt.show()

