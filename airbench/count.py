import glob
import torch
from tqdm import tqdm

pp = glob.glob('./logs/*/*.pt')
counts = []
for p in tqdm(pp):
    obj = torch.load(p)
    if isinstance(obj, dict):
        counts.append(len(obj['accs']))
    else:
#         print(5)
        pass
counts = torch.tensor(counts)
print(counts.sum(), counts[counts < 2000].sum())
