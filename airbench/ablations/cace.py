## snippet to compute an accurate estimate of the CACE (jiang et al 2021)
pred = logits.argmax(2)
pp = (pred[:, :, None] == torch.arange(10)).float().mean(0)

qq = torch.tensor([pp.quantile(q) for q in torch.arange(0, 1.00001, 0.005)]).unique()
qq = torch.tensor([-0.0001, 0]+qq[(qq > 0) & (qq < 1)].tolist()+[1])

cace = 0
for q1, q2 in zip(qq[:-1], qq[1:]):
    total = 0
    correct = 0
    for c in range(10):
        xx = pp[:, c]
        mask = (xx > q1) & (xx <= q2)
        correct += (labels[mask] == c).sum()
        total += mask.sum()
    
    avg_q = pp[(pp >= q1) & (pp <= q2)].mean()
    diff = (correct / total - avg_q).abs()
    cace += diff * (total / len(labels))
    
print(cace)
