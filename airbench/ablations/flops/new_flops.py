from ptflops import get_model_complexity_info
from torch import nn

act = lambda: nn.GELU()
bn = lambda ch: nn.BatchNorm2d(ch)
conv = lambda ch_in, ch_out, k=3, s=1: nn.Conv2d(ch_in, ch_out, kernel_size=k, stride=s, padding='same' if s == 1 else 1, bias=False)
pool = lambda: nn.MaxPool2d(2)

w1 = 128
#w1 = 48
#w2 = 256
#w2 = 128
#w2 = 192
#w3 = 256
#w3 = 192
w2 = 384
w3 = 384

net = nn.Sequential(
    nn.Sequential(
        nn.Conv2d(3, 24, kernel_size=2, padding=0, bias=False),
        bn(24), act(),
    ),
    nn.Sequential(
        conv(24, w1, k=3, s=1),
        #pool(),
        bn(w1), act(),
        conv(w1, w1, k=3),
        bn(w1), act(),
    ),
    nn.Sequential(
        conv(w1, w2, k=3, s=1),
        pool(),
        bn(w2), act(),
        conv(w2, w2, k=3),
        bn(w2), act(),
    ),
    nn.Sequential(
        conv(w2, w3, k=3, s=1),
        pool(),
        bn(w3), act(),
        conv(w3, w3, k=3),
        bn(w3), act(),
    ),
    nn.MaxPool2d(3),
)
macs, params = get_model_complexity_info(net, (3, 32, 32), as_strings=True,
                                       print_per_layer_stat=True, verbose=True)
print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
print('{:<30}  {:<8}'.format('Number of parameters: ', params))

def print_flops(epochs, tta, freeze):
    if freeze:
        train_avg = ((3/epochs) * 3.0 + (epochs-3)/epochs * (3.0 - 0.11141))
    else:
        train_avg = 3.0
    if tta:
        infer_avg = 6
    else:
        infer_avg = 2
    fwd_flops = 2*float(macs.split()[0])*1e6
    run_flops = (fwd_flops * (epochs*50000*train_avg + 10000*infer_avg))
    pflops = run_flops / 1e15
    print('PFLOPs: %.3f' % pflops)
    
print_flops(15, True, True)

