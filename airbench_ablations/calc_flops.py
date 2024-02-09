from ptflops import get_model_complexity_info
from torch import nn

act = nn.GELU()
bn = lambda ch: nn.BatchNorm2d(ch)
conv = lambda ch_in, ch_out: nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding='same', bias=False)
pool = nn.MaxPool2d(2)

net = nn.Sequential(
    nn.Sequential(
        nn.Conv2d(3, 24, kernel_size=2, padding=0, bias=False),
        bn(24), act,
    ),
    nn.Sequential(
        conv(24, 64),
        pool,
        bn(64), act,
        conv(64, 64),
        bn(64), act
    ),
    nn.Sequential(
        conv(64, 256),
        pool,
        bn(256), act,
        conv(256, 256),
        bn(256), act
    ),
    nn.Sequential(
        conv(256, 256),
        pool,
        bn(256), act,
        conv(256, 256),
        bn(256), act
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
    print('PFLOPS: %.2f' % pflops)
    
print_flops(35, False, False)
print_flops(21, False, False)
print_flops(18, False, False)
print_flops(13.5, False, False)
print_flops(13.5, False, True)
print_flops(12.0, False, True)
print_flops(10.8, True, True)
print_flops(9.9, True, True)
