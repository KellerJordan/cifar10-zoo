from ptflops import get_model_complexity_info
import torch
from torch import nn

act = lambda: nn.GELU()
bn = lambda ch: nn.BatchNorm2d(ch)
conv = lambda ch_in, ch_out: nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding='same', bias=False)
pool = lambda: nn.MaxPool2d(2)

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(len(x), -1)

k = 300
net = nn.Sequential(
    nn.Conv2d(3, k, kernel_size=3, padding=0),
    nn.MaxPool2d(3),
    nn.Conv2d(k, k, kernel_size=3, padding=0),
    nn.MaxPool2d(2),
    nn.Conv2d(k, k, kernel_size=3, padding=0),
    nn.MaxPool2d(2),
    Flatten(),
    nn.Linear(k, 300),
    nn.Linear(300, 100),
)
macs, params = get_model_complexity_info(net, (3, 32, 32), as_strings=True,
                                       print_per_layer_stat=True, verbose=True)
print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
print('{:<30}  {:<8}'.format('Number of parameters: ', params))


def print_flops(epochs):
    train_avg = 3.0
    infer_avg = 2
    fwd_flops = 2*float(macs.split()[0])*1e6
    run_flops = (fwd_flops * (epochs*50000*train_avg + 10000*infer_avg))
    pflops = run_flops / 1e15
    print('PFLOPs: %.3f' % pflops)
    
print_flops(500)
# This script computes the FLOPs used by the state-of-the-art method for CIFAR-10 in 2011
# https://arxiv.org/abs/1102.0183
# Section 4.1: "We use a variable learning rate that shrinks by a multiplicative constant after each epoch, from 10−3 down to 3 · 10−5 after 500 epochs."
# Section 4.3: "Like for MNIST, the learning rate is initialized by 0.001 and multiplied by 0.993 after every epoch."
# Conclusion: The CIFAR-10 model was also trained for 500 epochs.

