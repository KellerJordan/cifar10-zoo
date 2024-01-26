import torch
from torch import nn

# https://github.com/libffcv/ffcv/blob/77f11242cf9055b15fcaf9d2bb8e320de68dbfac/examples/cifar/train_cifar.py#L130
def make_net(w=1.0):

    class Mul(nn.Module):
        def __init__(self, weight):
            super().__init__()
            self.weight = weight
        def forward(self, x):
            return x * self.weight

    class Flatten(nn.Module):
        def forward(self, x):
            return x.view(len(x), -1)

    class Residual(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, x):
            return x + self.module(x)

    def conv_bn(channels_in, channels_out, kernel_size=3, stride=1, padding=1):
        bn = nn.BatchNorm2d(channels_out)
        bn.weight.requires_grad = False
        return nn.Sequential(
                nn.Conv2d(channels_in, channels_out,
                          kernel_size=kernel_size, stride=stride, padding=padding,
                          bias=False),
                bn,
                nn.ReLU(inplace=True)
        )

    NUM_CLASSES = 10
    w1 = int(w*64)
    w2 = int(w*128)
    w3 = int(w*256)
    model = nn.Sequential(
        conv_bn(3, w1, kernel_size=3, stride=1, padding=1),
        conv_bn(w1, w2, kernel_size=5, stride=2, padding=2),
        Residual(nn.Sequential(conv_bn(w2, w2), conv_bn(w2, w2))),
        conv_bn(w2, w3, kernel_size=3, stride=1, padding=1),
        nn.MaxPool2d(2),
        Residual(nn.Sequential(conv_bn(w3, w3), conv_bn(w3, w3))),
        conv_bn(w3, w2, kernel_size=3, stride=1, padding=0),
        nn.AdaptiveMaxPool2d((1, 1)),
        Flatten(),
        nn.Linear(w2, NUM_CLASSES, bias=False),
        Mul(0.2)
    )
    model = model.to(memory_format=torch.channels_last)
    model = model.cuda()
    for m in model.modules():
        if type(m) is not nn.BatchNorm2d:
            m.half()
    return model

