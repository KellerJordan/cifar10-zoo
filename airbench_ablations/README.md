# Airbench ablations

Each script adds a feature to the previous one.

`main_baseline.py`: Simply trains the architecture (using entirely standard initialization) with
Nesterov SGD and data augmentation, evaluated with random flipping test-time augmentation.
Reaches 94% mean accuracy in 35 epochs and 14.5 A100-seconds. [ 94.06 in n=25 ]

This architecture is taken from hlb-cifar10 with the following changes. (1) We put BatchNorms in fp32
because this results in faster epochs than putting them in fp16. (2) We replace the custom MaxPool at
the end with nn.MaxPool2d(3). (3) We reduce the final block width from 512 to 256.

`main_whiten.py`: Adds whitening initialization of the first conv layer, and removes proceeding BatchNorm.
94% in 22 epochs / 9.0 A100-seconds. [ 94.07 in n=25 ]

`main_scalebias.py`: Adds scaling of learning rate for BatchNorm biases by 64x.
94% in 17 epochs / 6.9 A100-seconds. [ 94.07 in n=25 ]

`main_dirac.py`: Adds dirac/identity initialization to conv filters.
94% in 14 epochs / 5.7 seconds. [ 94.07 in n=25 ]

`main_lookahead.py`: Adds the lookahead / EMA based optimization scheme from hlb-cifar10.

* progressive freezing
* more tta
* alternating flip

`main_freeze.py`: Adds progressive freezing of first conv layer biases after 3 epochs. No change to
epochs or accuracy, but increases speed to XX A100-seconds.

* `main_uncompiled.py`: same as full airbench but without `torch.compile`
* `main_basic.py`: 

