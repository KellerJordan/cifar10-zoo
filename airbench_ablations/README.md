# Airbench ablations

Each script adds a feature to the previous one.

`main_baseline.py`: Simply trains the architecture (with entirely standard init) using Nesterov SGD
and data augmentation, evaluated with random flipping test-time augmentation. Reaches 94% mean accuracy
in 35 epochs and 14.5 A100-seconds. [ 94.06 in n=25 ]

`main_whiten.py`: Adds whitening initialization of the first conv layer, and removes proceeding BatchNorm.
94% in 22 epochs / 9.0 A100-seconds. [ 94.07 in n=25 ]

`main_scalebias.py`: Adds scaling of learning rate for BatchNorm biases by 64x.
94% in 18 epochs / 7.4 A100-seconds. []

`main_dirac.py`: Adds dirac/identity initialization to conv filters. 94% in XX

* progressive freezing
* lookahead
* more tta
* alternating flip



`main_freeze.py`: Adds progressive freezing of first conv layer biases after 3 epochs. No change to
epochs or accuracy, but increases speed to XX A100-seconds.

* `main_uncompiled.py`: same as full airbench but without `torch.compile`
* `main_basic.py`: 

