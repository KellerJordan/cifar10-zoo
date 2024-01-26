`main_baseline.py`: Simply trains the architecture (with entirely standard init) using Nesterov SGD
and data augmentation, evaluated with random flipping test-time augmentation. Requires 40 epochs to
reach 94.1%, in 16.6 A100-seconds.

`main_whiten.py`: Adds whitening initialization of the first conv layer, and removes proceeding
BatchNorm. Requires XX epochs to reach 94%, in XX A100-seconds.

`main_dirac.py`: Adds dirac/identity initialization to conv filters. Epochs reduced to XX so that
we reach 94% in XX A100-seconds.

`main_biasscaling.py`: Scales learning rate for BatchNorm biases by 64x -> XX epochs in XX A100-seconds.

`main_freeze.py`: Adds progressive freezing of first conv layer biases after 3 epochs. No change to
epochs or accuracy, but increases speed to XX A100-seconds.

* `main_uncompiled.py`: same as full airbench but without `torch.compile`
* `main_basic.py`: 

