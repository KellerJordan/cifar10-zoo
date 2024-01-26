# Airbench ablations

Each script adds a feature to the previous one.

`main0_baseline.py`: Simply trains the architecture (using entirely standard initialization) with
Nesterov SGD and data augmentation, evaluated with random flipping test-time augmentation.
Reaches 94% mean accuracy in 35 epochs and 14.5 A100-seconds. [ 94.06 in n=25 ]

This architecture is taken from hlb-cifar10, with the following changes. (1) We put BatchNorms in fp32
because this results in faster epochs than putting them in fp16. (2) We replace the custom MaxPool at
the end with nn.MaxPool2d(3). (3) We reduce the final block width from 512 to 256. (4) We add a
learnable bias to the first conv layer.

`main1_whiten.py`: Adds whitening initialization of the first conv layer, and removes proceeding BatchNorm.
-> 94% in 21 epochs / 8.6 seconds. [ 94.00 in n=200 ]

`main2_dirac.py`: Uses dirac/identity initialization for all conv filters.
-> 94% in 18 epochs / 7.3 seconds. [ 94.01 in n=200 ]

`main3_scalebias.py`: Adds scaling of learning rate for BatchNorm biases by 64x.
-> 94% in 13.5 epochs / 5.5 seconds. [ 94.01 in n=200 ]

`main4_freeze.py`: Freezes first conv layer bias after 3 epochs.
-> 94% in 13.5 epochs / 5.2 seconds. [ 94.03 in n=500 ]

#`main_lookahead.py`: Adds the lookahead / EMA-based optimization scheme from hlb-cifar10.
#-> [ 93.97% in n=25 ] -- on top of `main_freeze`

`main_tta`: (on top of `main_freeze`): Adds extra multi-crop TTA.
n=25
-> 94.15% in 13.5 epochs / 5.3 seconds.
-> 94.10% in 12.0 epochs / 4.71 seconds.
-> 93.97% in 11.0 epochs / 4.4 seconds.
-> 93.98% in 11.5 epochs / 4.53 seconds.


---
Note: lookahead only helps when combined with fast BatchNorm momentum, and vice versa.



* alternating flip

* `main_uncompiled.py`: same as full airbench but without `torch.compile`

### Categories
* Initialization (whiten, dirac)
* Optimization (scalebias, lookahead, progressive freezing)
* Evaluation (more tta)
* Data distribution (alternating flip)

## TODO
* run `main_fastbn_lrsched.py` -- does fastbn + new lr sched do better than `main4_freeze`? doubt either one does alone

