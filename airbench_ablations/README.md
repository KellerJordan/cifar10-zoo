# Airbench ablations

Each script adds a feature to the previous one.

The architecture is taken from hlb-cifar10, with the following changes. (1) We put BatchNorms in fp32
because this results in faster epochs than putting them in fp16. (2) We replace the custom MaxPool at
the end with nn.MaxPool2d(3). (3) We reduce the final block width from 512 to 256. (4) We add a
learnable bias to the first conv layer.

| Script | Feature | Epochs | Time | Evidence for >= 94% |
| - | - | - | - | - |
| `main0_baseline.py` | Trains the network with standard initialization, Nesterov SGD and data augmentation. Evaluates using random flipping TTA. | 35.0 | 14.5 A100-seconds | 94.06 in n=25 |
| `main1_whiten.py` | Adds whitening initialization for the first layer & removes proceeding BatchNorm. | 21.0 | 8.6 | 94.00 in n=200 |
| `main2_dirac.py` | Uses dirac/identity initialization for all conv filters. | 18.0 | 7.3 | 94.01 in n=200 |
| `main3_scalebias.py` | Adds scaling of learning rate for BatchNorm biases by 64x. | 13.5 | 5.5 | 94.01 in n=200 |
| `main4_freeze.py` | Freezes first conv layer bias after 3 epochs. | 13.5 | 5.2 | 94.03 in n=500 |
| `main5_lookahead.py` | Adds lookahead / EMA-based optimization scheme from hlb-cifar10. | 12.0 | 4.6 | 94.03 in n=50 |
| `main6_tta.py` | Adds extra multi-crop TTA. | 10.8 | 4.3 | 94.01 in n=25 |
| `main7_mirror.py` | Adds alterating strategy for left-right flip augmentation. | 9.9 | 3.95 | 94.02 in n=700 |
| `main8_compile.py` | Adds usage of `torch.compile`. This is the final version. | 9.9 | 3.5 | 94.02 in n=700 |

---
Note: lookahead only helps when combined with fast BatchNorm momentum, and vice versa.

### Categories
* Initialization (whiten, dirac)
* Optimization (scalebias, lookahead, progressive freezing)
* Evaluation (more tta)
* Data distribution (alternating flip)

