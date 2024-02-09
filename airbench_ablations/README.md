# Airbench ablations

Each script in the series adds a feature to the previous one.

Our network architecture is the same as that in [hlb-cifar10](https://github.com/tysam-code/hlb-CIFAR10), with the following changes.
1. We put BatchNorms in fp32 because this results in faster epochs than putting them in fp16.
2. We replace the custom MaxPool at the end with nn.MaxPool2d(3).
3. We reduce the final block width from 512 to 256.
4. We add a learnable bias to the first conv layer.

Runtimes are measured in seconds on a single NVIDIA A100. Each script attains slightly over 94% accuracy.

| Script | Feature | Time | PFLOPs | Epochs |
| - | - | - | - | - |
| `main0_network.py` | Trains with standard initialization, Nesterov SGD and data augmentation. Evaluates with random-flip TTA. | 14.5 | 1.26 | 35.0 |
| `main1_whiten.py` | Initializes first conv layer as whitening transform & removes proceeding BatchNorm. | 8.6 | 0.76 | 21.0 |
| `main2_dirac.py` | Initializes all other conv layers as (partly) identity transforms. | 7.3 | 0.65 | 18.0 |
| `main3_scalebias.py` | Scales up the learning rate for BatchNorm biases by 64x. | 5.5 | 0.49 | 13.5 |
| `main4_freeze.py` | Freezes first conv layer bias after 3 epochs. | 5.2 | 0.47 | 13.5 |
| `main5_lookahead.py` | Adds lookahead / EMA-based optimization scheme from hlb-cifar10. | 4.6 | 0.42 | 12.0 |
| `main6_tta.py` | Evaluates with extra multi-crop TTA. | 4.3 | 0.39 | 10.8 |
| `main7_mirror.py` | Replaces the standard random-flip data augmentation with a semi-deterministic alternating flip method. | 3.9 | 0.36 | 9.9 |
| `main8_compile.py` | Compiles model with `torch.compile`. This is the final version. | 3.5 | 0.36 | 9.9 |

---
Note: lookahead only helps when combined with fast BatchNorm momentum, and vice versa.

### Categories
* Initialization: `whiten`, `dirac`
* Optimization: `scalebias`, `freeze`, `lookahead`
* Evaluation: `tta`
* Data distribution: `mirror`

### Baselines
| Script | Feature | Time | PFLOPs | Epochs |
| - | - | - | - | - |
| `train_resnet18.py` | Baseline ResNet-18 training script optimized for time-to-94% | 52.1  | 4.35 | 26.0 |
| [cifar10-fast](https://github.com/davidcpage/cifar10-fast) | Fast training script as described in [How to Train Your ResNet](https://myrtle.ai/learn/how-to-train-your-resnet/) | 14.9 | 1.14 | 10.0 |
| [hlb-cifar10](https://github.com/tysam-code/hlb-CIFAR10) | Hyperlightspeedbench -- fast training script & and prev record holder | 6.2 | 0.60 | 12.1 |

