# Airbench ablations

Each script in the `main\d_.*` series adds a feature to the previous one.

Our network architecture is the same as that in [hlb-cifar10](https://github.com/tysam-code/hlb-CIFAR10), with the following changes.
1. We put BatchNorms in fp32 because this results in faster epochs than putting them in fp16.
2. We replace the custom MaxPool at the end with nn.MaxPool2d(3).
3. We reduce the final block width from 512 to 256.
4. We add a learnable bias to the first conv layer.

Runtimes are measured in seconds on a single NVIDIA A100.

| Script | Feature | Time | TFLOPs | Epochs | Evidence for >= 94% |
| - | - | - | - | - | - |
| `train_resnet18.py` | ResNet-18 training script optimized for time-to-94% | 52.1  | 4350 | 26.0 | 94.01 in n=10 |
| [cifar10-fast](https://github.com/davidcpage/cifar10-fast) | Fast training script as described in [How to Train Your ResNet](https://myrtle.ai/learn/how-to-train-your-resnet/) | 14.9 | 1144 | 10.0 | -- |
| `main0_network.py` | Trains the network with standard initialization, Nesterov SGD and data augmentation. Evaluates using random-flip TTA. | 14.5 | 1223 | 35.0 | 94.06 in n=25 |
| `main1_whiten.py` | Initializes first conv layer as whitening transform & removes proceeding BatchNorm. | 8.6 | 735 | 21.0 | 94.00 in n=200 |
| `main2_dirac.py` | Initializes all other conv layers as (partly) identity transforms. | 7.3 | 631 | 18.0 | 94.01 in n=200 |
| [hlb-cifar10](https://github.com/tysam-code/hlb-CIFAR10) | Hyperlightspeedbench -- fast training script | 6.2 | 572 | 12.1 | -- |
| `main3_scalebias.py` | Scales the learning rate for BatchNorm biases by 64x. | 5.5 | 474 | 13.5 | 94.01 in n=200 |
| `main4_freeze.py` | Freezes first conv layer bias after 3 epochs. | 5.2 | 461 | 13.5 | 94.03 in n=500 |
| `main5_lookahead.py` | Adds lookahead / EMA-based optimization scheme from hlb-cifar10. | 4.6 | 410 | 12.0 | 94.02 in n=200 |
| `main6_tta.py` | Evaluates with extra multi-crop TTA. | 4.3 | 380 | 10.8 | 94.02 in n=200 |
| `main7_mirror.py` | Replaces random-flip data augmentation with a semi-deterministic alternating strategy. | 3.95 | 350 | 9.9 | 94.02 in n=700 |
| `main8_compile.py` | Compiles model with `torch.compile`. This is the final version. | 3.5 | 350 | 9.9 | 94.02 in n=700 |

---
Note: lookahead only helps when combined with fast BatchNorm momentum, and vice versa.

### Categories
* Initialization (`whiten`, `dirac`)
* Optimization (`scalebias`, `freeze`, `lookahead`)
* Evaluation (`tta`)
* Data distribution (`mirror`)

