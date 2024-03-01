# Scaling

We develop a variant of airbench which reaches 96% accuracy (~= full ResNet-18 training).
This is a harder target which requires 20x more compute.

We use a larger network that contains 10 convolutional layers, residual connections, and a
total of 12.7M parameters (a ResNet-18 has 11.2M).

| Script | Description | Time | PFLOPs | Epochs |
| - | - | - | - | - | 
| `train_resnet18_96.py` | Baseline ResNet18 training script optimized for time-to-96%. | 167s | 13.4 | 80 |
| `long_dawnbench.py` | Long ResNet-9 training described in [How to Train Your ResNet](https://myrtle.ai/learn/how-to-train-your-resnet-8-bag-of-tricks/) | 104s | 8.3 | 70 |
| `airbench96.py` | Scaled-up variant of airbench optimized for time-to-96%. | 49s | 7.5 | 40 |

We additionally develop a variant for 95% accuracy.

| Script | Description | Time | PFLOPs | Epochs |
| - | - | - | - | - | 
| `airbench95.py` | Scaled-up variant of airbench optimized for time-to-95%. | 10.8s | 1.4 | 15 |

