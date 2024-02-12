# Scaling

We develop a variant of airbench which reaches 96% accuracy (>= full ResNet-18 training).
This is a harder target which ends up requiring 20x more compute.

| Script | Description | Time | PFLOPs | Epochs |
| - | - | - | - | - | 
| `resnet18_96p.py` | Baseline ResNet18 training script optimized for time-to-96%. | 186s | 13.4 | 80 |
| `long_dawnbench.py` | Long ResNet-9 training described in [How to Train Your ResNet](https://myrtle.ai/learn/how-to-train-your-resnet-8-bag-of-tricks/) | 104s | 8.3 | 70 |
| `waterbench.py` | Scaled-up variant of airbench optimized for time-to-96%. | 49s | 7.5 | 40 |
| `waterbench_lite.py` | Scaled-up variant of airbench optimized for time-to-95%. | 10.8s | 1.4 | 15 |

