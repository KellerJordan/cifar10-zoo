# Scaling

We develop a variant of airbench optimized for time-to-96%.
This is a much harder target which ends up requiring 20x more compute.

| Script | Description | Time | PFLOPs | Epochs |
| - | - | - | - | - | 
| `resnet18_96p.py` | Baseline ResNet18 training script optimized for time-to-96%. | 186s | 13.4 | 80 |
| `long_dawnbench.py` | Long ResNet-9 training described in [How to Train Your ResNet](https://myrtle.ai/learn/how-to-train-your-resnet-8-bag-of-tricks/) | 104s | 8.3 | 70 |
| `waterbench.py` | Scaled variant of airbench optimized for time-to-96%. | 49s | 7.5 | 40 |

