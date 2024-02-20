# CIFAR-10 technology zoo

### Training speed records

| Script | Description |
| - | - |
| [airbench.py](./airbench.py) | 94% on CIFAR-10 in 3.5 seconds |
| [airbench_medium.py](./airbench_ablations/scaling/airbench_medium.py) | 95% on CIFAR-10 in 10.8 seconds |
| [airbench_heavy.py](./airbench_ablations/scaling/airbench_heavy.py) | 96% on CIFAR-10 in 49 seconds |
| [dawnbench.py](./dawnbench_replication/dawnbench.py) | Simplified replication of [How to Train Your ResNet](https://myrtle.ai/learn/how-to-train-your-resnet-8-bag-of-tricks/) |

All timings are on a single NVIDIA A100.

[Ablations](./airbench_ablations)

