# CIFAR-10 technology zoo

### Training speed records

| Script | Description |
| - | - |
| [airbench94_compiled.py](./airbench94_compiled.py) | 94% on CIFAR-10 in 3.5 seconds |
| [airbench95.py](./airbench_ablations/scaling/airbench95.py) | 95% on CIFAR-10 in 10.8 seconds |
| [airbench96.py](./airbench_ablations/scaling/airbench96.py) | 96% on CIFAR-10 in 49 seconds |
| [dawnbench.py](./dawnbench_replication/dawnbench.py) | Simplified replication of [How to Train Your ResNet](https://myrtle.ai/learn/how-to-train-your-resnet-8-bag-of-tricks/) |

[Ablations](./airbench_ablations)

All timings are on a single NVIDIA A100.

