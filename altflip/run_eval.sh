#!/bin/bash
python eval.py --config-file rn18_16_epochs.yaml \
    --data.val_dataset=./imagenet_ffcv/val_400_0.10_90.ffcv \
    --data.num_workers=26 --data.in_memory=1

