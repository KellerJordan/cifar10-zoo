train() {
python train4.py --config-file rn18_16_epochs.yaml \
    --data.train_dataset=./imagenet_ffcv/train_400_0.10_90.ffcv \
    --data.val_dataset=./imagenet_ffcv/val_400_0.10_90.ffcv \
    --data.num_workers=26 --data.in_memory=1 \
    --logging.folder=./logs/
}

train()
train()
train()
train()
train()
train()
train()
train()
train()
train()
