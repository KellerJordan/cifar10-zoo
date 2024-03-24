import torch
import torch as ch
from torch.cuda.amp import autocast
import torch.nn.functional as F
ch.backends.cudnn.benchmark = True
ch.autograd.profiler.emit_nvtx(False)
ch.autograd.profiler.profile(False)

import torchvision
from torchvision import models
import torchmetrics
import numpy as np
from tqdm import tqdm

import os
import time
import json
from pathlib import Path
from argparse import ArgumentParser

from fastargs import get_current_config
from fastargs.decorators import param
from fastargs import Param, Section
from fastargs.validation import And, OneOf

from ffcv.loader import Loader, OrderOption
from ffcv.transforms import ToTensor, ToDevice, Squeeze, \
    RandomHorizontalFlip, ToTorchImage, Convert
from ffcv.fields.rgb_image import CenterCropRGBImageDecoder
from ffcv.fields.basics import IntDecoder

Section('model', 'model details').params(
    arch=Param(And(str, OneOf(models.__dir__())), default='resnet18'),
)

Section('data', 'data related stuff').params(
    val_dataset=Param(str, '.dat file to use for validation', required=True),
    num_workers=Param(int, 'The number of workers', required=True),
    in_memory=Param(int, 'does the dataset fit in memory? (1/0)', required=True)
)

Section('validation', 'Validation parameters stuff').params(
    batch_size=Param(int, 'The batch size for validation', default=512),
    resolution=Param(int, 'final resized validation image size', default=224),
    lr_tta=Param(int, 'should do lr flipping/avging at test time', default=1)
)

Section('training', 'training hyper param stuff').params(
    use_blurpool=Param(int, 'use blurpool?', default=0)
)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]) * 255
IMAGENET_STD = np.array([0.229, 0.224, 0.225]) * 255

class BlurPoolConv2d(ch.nn.Module):
    def __init__(self, conv):
        super().__init__()
        default_filter = ch.tensor([[[[1, 2, 1], [2, 4, 2], [1, 2, 1]]]]) / 16.0
        filt = default_filter.repeat(conv.in_channels, 1, 1, 1)
        self.conv = conv
        self.register_buffer('blur_filter', filt)

    def forward(self, x):
        blurred = F.conv2d(x, self.blur_filter, stride=1, padding=(1, 1),
                           groups=self.conv.in_channels, bias=None)
        return self.conv.forward(blurred)

class ImageNetTrainer:
    def __init__(self):
        self.all_params = get_current_config()

        self.model1 = self.create_model()
        self.model2 = self.create_model()
        self.model3 = self.create_model()
        self.model4 = self.create_model()
        self.initialize_logger()

    def load(self, checkpoint):
        checkpoint = Path(checkpoint)
        self.model1.load_state_dict(torch.load(checkpoint / 'final_weights1.pt'))
        self.model2.load_state_dict(torch.load(checkpoint / 'final_weights2.pt'))
        self.model3.load_state_dict(torch.load(checkpoint / 'final_weights3.pt'))
        self.model4.load_state_dict(torch.load(checkpoint / 'final_weights4.pt'))

    @param('data.val_dataset')
    @param('data.num_workers')
    @param('validation.batch_size')
    def create_val_loader(self, val_dataset, num_workers, batch_size):
        assert Path(val_dataset).is_file()

        #cropper = CenterCropRGBImageDecoder((256, 256), ratio=0.875)
        cropper = CenterCropRGBImageDecoder((192, 192), ratio=1.0)
        image_pipeline = [
            cropper,
            ToTensor(),
            ToDevice(ch.device(0), non_blocking=True),
            ToTorchImage(),
            Convert(ch.float16),
            torchvision.transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]

        label_pipeline = [
            IntDecoder(),
            ToTensor(),
            ToDevice(ch.device(0), non_blocking=True),
            Squeeze(),
        ]

        loader = Loader(val_dataset,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        order=OrderOption.SEQUENTIAL,
                        drop_last=False,
                        pipelines={
                            'image': image_pipeline,
                            'label': label_pipeline
                        })
        return loader

    @param('model.arch')
    @param('training.use_blurpool')
    def create_model(self, arch, use_blurpool):
        model = getattr(models, arch)()
        def apply_blurpool(mod: ch.nn.Module):
            for (name, child) in mod.named_children():
                if isinstance(child, ch.nn.Conv2d) and (np.max(child.stride) > 1 and child.in_channels >= 16): 
                    setattr(mod, name, BlurPoolConv2d(child))
                else: apply_blurpool(child)
        if use_blurpool: apply_blurpool(model)

        model = model.to(memory_format=ch.channels_last)
        model = model.cuda()

        return model

    @param('validation.lr_tta')
    def val_loop(self, lr_tta):
        lr_tta = False
        self.model1.eval()
        self.model2.eval()
        self.model3.eval()
        self.model4.eval()

        with ch.no_grad(), autocast():
            for images, target, _ in tqdm(self.val_loader):
                output1 = self.model1(images)
                output2 = self.model2(images)
                output3 = self.model3(images)
                output4 = self.model4(images)
                if lr_tta:
                    output1 += self.model1(ch.flip(images, dims=[3]))
                    output2 += self.model2(ch.flip(images, dims=[3]))
                    output3 += self.model3(ch.flip(images, dims=[3]))
                    output4 += self.model4(ch.flip(images, dims=[3]))

                for k in ['top_1_model1', 'top_5_model1']:
                    self.val_meters[k](output1, target)
                for k in ['top_1_model2', 'top_5_model2']:
                    self.val_meters[k](output2, target)
                for k in ['top_1_model3', 'top_5_model3']:
                    self.val_meters[k](output3, target)
                for k in ['top_1_model4', 'top_5_model4']:
                    self.val_meters[k](output4, target)

        stats = {k: m.compute().item() for k, m in self.val_meters.items()}
        [meter.reset() for meter in self.val_meters.values()]
        return stats

    def initialize_logger(self):
        self.val_meters = { 
            'top_1_model1': torchmetrics.Accuracy(task='multiclass', num_classes=1000).cuda(),
            'top_1_model2': torchmetrics.Accuracy(task='multiclass', num_classes=1000).cuda(),
            'top_1_model3': torchmetrics.Accuracy(task='multiclass', num_classes=1000).cuda(),
            'top_1_model4': torchmetrics.Accuracy(task='multiclass', num_classes=1000).cuda(),
            'top_5_model1': torchmetrics.Accuracy(task='multiclass', num_classes=1000, top_k=5).cuda(),
            'top_5_model2': torchmetrics.Accuracy(task='multiclass', num_classes=1000, top_k=5).cuda(),
            'top_5_model3': torchmetrics.Accuracy(task='multiclass', num_classes=1000, top_k=5).cuda(),
            'top_5_model4': torchmetrics.Accuracy(task='multiclass', num_classes=1000, top_k=5).cuda(),
        }   


def make_config(quiet=False):
    config = get_current_config()
    parser = ArgumentParser(description='Fast imagenet training')
    config.augment_argparse(parser)
    config.collect_argparse_args(parser)
    config.validate(mode='stderr')
    if not quiet:
        config.summary()

if __name__ == "__main__":
    make_config()
    trainer = ImageNetTrainer()
    trainer.val_loader = trainer.create_val_loader()

    import glob
    #ckpts = glob.glob('/home/ubuntu/ffcv/ffcv-imagenet/logs_translate_noflip/*')
    ckpts = glob.glob('/home/ubuntu/ffcv/ffcv-imagenet/logs_noflip/*')

    obj = {}
    for ckpt in ckpts:
        key = os.path.basename(ckpt)
        if not os.path.exists(os.path.join(ckpt, 'log')):
            continue
        trainer.load(ckpt)
        stats = trainer.val_loop()
        obj[key] = stats

    torch.save(obj, 'result_eval.pt')

