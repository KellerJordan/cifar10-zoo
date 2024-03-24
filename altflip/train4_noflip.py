import sys
import hashlib
import torch
import torch as ch
from torch.cuda.amp import GradScaler
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
from uuid import uuid4
from typing import List
from pathlib import Path
from argparse import ArgumentParser

from fastargs import get_current_config
from fastargs.decorators import param
from fastargs import Param, Section
from fastargs.validation import And, OneOf

from ffcv.pipeline.operation import Operation
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import ToTensor, ToDevice, Squeeze, \
    RandomHorizontalFlip, ToTorchImage, Convert
from ffcv.fields.rgb_image import CenterCropRGBImageDecoder, \
    RandomResizedCropRGBImageDecoder
from ffcv.fields.basics import IntDecoder

Section('model', 'model details').params(
    arch=Param(And(str, OneOf(models.__dir__())), default='resnet18'),
)

Section('resolution', 'resolution scheduling').params(
    min_res=Param(int, 'the minimum (starting) resolution', default=160),
    max_res=Param(int, 'the maximum (starting) resolution', default=160),
    end_ramp=Param(int, 'when to stop interpolating resolution', default=0),
    start_ramp=Param(int, 'when to start interpolating resolution', default=0)
)

Section('data', 'data related stuff').params(
    train_dataset=Param(str, '.dat file to use for training', required=True),
    val_dataset=Param(str, '.dat file to use for validation', required=True),
    num_workers=Param(int, 'The number of workers', required=True),
    in_memory=Param(int, 'does the dataset fit in memory? (1/0)', required=True)
)

Section('lr', 'lr scheduling').params(
    step_ratio=Param(float, 'learning rate step ratio', default=0.1),
    step_length=Param(int, 'learning rate step length', default=30),
    lr_schedule_type=Param(OneOf(['step', 'cyclic']), default='cyclic'),
    lr=Param(float, 'learning rate', default=0.5),
    lr_peak_epoch=Param(int, 'Epoch at which LR peaks', default=2),
)

Section('logging', 'how to log stuff').params(
    folder=Param(str, 'log location', required=True),
)

Section('validation', 'Validation parameters stuff').params(
    batch_size=Param(int, 'The batch size for validation', default=512),
    resolution=Param(int, 'final resized validation image size', default=224),
    lr_tta=Param(int, 'should do lr flipping/avging at test time', default=1)
)

Section('training', 'training hyper param stuff').params(
    batch_size=Param(int, 'The batch size', default=512),
    momentum=Param(float, 'SGD momentum', default=0.9),
    weight_decay=Param(float, 'weight decay', default=4e-5),
    epochs=Param(int, 'number of epochs', default=30),
    label_smoothing=Param(float, 'label smoothing parameter', default=0.1),
    use_blurpool=Param(int, 'use blurpool?', default=0)
)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]) * 255
IMAGENET_STD = np.array([0.229, 0.224, 0.225]) * 255
DEFAULT_CROP_RATIO = 224/256

@param('lr.lr')
@param('lr.step_ratio')
@param('lr.step_length')
@param('training.epochs')
def get_step_lr(epoch, lr, step_ratio, step_length, epochs):
    if epoch >= epochs:
        return 0

    num_steps = epoch // step_length
    return step_ratio**num_steps * lr

@param('lr.lr')
@param('training.epochs')
@param('lr.lr_peak_epoch')
def get_cyclic_lr(epoch, lr, epochs, lr_peak_epoch):
    xs = [0, lr_peak_epoch, epochs]
    ys = [1e-4 * lr, lr, 0]
    return np.interp([epoch], xs, ys)[0]

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

def batch_flip_lr(inputs):
    flip_mask = (torch.rand(len(inputs), device=inputs.device) < 0.5).view(-1, 1, 1, 1)
    return torch.where(flip_mask, inputs.flip(-1), inputs)

def calc_hash(n):
    return int(hashlib.md5(bytes(str(n), 'utf-8')).hexdigest()[-8:], 16)
seed = torch.randint(0, 372036854775808, size=()).item()
def batch_detflip_lr(inputs, indices, epoch):
    res = torch.tensor([calc_hash(i + seed) for i in indices.flatten().tolist()]).to(inputs.device)
    flip_mask = ((res + epoch) % 2 == 0).view(-1, 1, 1, 1)
    return torch.where(flip_mask, inputs.flip(-1), inputs)

class ImageNetTrainer:
    def __init__(self):
        self.all_params = get_current_config()

        self.uid = str(uuid4())

        self.train_loader = self.create_train_loader()
        self.val_loader = self.create_val_loader()
        self.model1, self.scaler1 = self.create_model_and_scaler()
        self.model2, self.scaler2 = self.create_model_and_scaler()
        self.model3, self.scaler3 = self.create_model_and_scaler()
        self.model4, self.scaler4 = self.create_model_and_scaler()
        self.optimizer1 = self.create_optimizer(self.model1)
        self.optimizer2 = self.create_optimizer(self.model2)
        self.optimizer3 = self.create_optimizer(self.model3)
        self.optimizer4 = self.create_optimizer(self.model4)
        self.initialize_logger()
        
    @param('lr.lr_schedule_type')
    def get_lr(self, epoch, lr_schedule_type):
        lr_schedules = {
            'cyclic': get_cyclic_lr,
            'step': get_step_lr
        }

        return lr_schedules[lr_schedule_type](epoch)

    # resolution tools
    @param('resolution.min_res')
    @param('resolution.max_res')
    @param('resolution.end_ramp')
    @param('resolution.start_ramp')
    def get_resolution(self, epoch, min_res, max_res, end_ramp, start_ramp):
        assert min_res <= max_res

        if epoch <= start_ramp:
            return min_res

        if epoch >= end_ramp:
            return max_res

        # otherwise, linearly interpolate to the nearest multiple of 32
        interp = np.interp([epoch], [start_ramp, end_ramp], [min_res, max_res])
        final_res = int(np.round(interp[0] / 32)) * 32
        return final_res

    @param('training.momentum')
    @param('training.weight_decay')
    @param('training.label_smoothing')
    def create_optimizer(self, model, momentum, weight_decay,
                         label_smoothing):

        # Only do weight decay on non-batchnorm parameters
        all_params = list(model.named_parameters())
        bn_params = [v for k, v in all_params if ('bn' in k)]
        other_params = [v for k, v in all_params if not ('bn' in k)]
        param_groups = [{'params': bn_params, 'weight_decay': 0.},
                        {'params': other_params, 'weight_decay': weight_decay}]

        self.loss = ch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        optimizer = ch.optim.SGD(param_groups, lr=1, momentum=momentum)
        return optimizer

    @param('data.train_dataset')
    @param('data.num_workers')
    @param('training.batch_size')
    @param('data.in_memory')
    def create_train_loader(self, train_dataset, num_workers, batch_size,
                            in_memory):
        assert Path(train_dataset).is_file()

        res = self.get_resolution(epoch=0)
        self.decoder = RandomResizedCropRGBImageDecoder((res, res))
        #self.decoder = RandomResizedCropRGBImageDecoder((res, res), scale=(0.5, 1.0), ratio=(1.0, 1.0))
        image_pipeline: List[Operation] = [
            self.decoder,
            #RandomHorizontalFlip(),
            ToTensor(),
            ToDevice(ch.device(0), non_blocking=True),
            ToTorchImage(),
            Convert(ch.float16),
            torchvision.transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]

        label_pipeline: List[Operation] = [
            IntDecoder(),
            ToTensor(),
            ToDevice(ch.device(0), non_blocking=True),
            Squeeze(),
        ]

        loader = Loader(train_dataset,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        #order=OrderOption.QUASI_RANDOM,
                        order=OrderOption.RANDOM,
                        os_cache=in_memory,
                        drop_last=True,
                        pipelines={
                            'image': image_pipeline,
                            'label': label_pipeline,
                        })

        return loader

    @param('data.val_dataset')
    @param('data.num_workers')
    @param('validation.batch_size')
    @param('validation.resolution')
    def create_val_loader(self, val_dataset, num_workers, batch_size,
                          resolution):
        assert Path(val_dataset).is_file()

        res_tuple = (resolution, resolution)
        cropper = CenterCropRGBImageDecoder(res_tuple, ratio=DEFAULT_CROP_RATIO)
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

    @param('training.epochs')
    def train(self, epochs):
        for epoch in range(epochs):
            res = self.get_resolution(epoch)
            self.decoder.output_size = (res, res)
            self.train_loop(epoch)

        extra_dict = {'epoch': epoch}
        start_val = time.time()
        stats = self.val_loop()
        val_time = time.time() - start_val
        self.log(dict({'current_lr': self.optimizer1.param_groups[0]['lr'], 'val_time': val_time},
                      **stats, **extra_dict))

        ch.save(self.model1.state_dict(), self.log_folder / 'final_weights1.pt')
        ch.save(self.model2.state_dict(), self.log_folder / 'final_weights2.pt')
        ch.save(self.model3.state_dict(), self.log_folder / 'final_weights3.pt')
        ch.save(self.model4.state_dict(), self.log_folder / 'final_weights4.pt')

    @param('model.arch')
    @param('training.use_blurpool')
    def create_model_and_scaler(self, arch, use_blurpool):
        scaler = GradScaler()
        model = getattr(models, arch)()
        def apply_blurpool(mod: ch.nn.Module):
            for (name, child) in mod.named_children():
                if isinstance(child, ch.nn.Conv2d) and (np.max(child.stride) > 1 and child.in_channels >= 16): 
                    setattr(mod, name, BlurPoolConv2d(child))
                else: apply_blurpool(child)
        if use_blurpool: apply_blurpool(model)

        model = model.to(memory_format=ch.channels_last)
        model = model.cuda()

        return model, scaler

    def train_loop(self, epoch):
        self.model1.train()
        self.model2.train()
        self.model3.train()
        self.model4.train()

        lr_start, lr_end = self.get_lr(epoch), self.get_lr(epoch + 1)
        iters = len(self.train_loader)
        lrs = np.interp(np.arange(iters), [0, iters], [lr_start, lr_end])

        iterator = tqdm(self.train_loader)
        for ix, (images, target, indices) in enumerate(iterator):

            #images1 = batch_flip_lr(images)
            #images2 = batch_detflip_lr(images, indices, epoch)
            images1 = images
            images2 = images

            for param_group in self.optimizer1.param_groups:
                param_group['lr'] = lrs[ix]
            for param_group in self.optimizer2.param_groups:
                param_group['lr'] = lrs[ix]
            for param_group in self.optimizer3.param_groups:
                param_group['lr'] = lrs[ix]
            for param_group in self.optimizer4.param_groups:
                param_group['lr'] = lrs[ix]

            self.optimizer1.zero_grad(set_to_none=True)
            self.optimizer2.zero_grad(set_to_none=True)
            self.optimizer3.zero_grad(set_to_none=True)
            self.optimizer4.zero_grad(set_to_none=True)
            with autocast():
                loss_train1 = self.loss(self.model1(images1), target)
                loss_train2 = self.loss(self.model2(images1), target)
                loss_train3 = self.loss(self.model3(images2), target)
                loss_train4 = self.loss(self.model4(images2), target)

            self.scaler1.scale(loss_train1).backward()
            self.scaler1.step(self.optimizer1)
            self.scaler1.update()
            self.scaler2.scale(loss_train2).backward()
            self.scaler2.step(self.optimizer2)
            self.scaler2.update()
            self.scaler3.scale(loss_train3).backward()
            self.scaler3.step(self.optimizer3)
            self.scaler3.update()
            self.scaler4.scale(loss_train4).backward()
            self.scaler4.step(self.optimizer4)
            self.scaler4.update()

    @param('validation.lr_tta')
    def val_loop(self, lr_tta):
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

    @param('logging.folder')
    def initialize_logger(self, folder):
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

        folder = (Path(folder) / str(self.uid)).absolute()
        folder.mkdir(parents=True)

        self.log_folder = folder
        self.start_time = time.time()

        print(f'=> Logging in {self.log_folder}')
        params = {'.'.join(k): self.all_params[k] for k in self.all_params.entries.keys()}
        with open(folder / 'params.json', 'w+') as handle:
            json.dump(params, handle)
        with open(folder / 'code.txt', 'w') as f:
            f.write(code)

    def log(self, content):
        print(f'=> Log: {content}')
        cur_time = time.time()
        with open(self.log_folder / 'log', 'a+') as fd:
            fd.write(json.dumps({
                'timestamp': cur_time,
                'relative_time': cur_time - self.start_time,
                **content
            }) + '\n')
            fd.flush()


def make_config(quiet=False):
    config = get_current_config()
    parser = ArgumentParser(description='Fast imagenet training')
    config.augment_argparse(parser)
    config.collect_argparse_args(parser)
    config.validate(mode='stderr')
    if not quiet:
        config.summary()

if __name__ == "__main__":
    with open(sys.argv[0]) as f:
        code = f.read()
    make_config()
    ImageNetTrainer().train()

