# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
# import argparse
import json
import math
import os
import random
# import signal
# import subprocess
# import sys
import time

# from PIL import Image, ImageOps, ImageFilter
from torch import nn, optim
import torch
import torchvision.models as torchvisionModels
import audioset_tagging_cnn.pytorch.models as audiosetModels

# import torchvision.transforms as transforms

from barlowtwins.audioTransformer import AudioTransformer, AudioTransformerBatch
from barlowtwins.audioDataset import AudioDataset
from barlowtwins.metricsReporter import MetricsReporter

from common.utils.pathUtils import createFullPathTree, ensureDir, savePickle, loadPickle
from common.utils.logger import CreateLogger

import logging
import azureml.core.authentication as authLog
import msrest.http_logger as http_logger
from msrest.universal_http.__init__ import _LOGGER as universalHttpLogger
from msrest.service_client import _LOGGER as serviceLogger
from urllib3.connectionpool import log as urllib3Logger

# parser = argparse.ArgumentParser(description='Barlow Twins Training')
# parser.add_argument('data', type=Path, metavar='DIR',
#                     help='path to dataset')
# parser.add_argument('--workers', default=8, type=int, metavar='N',
#                     help='number of data loader workers')
# parser.add_argument('--epochs', default=1000, type=int, metavar='N',
#                     help='number of total epochs to run')
# parser.add_argument('--batch-size', default=2048, type=int, metavar='N',
#                     help='mini-batch size')
# parser.add_argument('--learning-rate-weights', default=0.2, type=float, metavar='LR',
#                     help='base learning rate for weights')
# parser.add_argument('--learning-rate-biases', default=0.0048, type=float, metavar='LR',
#                     help='base learning rate for biases and batch norm parameters')
# parser.add_argument('--weight-decay', default=1e-6, type=float, metavar='W',
#                     help='weight decay')
# parser.add_argument('--lambd', default=0.0051, type=float, metavar='L',
#                     help='weight on off-diagonal terms')
# parser.add_argument('--projector', default='8192-8192-8192', type=str,
#                     metavar='MLP', help='projector MLP')
# parser.add_argument('--print-freq', default=100, type=int, metavar='N',
#                     help='print frequency')
# parser.add_argument('--checkpoint-dir', default='./checkpoint/', type=Path,
#                     metavar='DIR', help='path to checkpoint directory')


# def main():
#     args = parser.parse_args()
#     args.ngpus_per_node = torch.cuda.device_count()
#     if 'SLURM_JOB_ID' in os.environ:
#         # single-node and multi-node distributed training on SLURM cluster
#         # requeue job on SLURM preemption
#         signal.signal(signal.SIGUSR1, handle_sigusr1)
#         signal.signal(signal.SIGTERM, handle_sigterm)
#         # find a common host name on all nodes
#         # assume scontrol returns hosts in the same order on all nodes
#         cmd = 'scontrol show hostnames ' + os.getenv('SLURM_JOB_NODELIST')
#         stdout = subprocess.check_output(cmd.split())
#         host_name = stdout.decode().splitlines()[0]
#         args.rank = int(os.getenv('SLURM_NODEID')) * args.ngpus_per_node
#         args.world_size = int(os.getenv('SLURM_NNODES')) * args.ngpus_per_node
#         args.dist_url = f'tcp://{host_name}:58472'
#     else:
#         # single-node distributed training
#         args.rank = 0
#         args.dist_url = 'tcp://localhost:58472'
#         args.world_size = args.ngpus_per_node
#     torch.multiprocessing.spawn(main_worker, (args,), args.ngpus_per_node)

class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.logger = None

    def loggerWorkaround(self, azureLogger, name):
        '''
        Workaround around for azure loggers that by default spew debug logging that flood the output
        Simply set logging level to WARN
        '''
        before = azureLogger.getEffectiveLevel()
        azureLogger.setLevel(logging.WARNING)
        self.logger.info("{} logger workaround Loglevel Before {} After {}".format(
            name, before, azureLogger.getEffectiveLevel()))
    
    def loggerWorkaroundAll(self):

        # Workarounds for issue in S/C cluster that gets a wierd loglevel
        self.loggerWorkaround(authLog.module_logger, 'AzureAuthority')
        self.loggerWorkaround(http_logger._LOGGER, "http logger")
        self.loggerWorkaround(logging.getLogger("azureml"), "azureml logger")
        universalHttpLogger.debug("universalHttpLogger Debug Configuring requets Before")
        universalHttpLogger.info("universalHttpLogger INFO Configuring requets Before")
        self.loggerWorkaround(universalHttpLogger, "universal logger")
        universalHttpLogger.debug("universalHttpLogger DEBUG Configuring requets Before")
        self.loggerWorkaround(serviceLogger, "serviceLogger")
        self.loggerWorkaround(urllib3Logger, "urllib3 logger")


    def run(self, gpu):
        with CreateLogger(self.args, logger_type=self.args.logger_type) as logger:
            self.logger = logger
            self.loggerWorkaroundAll()
            self.args.checkpoint_dir = Path(self.args.output_dir)

            main_worker(self.args, logger, gpu)


def plotStats(args, logger, stats, ite, typ):
    if args.rank == 0 and stats is not None:
        for k, v in stats.items():
            try:
                val = float(v)
                nme = "{}_{}".format(typ, k)
                maxx = args.data_plot_max_limits.get(k, None)
                val = min(val, maxx) if maxx is not None else val
                minn = args.data_plot_min_limits.get(k, None)
                val = max(val, minn) if minn is not None else val
                logger.log_row(name=nme, iter=ite, val=val, description="{} master proc".format(nme))
            except ValueError:
                pass

def chooseBackbone(backbone_model, backbone_kwargs, logger):
    '''
    return a pre-configured pytorch model
    '''
    model = None

    if hasattr(audiosetModels, backbone_model):
        model = getattr(audiosetModels, backbone_model)(logger, **backbone_kwargs)
        lastLayerName = list(model.named_modules())[-1][0]
        lastLayersize = getattr(model, lastLayerName).out_features            
        logger.info("Found {} in audiosetModels".format(backbone_model))

    # Check if it is a torchvision model - they require some fiddling
    if model is None and  hasattr(torchvisionModels, backbone_model):
        model = getattr(torchvisionModels, backbone_model)(**backbone_kwargs)
        lastLayerName = list(model.named_modules())[-1][0]
        lastLayersize = getattr(model, lastLayerName).in_features    
        # Vision model have uniform structure just need to fiddle it for audio
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        nn.init.kaiming_normal_(model.conv1.weight, mode='fan_out', nonlinearity='relu')
        model.fc = nn.Identity()
        logger.info("Found {} in torchvisionModels".format(backbone_model))

    assert model is not None, "Cannot find an implementation for backbone model {}".format(backbone_model)
    return model, lastLayersize, lastLayerName

def main_worker(args, logger, gpu):
    logger.info("Starting unsupervised training on Device {}".format(gpu))

    torch.backends.cudnn.benchmark = True
    if args.data_batch_transforms_1 is not None and args.data_batch_transforms_2 is not None:
        batchTransforms = AudioTransformerBatch(args, logger)
    else:
        batchTransforms = None

    if torch.cuda.is_available():
        model = BarlowTwins(args, logger, batchTransforms).cuda(args.rank)
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        batchTransforms.convertToGpu(args.rank)
    else:
        model =  BarlowTwins(args, logger, batchTransforms)

    param_weights = []
    param_biases = []
    for param in model.parameters():
        if param.ndim == 1:
            param_biases.append(param)
        else:
            param_weights.append(param)
    parameters = [{'params': param_weights}, {'params': param_biases}]
    if torch.cuda.is_available():
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu],)
    optimizer = LARS(parameters, lr=0, weight_decay=args.weight_decay,
                    weight_decay_filter=True,
                    lars_adaptation_filter=True)

    early_stopper = EarlyStopper(args)

    # automatically resume from checkpoint if it exists
    if (args.checkpoint_dir / args.checkpoint_name ).is_file():
        ckpt = torch.load(args.checkpoint_dir /  args.checkpoint_name,
                        map_location='cpu')
        start_epoch = ckpt['epoch']
        if torch.cuda.is_available():
             model.module.load_state_dict(ckpt['model'])
        else:
            model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        logger.info("Unsupervised loaded checkpoint {}".format(args.checkpoint_dir /  args.checkpoint_name))
    else:
        start_epoch = 0

    # dataset = torchvision.datasets.ImageFolder(args.data / 'train', Transform())
    dataset = AudioDataset(args=args, logger=logger, mode='train', transform=AudioTransformer(args, logger))
    dataset_val = AudioDataset(args=args, logger=logger, mode='val', transform=AudioTransformer(args, logger))
    
    sampler = torch.utils.data.distributed.DistributedSampler(dataset) if torch.cuda.is_available() \
        else torch.utils.data.RandomSampler(dataset)
    sampler_val = torch.utils.data.distributed.DistributedSampler(dataset_val, shuffle=False) if torch.cuda.is_available() \
        else torch.utils.data.SequentialSampler(dataset_val)

    assert args.batch_size % args.world_size == 0
    per_device_batch_size = args.batch_size // args.world_size
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=per_device_batch_size, num_workers=args.workers,
        pin_memory=True, sampler=sampler)
    loader_val = torch.utils.data.DataLoader(
        dataset, batch_size=per_device_batch_size, num_workers=args.workers,
        pin_memory=True, sampler=sampler_val)       

    def _calc_val_loss_and_save_checkpoint():
        # model.eval() # uncommented because it makes val_loss jump at the start of training (due to normalization layers presumably)
        val_loss = 0
        with torch.no_grad():      
            for (y1, y2), _,idxs in loader_val:
                if torch.cuda.is_available():
                    y1 = y1.cuda(gpu, non_blocking=True)
                    y2 = y2.cuda(gpu, non_blocking=True)
                val_loss += model.forward(y1, y2)[0].item()
        val_loss = val_loss / len(loader_val)

        if step == 0:
            dataset_val.reportClipStats()
        dataset_val.resetCounters()
        # model.train()

        if args.rank == 0:
            stats = dict(epoch=epoch, step=step,
                        val_loss=val_loss,
                        best_val_loss=early_stopper.best_val_loss,
                        time=int(time.time() - start_time))
            if step % args.print_freq == 0:    
                logger.info(json.dumps(stats))
            if step % args.plot_freq == 0:
                stats = dict(us_val_loss=val_loss)    
                plotStats(args, logger, stats, step, 'TrainIter')

            if val_loss<early_stopper.best_val_loss:
                # save checkpoint
                statedict = model.module.state_dict() if torch.cuda.is_available() else model.state_dict()
                state = dict(epoch=epoch + 1, model=statedict,
                            optimizer=optimizer.state_dict())
                torch.save(state, args.checkpoint_dir / args.checkpoint_name)    
                logger.info('Unsupervised Checkpoint saved {}'.format(args.checkpoint_dir / args.checkpoint_name))                    

        # stop early if validation accuracy does not improve
        stop_early = early_stopper.step(val_loss, step)
        return stop_early

    start_time = time.time()
    scaler = torch.cuda.amp.GradScaler()
    step = epoch = 0
    metricsReporter = MetricsReporter(args, logger)
    
    for epoch in range(start_epoch, args.epochs):
        if torch.cuda.is_available():
            sampler.set_epoch(epoch)
        for step, ((y1, y2), _,idxs) in enumerate(loader, start=epoch * len(loader)):
            if torch.cuda.is_available():
                y1 = y1.cuda(gpu, non_blocking=True)
                y2 = y2.cuda(gpu, non_blocking=True)
            adjust_learning_rate(args, optimizer, loader, step)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                loss, on_diag, off_diag = model.forward(y1, y2)

            isNan = torch.isnan(loss)
            assert not torch.all(isNan), "Hit Nan at step {} idxs {}".format(step, idxs)                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if step %args.plot_freq == 0:
                if args.rank == 0:
                    stats = dict(
                                lr_weights=optimizer.param_groups[0]['lr'],
                                lr_biases=optimizer.param_groups[1]['lr'],
                                us_loss=loss.item(),
                                us_on_diag=on_diag.item(),
                                us_off_diag=off_diag.item()
                                )
                    metricsReporter.plotStats(stats, step, 'UnsupervisedTrain')

            if step % args.print_freq == 0:
                if args.rank == 0:
                    stats = dict(epoch=epoch, step=step,
                                lr_weights=optimizer.param_groups[0]['lr'],
                                lr_biases=optimizer.param_groups[1]['lr'],
                                us_loss=loss.item(),
                                us_on_diag=on_diag.item(),
                                us_off_diag=off_diag.item(),
                                time=int(time.time() - start_time))
                    logger.info(json.dumps(stats))

            if step % args.val_freq == 0:
                stop_early = _calc_val_loss_and_save_checkpoint()
                if stop_early:
                    if args.rank == 0:
                        logger.info("epoch: {}  step : {} Unsupervised training early stop ".format(epoch, step))
                        stats = dict(epoch=epoch, step=step,
                                    lr_weights=optimizer.param_groups[0]['lr'],
                                    lr_biases=optimizer.param_groups[1]['lr'],
                                    us_loss=loss.item(),
                                    us_on_diag=on_diag.item(),
                                    us_off_diag=off_diag.item(),
                                    time=int(time.time() - start_time))
                        logger.info(json.dumps(stats))
                        stats = dict(
                                    lr_weights=optimizer.param_groups[0]['lr'],
                                    lr_biases=optimizer.param_groups[1]['lr'],
                                    us_loss=loss.item(),
                                    us_on_diag=on_diag.item(),
                                    us_off_diag=off_diag.item()
                                    )
                        metricsReporter.plotStats(stats, step, 'UnsupervisedTrain')

                    return
        if epoch == 0:
            dataset.reportClipStats()
        dataset.resetCounters()
    # calc final val_loss and save checkpoint if better than last saved checkpoint
    if step > 0:
        _calc_val_loss_and_save_checkpoint()
    
    if args.rank == 0:
        # save final model
        statedict = model.module.backbone.state_dict() if torch.cuda.is_available() else model.backbone.state_dict()
        chk = args.backbone_model + '.pth'
        torch.save(statedict,
                args.checkpoint_dir / chk)
        logger.info("Unsupervised model save {}".format(args.checkpoint_dir / chk))


class EarlyStopper(object):          
    def __init__(self, args):
        self.patience = args.early_stop_patience
        self.args = args
        self.best_val_loss = 1e100
        self.best_step = 0
        self.cnt = -1

    def step(self, val_loss, epoch):
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_step = epoch
            self.cnt = -1          
        self.cnt += 1 

        if self.cnt >= self.patience:
            stop_early = True
            return stop_early
        else:
            stop_early = False
            return stop_early


def adjust_learning_rate(args, optimizer, loader, step):
    max_steps = args.epochs * len(loader)
    warmup_steps = 10 * len(loader)
    base_lr = args.batch_size / 256
    if step < warmup_steps:
        lr = base_lr * step / warmup_steps
    else:
        step -= warmup_steps
        max_steps -= warmup_steps
        q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
        end_lr = base_lr * 0.001
        lr = base_lr * q + end_lr * (1 - q)
    optimizer.param_groups[0]['lr'] = lr * args.learning_rate_weights
    optimizer.param_groups[1]['lr'] = lr * args.learning_rate_biases


def handle_sigusr1(signum, frame):
    os.system(f'scontrol requeue {os.getenv("SLURM_JOB_ID")}')
    exit()


def handle_sigterm(signum, frame):
    pass


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class BarlowTwins(nn.Module):
    def __init__(self, args, logger, batchTransforms=None):
        super().__init__()
        self.args = args
        self.logger = logger
        backbone, lastLayerSize, lastLayerName  = chooseBackbone(self.args.backbone_model, self.args.backbone_kwargs[0], self.logger)
        self.backbone = backbone
        self.lastLayerSize = lastLayerSize
        self.lastLayerName = lastLayerName
        
        # self.backbone = torchvision.models.resnet50(zero_init_residual=True)
        
        # Update the native resNet for audio (single input channel)
        # Create an Audio input for resNet
        # self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # nn.init.kaiming_normal_(self.backbone.conv1.weight, mode='fan_out', nonlinearity='relu')
        # self.backbone.fc = nn.Identity()

        # projector
        sizes = [lastLayerSize] + list(map(int, args.projector.split('-')))
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projector = nn.Sequential(*layers)

        # normalization layer for the representations z1 and z2
        self.bn = nn.BatchNorm1d(sizes[-1], affine=False)

        self.logger.info("Created BarlowTwins using backbone {}".format(self.args.backbone_model))
        self.logger.debug(self)
        self.batchTransforms = batchTransforms

    def forward(self, y1, y2):
        if self.batchTransforms is not None:
            y1, y2 = self.batchTransforms(y1, y2)
        
        y1 = self.backbone(y1)
        z1 = self.projector(y1)
        z2 = self.projector(self.backbone(y2))

        # empirical cross-correlation matrix
        c = self.bn(z1).T @ self.bn(z2)

        # sum the cross-correlation matrix between all gpus
        c.div_(self.args.batch_size)
        if torch.cuda.is_available():
            torch.distributed.all_reduce(c)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.args.lambd * off_diag
        return loss, on_diag, off_diag


class LARS(optim.Optimizer):
    def __init__(self, params, lr, weight_decay=0, momentum=0.9, eta=0.001,
                 weight_decay_filter=False, lars_adaptation_filter=False):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum,
                        eta=eta, weight_decay_filter=weight_decay_filter,
                        lars_adaptation_filter=lars_adaptation_filter)
        super().__init__(params, defaults)


    def exclude_bias_and_norm(self, p):
        return p.ndim == 1

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g['params']:
                dp = p.grad

                if dp is None:
                    continue

                if not g['weight_decay_filter'] or not self.exclude_bias_and_norm(p):
                    dp = dp.add(p, alpha=g['weight_decay'])

                if not g['lars_adaptation_filter'] or not self.exclude_bias_and_norm(p):
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(param_norm > 0.,
                                    torch.where(update_norm > 0,
                                                (g['eta'] * param_norm / update_norm), one), one)
                    dp = dp.mul(q)

                param_state = self.state[p]
                if 'mu' not in param_state:
                    param_state['mu'] = torch.zeros_like(p)
                mu = param_state['mu']
                mu.mul_(g['momentum']).add_(dp)

                p.add_(mu, alpha=-g['lr'])



class GaussianBlur(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            sigma = random.random() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img


class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


class Transform:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(224, interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=1.0),
            Solarization(p=0.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.transform_prime = transforms.Compose([
            transforms.RandomResizedCrop(224, interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=0.1),
            Solarization(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, x):
        y1 = self.transform(x)
        y2 = self.transform_prime(x)
        return y1, y2


# if __name__ == '__main__':
#     main()
