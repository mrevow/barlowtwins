from pathlib import Path
import json
import subprocess
import time

import numpy as np
from sklearn.metrics import recall_score, precision_score, accuracy_score

# from PIL import Image, ImageOps, ImageFilter
from torch import nn, optim
import torch

from barlowtwins.main import BarlowTwins
from barlowtwins.audioDataset import AudioDataset
from barlowtwins.audioTransformer import AudioTransformer, AudioTransformerBatch
from barlowtwins.metricsReporter import MetricsReporter

# from common.utils.pathUtils import createFullPathTree, ensureDir, savePickle, loadPickle
from common.utils.logger import CreateLogger

import logging
import azureml.core.authentication as authLog
import msrest.http_logger as http_logger
from msrest.universal_http.__init__ import _LOGGER as universalHttpLogger
from msrest.service_client import _LOGGER as serviceLogger
from urllib3.connectionpool import log as urllib3Logger


class MusicClassifier(object):
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


    def train(self):
        with CreateLogger(self.args, logger_type=self.args.logger_type) as logger:
            self.logger = logger
            self.loggerWorkaroundAll()
            self.args.checkpoint_dir = Path(self.args.output_dir)
            self.args.rank = 0

            train(self.args, logger)

    def eval(self):
        with CreateLogger(self.args, logger_type=self.args.logger_type) as logger:
            self.logger = logger
            self.loggerWorkaroundAll()
            self.args.checkpoint_dir = Path(self.args.output_dir)
            self.args.rank = 0

            eval_test_set(self.args, logger)


def train(args, logger):
    logger.info("Start Supervised training")
    if args.data_batch_transforms_1 is not None and args.data_batch_transforms_2 is not None:
        batchTransforms = AudioTransformerBatch(args, logger)
    else:
        batchTransforms = None

    model =  musicClassifier(args, logger, batchTransforms)
    logger.info('Loaded music classifier model')
    logger.debug(model)

    # automatically resume from checkpoint if it exists
    model = load_checkpoint(args, logger, model, args.checkpoint_name)

    # train on gpu if available
    dev, model = get_device(logger, model)

    # load datasets
    dataset_train = AudioDataset(args=args, logger=logger, mode='suptrain', transform=AudioTransformer(args, logger, supervised=True))
    dataset_val = AudioDataset(args=args, logger=logger, mode='supval', transform=AudioTransformer(args, logger,  supervised=True))
    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    loader_train = torch.utils.data.DataLoader(
        dataset_train, 
        batch_size=args.batch_size, 
        num_workers=args.workers, 
        sampler=sampler_train)
    loader_val = torch.utils.data.DataLoader(
        dataset_val, 
        batch_size=args.batch_size, 
        num_workers=args.workers,
        shuffle=False,
        drop_last=False,
        )

    # prepare for training
    optimizer = optim.Adam(model.parameters(), lr=args.music_classifier_learning_rate)
    criterion = nn.BCELoss()
    early_stopper = EarlyStopper(args)
    # To calculate binary Stats
    metricsReporter = MetricsReporter(args, logger)

    start_time = time.time()
    logger.info('Start musicClassifier training for {} epochs'.format(args.music_classifier_epochs))
    for epoch in range(0, args.music_classifier_epochs):
        predsAll = []
        labelsAll = []
        for step, ((x1, _), y1, _) in enumerate(loader_train, start=epoch * len(loader_train)):
            y1 = y1.type(torch.float).to(dev)
            x1 = x1.to(dev)

            optimizer.zero_grad()
            x1 = model.forward(x1)
            predsAll.extend(x1.detach().cpu().numpy())
            labelsAll.extend(y1.detach().cpu().numpy())

            loss = criterion(x1,y1)
            loss.backward()
            optimizer.step()

            # print/plot info
            logger.log_row(name='sup_loss', step=step, loss=loss.item())
            stats = dict(epoch=epoch, step=step,
                        loss=loss.item(),
                        time=int(time.time() - start_time))
            logger.info(json.dumps(stats))
            metrics = metricsReporter.calcBinaryStats(predsAll, labelsAll)
            metrics['loss'] = loss.item()
            metricsReporter.plotStats(metrics, ite=step, typ='Supervised train')

        # evaluate on validation set after each epoch
        results = evaluate(model, loader_val, dev, args, logger)
        metrics = metricsReporter.calcBinaryStats(predsAll, labelsAll)
        metrics.update(results)
        metricsReporter.plotStats(metrics, ite=epoch, typ='SupervisedValidation')
        logger.info('Supervised Val Epoch: {}, {}: {:0.3f}, best {}: {:0.3f}'.format(
            epoch+1, 
            args.music_classifier_early_stop_metric,
            float(results[args.music_classifier_early_stop_metric]), 
            args.music_classifier_early_stop_metric, 
            early_stopper.best_metric))

        # save checkpoint
        if float(results[args.music_classifier_early_stop_metric])>early_stopper.best_metric:
            statedict = model.module.state_dict() if (torch.cuda.device_count()>1) else model.state_dict()
            state = dict(epoch=epoch + 1, model=statedict,
                        optimizer=optimizer.state_dict())
            torch.save(state, args.checkpoint_dir / args.music_classifier_checkpoint_name)
            logger.info('Checkpoint saved {}'.format(args.checkpoint_dir / args.music_classifier_checkpoint_name))
            logger.log_value(name='val_best_' + args.music_classifier_early_stop_metric, value=float(results[args.music_classifier_early_stop_metric]))

        # stop early if validation accuracy does not improve
        stop_early = early_stopper.step(float(results[args.music_classifier_early_stop_metric]), epoch+1)
        if stop_early:
            logger.info("epoch: {}  step : {} Supervised training early stop ".format(epoch, step))
            # Plot PR curves for best 
            results = evaluate(model, loader_val, dev, args, logger, name='Best Validation', doPrPlot=True)
            return

        if epoch == 0:
            dataset_train.reportClipStats()
        dataset_train.resetCounters()

    results = evaluate(model, loader_val, dev, args, logger, name='Final Validation', doPrPlot=True)
    logger.info("Supervised training terminating after Epoch {} Step {}".format(epoch, step))


def eval_test_set(args, logger):
    logger.info("Start musicClassifier supervised evaluation")
    if args.data_batch_transforms_1 is not None and args.data_batch_transforms_2 is not None:
        batchTransforms = AudioTransformerBatch(args, logger)
    else:
        batchTransforms = None

    model =  musicClassifier(args, logger, batchTransforms)
    logger.info('Loaded music classifier model')
    logger.debug(model)

    # load checkpoint
    model = load_checkpoint(args, logger, model, args.music_classifier_checkpoint_name)

    # run on gpu if available
    dev, model = get_device(logger, model)

    # load datasets
    dataset_test = AudioDataset(args=args, logger=logger, mode='suptest', transform=AudioTransformer(args, logger, supervised=True))
    loader_test = torch.utils.data.DataLoader(
        dataset_test, 
        batch_size=args.batch_size, 
        num_workers=args.workers,
        shuffle=False,
        drop_last=False,
        )
    logger.info('Start musicClassifier evaluation on {} samples '.format(len(dataset_test)))
    results = evaluate(model, loader_test, dev, args, logger, name='Test', doPrPlot=True)
    resultPrint = { k: "{:.3f}".format(float(v)) for k,v in results.items()}
    logger.info("Test result N {} {}".format(len(dataset_test), resultPrint))

    metricsReporter = MetricsReporter(args, logger)
    metricsReporter.logValues(results, typ='SupervisedTest')
    dataset_test.reportClipStats()
    logger.info("Supervised testing completed")

def updateCheckPointKeys(chkPoint, subs="module."):
    keys = list(chkPoint.keys())
    for k in keys:
        if subs in k:
            kNew = k.replace(subs, "")
            chkPoint[kNew] = chkPoint[k]
            del chkPoint[k]

    return chkPoint

def load_checkpoint(args, logger, model, checkpoint_name):
    # automatically resume from checkpoint if it exists
    pth = str(args.checkpoint_dir / checkpoint_name)
    if (args.checkpoint_dir /checkpoint_name).is_file():
        ckpt = torch.load(args.checkpoint_dir / checkpoint_name,
                        map_location='cpu')

        ckpt['model'] = updateCheckPointKeys(ckpt['model'])
        [missing_keys, unexpected_keys ]  = model.load_state_dict(ckpt['model'], strict=False)

        for missed_key in missing_keys:
            if not (missed_key.startswith('backbone.fc') or missed_key.startswith('classHead')):
                raise ValueError('{} Found missing keys in checkpoint {}'.format(pth, missing_keys))
        for unexpected_key in unexpected_keys:
            if not ((unexpected_key.startswith('bn.')) or (unexpected_key.startswith('projector.'))):
                raise ValueError('{} Found unexpected keys in checkpoint {}'.format(pth, unexpected_keys))        
        logger.info('Checkpoint loaded from {}'.format(pth))
    else:
        logger.info('No checkpoint found at {}'.format(pth))
    return model


def get_device(logger, model):
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        model.to(dev)
        if torch.cuda.device_count()>1:
            model = torch.nn.parallel.DataParallel(model)
            logger.info('Train on gpu with data parallel on {} devices'.format(torch.cuda.device_count()))
        else:
            logger.info('Train on gpu without data parallel')
    else:
        dev = torch.device("cpu")
        logger.info('Train on cpu')
    return dev, model


def evaluate(model, loader, dev, args, logger, name='', doPrPlot=False):
        model.eval()
        with torch.no_grad():
            yy = [ [model(x1.to(dev)).cpu().numpy(), y1.cpu().numpy()] for ((x1, _), y1, _) in loader]
        model.train()
        metricsReporter = MetricsReporter(args, logger)
        yy = np.concatenate( yy, axis=1 )
        y_pred = yy[0,:].reshape(-1,1)
        y_true = yy[1,:].reshape(-1,1)
        accuracy = accuracy_score(y_true, y_pred > 0.5)
        recall = recall_score(y_true, y_pred > 0.5)
        precision = precision_score(y_true, y_pred > 0.5)
        results = {'accuracy': accuracy, 'recall': recall, 'precision': precision}
        results.update(metricsReporter.calcBinaryStats(y_pred, y_true, name, doPrPlot))
        return results


class EarlyStopper(object):          
    def __init__(self, args):
        self.patience = args.music_classifier_early_stop_patience
        self.metric = args.music_classifier_early_stop_metric
        self.args = args
        self.best_metric = -1e10
        self.best_epoch = 0
        self.cnt = -1
        
    def step(self, metric, epoch):
        if metric > self.best_metric:
            self.best_metric = metric
            self.best_epoch = epoch
            self.cnt = -1          
        self.cnt += 1 

        if self.cnt >= self.patience:
            stop_early = True
            return stop_early
        else:
            stop_early = False
            return stop_early


class musicClassifier(nn.Module):
    def __init__(self, args, logger, batchTransforms):
        super().__init__()
        self.args = args
        
        barlow_model = BarlowTwins(self.args, logger, batchTransforms=batchTransforms)
        self.backbone = barlow_model.backbone
        self.batchTransforms = batchTransforms
        self.classHead = nn.Linear(barlow_model.lastLayerSize, 1, bias=True)        

    def forward(self, x):
        if self.args.music_classifier_freeze_backbone:
            with torch.no_grad():
                x = self.backbone(x)
        else:
            x = self.backbone(x)

        x = self.classHead(x)
        x = torch.sigmoid(x).view(-1)
        return x

