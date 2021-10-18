from torch import nn
import torch
from torchaudio import transforms as torchAudioTransforms 
import barlowtwins.audioTransforms as localTransforms
import soundfile as sf
import librosa
import os
import uuid
import torchaudio
import random

class AudioTransformer(nn.Module):
  '''
  Creates transformers for the BarlowTwins network.
  An input audio tensor is transformed twice for each pathway in the network
  The transforms are specified in the .yaml file using data_transforms_1 and data_transforms_2
  Each is an array of tuples:
    [name, arguments]
  name is the transform name. Can be either a totch audio transform or one found in audioTransforms.py
  arguments is a dict of argName: value  that will be accepted by the transform
  '''
  def __init__(self, args, logger, supervised=False):
    super().__init__()
    self.args = args
    self.logger = logger

    self.logger.info("Creating sample transforms")
    if supervised:
      self.transform_1 = self.createTransforms(self.args.data_sup_transforms)
      self.transform_2 = self.createTransforms(self.args.data_sup_transforms)
    else:      
      self.transform_1 = self.createTransforms(self.args.data_transforms_1)
      self.transform_2 = self.createTransforms(self.args.data_transforms_2)

  def createTransforms(self, transformSpec):
    '''
    Creates a transform from a transform Spec
    transformSpec (list) transformName, dict of transform arguments
    '''
    transforms = []
    for t, kwargs in transformSpec:
      found = False
      for tCollection in [torchAudioTransforms, localTransforms]:
        if hasattr(tCollection, t):
          trans = getattr(tCollection, t)
          transforms.append(trans(**kwargs))
          found = True
          break

      if not found:
        self.logger.info("{} is not an Audio Transform - skipping".format(t))
    
    self.logger.info("Final Transforms: {} ".format(transforms))
    return nn.Sequential(*transforms)
  
  def write_wav_to_azure_with_librosa(self, x, file_name):
    # Rate is 16K
    try:
      sf.write(os.path.join("./outputs", 'lib_'+file_name+".wav"), x, 16000)
    except Exception as e:
      print("Failed writting librosa")
      print(e)

  def write_wav_to_azure_with_torchaudio(self, x, file_name):
  # Rate is 16K
  # write as binary file
    try:
      print("*** SHAPE {}".format(x.shape))
      path = os.path.join("./outputs", "torch_"+file_name+".wav")
      torchaudio.save(path, x, 16000)
    except Exception as e:
      print("**Failed writting torchaudio file to outputs")
      print(e)

  def __call__(self, x):
    # log 1% of the files. 
    store_audio = random.randint(1, 10) == 5

    file_name = str(uuid.uuid4())
    if store_audio:
      print("*** Before any transform")
      self.write_wav_to_azure_with_librosa(x, file_name+"_BT")
    y1 = self.transform_1(x)
    
    if store_audio:
      print("*** After first transform")
      self.write_wav_to_azure_with_torchaudio(y1, file_name+"_AT")
    y2 = self.transform_2(x)
    return y1, y2


class AudioTransformerBatch(AudioTransformer):
  '''
  Creates transformers for the BarlowTwins network that run on the data batch before it is
  fed to the backbone. Behaves same as AudioTransformer
  The transforms are specified in the .yaml file using data_batch_transforms_1 and data_batch_transforms_2
  Each is an array of tuples:
    [name, arguments]
  name is the transform name. Can be either a totch audio transform or one found in audioTransforms.py
  arguments is a dict of argName: value  that will be accepted by the transform
  '''
  def __init__(self, args, logger):
    super().__init__(args, logger)
    self.args = args
    self.logger = logger

    self.logger.info("Creating batch transforms")
    self.transform_1 = self.createTransforms(self.args.data_batch_transforms_1)
    self.transform_2 = self.createTransforms(self.args.data_batch_transforms_2)

  def __call__(self, x1, x2):
    y1 = self.transform_1(x1)
    y2 = self.transform_2(x2)
    return y1, y2

  def _convertToGpu(self, gpu, inTransforms):
    transforms = []
    for t in inTransforms:
      t.cuda(gpu)
      t = nn.SyncBatchNorm.convert_sync_batchnorm(t)
      t = torch.nn.parallel.DistributedDataParallel(t, device_ids=[gpu])
      transforms.append(t)

    self.logger.info("Final batch Transforms: {} ".format(transforms))
    return nn.Sequential(*transforms)

  def convertToGpu(self, gpu):
    self.transform_1 = self._convertToGpu(gpu, self.transform_1)
    self.transform_2 = self._convertToGpu(gpu, self.transform_2)
