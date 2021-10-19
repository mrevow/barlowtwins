from torch import nn
import torch
from torchaudio import sox_effects
from typing import List
import librosa as lb
import numpy as np
import soundfile as sf
import os

def should_save_wav(count, stop_count):
  if count >= stop_count:
    return False, None
  
  return True, str(uuid.uuid4())


def save_file(wav, rate, file_name):
  try:
    sf.write(os.path.join("./outputs", file_name+".wav"), wav, rate)
  except Exception as e:
    print("Failed saving wav file: {}".format(e))

class Identity(nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self, x):
    return x

class ExpandDim(nn.Module):
  def __init__(self, dim):
    super().__init__()
    self.dim = dim

  def forward(self, x):
    return torch.unsqueeze(x, dim=self.dim)

''' Transformer example: https://pytorch.org/audio/stable/sox_effects.html
SOX effects explained: http://sox.sourceforge.net/sox.html#EFFECTS
List of PyTorch SOX audio effects: https://github.com/pytorch/audio/blob/8a347b62cf5c907d2676bdc983354834e500a282/test/torchaudio_unittest/assets/sox_effect_test_args.jsonl
'''
class SoxEffectTransform(nn.Module):
  def __init__(self, effects: List[List[str]]):
    super().__init__()
    self.effects = effects
    self.save_count = 0

  def forward(self, tensor: torch.Tensor, sample_rate: int = 16000):
    should_save, file_name = save_file(self.save_count, self.wav_files_to_save)
    if(should_save):
      save_file(tensor, sample_rate, "BT_"+file_name)
      self.save_count += 1
    
    waveform, sample_rate = sox_effects.apply_effects_tensor(tensor.unsqueeze(0), sample_rate, self.effects)

    if(should_save):
      save_file(waveform, sample_rate, "AT_"+file_name)
    return waveform.squeeze()

class SoxPitchTransform(nn.Module):
  def __init__(self, pRange):
    super().__init__()
    self.pitch_range = pRange

  def forward(self, tensor: torch.Tensor, sample_rate: int = 16000):
    pitch = np.random.randint(low=-self.pitch_range, high=self.pitch_range+1)
    effects = [ ['pitch', f'{pitch}'], ['rate', f'{sample_rate}'] ]

    should_save, file_name = save_file(self.save_count, self.wav_files_to_save)
    if(should_save):
      save_file(tensor, sample_rate, "BT_"+file_name)
      self.save_count += 1

    waveform, sample_rate = sox_effects.apply_effects_tensor(tensor.unsqueeze(0), sample_rate, effects)

    if(should_save):
      save_file(waveform, sample_rate, "AT_"+file_name)
      
    return waveform.squeeze()

class SoxGainTransform(nn.Module):
  def __init__(self, gain):
    super().__init__()
    self.gain = gain

  def forward(self, x: torch.Tensor, sample_rate: int = 16000):
    gain = np.random.randint(low=self.gain, high=1)
    effects = [ ['gain', '-n', f'{gain}'] ]

    should_save, file_name = save_file(self.save_count, self.wav_files_to_save)
    if(should_save):
      save_file(x, sample_rate, "BT_"+file_name)
      self.save_count += 1

    waveform, sample_rate = sox_effects.apply_effects_tensor(x.unsqueeze(0), sample_rate, effects)

    if(should_save):
      save_file(waveform, sample_rate, "AT_"+file_name)

    return waveform.squeeze()

class WhiteNoiseTransform(nn.Module):
  def __init__(self, snrLow, snrHigh):
    super().__init__()
    self.snr_low = snrLow
    self.snr_high = snrHigh

  def forward(self, x: torch.Tensor, sample_rate: int = 16000):
    snr_db = np.random.randint(low=self.snr_low, high=self.snr_high+1)
    noise = torch.randn_like(x)
    speech_power = x.norm(p=2)
    noise_power = noise.norm(p=2)
    scale = 1/(10 ** (snr_db/10)) * speech_power/noise_power 
    noise = scale * noise
    # snr_sanity_check = 10*math.log10(x.norm(p=2)/noise.norm(p=2))
    x = (x + noise)/2
    return x.squeeze()

class PolarityTransform(nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self, x: torch.Tensor, sample_rate: int = 16000):
    if np.random.randn()>0:
      x = -x
    return x.squeeze()

class LibrPitch(nn.Module):
  '''
  Built in 'eefect'  from librosa
  http://librosa.org/doc/main/effects.html
  '''
  def __init__(self, sr, pRange):
    super().__init__()
    self.sr = sr
    self.pitch_range = pRange

  def forward(self, x):
    shift = torch.randint(low=1, high=self.pitch_range, size=[1])
    x = lb.effects.pitch_shift(x.numpy(), sr=self.sr, n_steps=shift)
    return torch.tensor(x)

###################### Batch Transforms #################################
class NormalizeInputBatch(nn.Module):
  def __init__(self, chanCount, transpose=None, biasInit=0.0, weightinit=1.0):
    '''
    Perform batchNormalization on an input spectrogram
    chanCount (int) Channel count over which normalization is performed
    transpose:(list) Enable dimension transformation before and after normalization
      Example: Say expected input tensor is of order (N, 1, nFreq, nTime) then typicall want to transform
      to (N, nFreq, 1, nTime) before normalization. After normalization transform back
      For this case transform = [1, 2]
    '''
    super().__init__()

    self.bn = nn.BatchNorm2d(chanCount)
    self.transpose = transpose

    # Initialize
    self.bn.bias.data.fill_(biasInit)
    self.bn.weight.data.fill_(weightinit)

  def forward(self, x):
    if self.transpose is not None:
      x = x.transpose(*self.transpose)

    x = self.bn(x)
    if self.transpose is not None:
      x = x.transpose(*self.transpose)

    return x
