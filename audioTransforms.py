from torch import nn
import torch
from torchaudio import sox_effects
from typing import List
import librosa as lb

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

  def forward(self, tensor: torch.Tensor, sample_rate: int = 16000):
    waveform, sample_rate = sox_effects.apply_effects_tensor(tensor.unsqueeze(0), sample_rate, self.effects)
    #return waveform.squeeze().numpy()
    return waveform.squeeze()


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
