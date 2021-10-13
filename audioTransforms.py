from torch import nn
import torch
from torchaudio import sox_effects
from typing import List

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
