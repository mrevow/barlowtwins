from torch import nn
import torch
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

class SoxEffectTransform(torch.nn.Module):
  effects: List[List[str]]

  def __init__(self, effects: List[List[str]]):
    super().__init__()
    self.effects = effects

  def forward(self, tensor: torch.Tensor, sample_rate: int):
    return sox_effects.apply_effects_tensor(tensor, sample_rate, self.effects)
