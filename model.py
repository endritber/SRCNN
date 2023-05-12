#!/usr/local/bin/python3
import torch
import torch.nn as nn


class SRCNN(nn.Module):
  def __init__(self, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    self.encoder = nn.Sequential(
      nn.Conv2d(3, 64, kernel_size=(9, 9), padding=9//2),
      nn.ReLU(),
      nn.Conv2d(64, 32, kernel_size=(3, 3), padding=3//2),
      nn.ReLU()
    )
    
    self.output = nn.Sequential(
      nn.Conv2d(32, 3, kernel_size=(5, 5), padding=5//2),
    )
    
  def forward(self, x):
    x = x.permute(0, 3, 1, 2)
    x = self.encoder(x)
    x = self.output(x)
    return x
  