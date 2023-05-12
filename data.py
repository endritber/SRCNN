#!/usr/local/bin/python3
import os
import h5py

import numpy as np

from torch.utils.data import Dataset
from torchvision.transforms import functional as F


class SuperResolutionDataset(Dataset):
  def __init__(self, h5_file) -> None:
    super().__init__()
    self.h5_file = h5py.File(h5_file, 'r')
    
  def __len__(self) -> int:
    return len(self.h5_file['lr'])
  
  def __getitem__(self, index):
    return np.expand_dims(self.h5_file['lr'][index]/255., 0), np.expand_dims(self.h5_file['hr'][index]/255., 0)
    
if __name__ == '__main__':
  dataset = SuperResolutionDataset('datasets/T91.h5')
  mean_x = []
  mean_y = []
  for x in dataset:
    print(x[0].shape, x[1].shape)
    pass
  
  