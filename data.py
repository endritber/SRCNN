#!/usr/local/bin/python3
import h5py
import numpy as np
import torch


class SuperResolutionDataset(torch.utils.data.Dataset):
  def __init__(self, h5_file) -> None:
    super().__init__()
    self.h5_file = h5py.File(h5_file, 'r')
    
  def __len__(self) -> int:
    return len(self.h5_file['lr'])
  
  def __getitem__(self, index):
    return self.h5_file['lr'][index]/255., self.h5_file['hr'][index]/255.
    
if __name__ == '__main__':
  dataset = SuperResolutionDataset('datasets/T91.h5')
  mean_x = []
  mean_y = []
  for x in dataset:
    print(x[0].shape, x[1].shape)
    pass
