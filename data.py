#!/usr/local/bin/python3
import os

import numpy as np
import matplotlib.pyplot as plt
import cv2
import patchify

import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as F


def show_resolution_image(image):
  plt.figure()
  plt.imshow(image)
  plt.show()
  

def get_patches(path):
  lr_patches = [patch for patch in os.listdir(os.path.join(path, 'lr_patches'))]
  hr_patches = [patch for patch in os.listdir(os.path.join(path, 'hr_patches'))]
  return lr_patches, hr_patches

class SuperResolutionDataset(Dataset):
  def __init__(self, path='datasets/T91/sub', upscale_factor=3) -> None:
    super().__init__()
    self.lr_patches, self.hr_patches = get_patches(path)
    self.upscale_factor = upscale_factor
    self.path = path
    
  def __len__(self) -> int:
    return len(self.lr_patches)
  
  def __getitem__(self, index):
    lr_patch, hr_patch = cv2.imread(os.path.join(self.path, 'lr_patches', self.lr_patches[index])), cv2.imread(os.path.join(self.path, 'hr_patches', self.hr_patches[index])) 
    return lr_patch.astype(np.float32)/255., hr_patch.astype(np.float32)/255.
    
if __name__ == '__main__':
  dataset = SuperResolutionDataset()
  mean_x = []
  mean_y = []
  for x in dataset:
    pass
  
  