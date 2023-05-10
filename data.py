#!/usr/bin/python3
import os

import numpy as np
import matplotlib.pyplot as plt
import cv2

from torch.utils.data import Dataset
from torchvision.transforms import functional as F


def show_resolution_image(image):
  plt.figure()
  plt.imshow(image)
  plt.show()
  
def get_image_file_names(path):
  files = [os.path.join(path, image_name) for image_name in os.listdir(path)]
  return files



class SuperResolutionDataset(Dataset):
  def __init__(self, path='datasets/T91') -> None:
    super().__init__()
    self.images = get_image_file_names(path)
    
  def __len__(self) -> int:
    return len(self.images)
  
  def __getitem__(self, index):
    image = cv2.imread(self.images[index], cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
    interpolated_image = cv2.resize(image, (240, 240), interpolation=cv2.INTER_CUBIC)
    # interpolated_image = bicubic(image, self.upscale_factor, 1)  
    print(interpolated_image.shape)
    tensor = F.to_tensor(interpolated_image)
    return tensor
  
    
if __name__ == '__main__':
  # show_resolution_image(cv2.imread(os.path.join('datasets/T91', 't1.png')))
  
  dataset = SuperResolutionDataset()
  mean_x = []
  mean_y = []
  for x in dataset:
    # print(x)
    mean_x.append(x.shape[0])
    mean_y.append(x.shape[1])
  
  print('x', sum(mean_x)/len(mean_x), 'y', sum(mean_y)/len(mean_y))
    
  
  
  