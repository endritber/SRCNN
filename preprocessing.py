#!/usr/local/bin/python3
import numpy as np
import cv2
import patchify
from tqdm import tqdm

import os
import argparse
import h5py


PATH = './datasets/'

def get_files(path):
  files = [os.path.join(path, image_name) for image_name in os.listdir(path)]
  return files

def main(args):
  h5_file = h5py.File(os.path.join(PATH, args.output_path)+'.h5', 'w')
  
  lr_patches, hr_patches = [], []
  for file in tqdm(get_files(path=os.path.join(PATH, args.dataset))):
    src = cv2.imread(file, cv2.IMREAD_UNCHANGED).astype(np.float32)
    rgb_image = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
    H, W, _ = rgb_image.shape
    lr = cv2.resize(rgb_image, (W // args.upscale_factor, H // args.upscale_factor), interpolation=cv2.INTER_CUBIC)
    dim = ((W // args.upscale_factor) * args.upscale_factor, (H // args.upscale_factor) * args.upscale_factor)
    hr = cv2.resize(lr, dim, interpolation=cv2.INTER_CUBIC)
    lr = cv2.resize(hr, (lr.shape[1] * args.upscale_factor, lr.shape[0] * args.upscale_factor), interpolation=cv2.INTER_CUBIC)
      
    #create patches
    for i in range(0, lr.shape[0]-args.patch_size+1, args.stride):
      for j in range(0, lr.shape[1]-args.patch_size+1, args.stride):
        lr_patches.append(lr[i:i+args.patch_size, j:j+args.patch_size])
        hr_patches.append(hr[i:i+args.patch_size, j:j+args.patch_size])
        
  h5_file.create_dataset('lr', data=np.array(lr_patches).astype(np.float32))
  h5_file.create_dataset('hr', data=np.array(hr_patches).astype(np.float32))
  h5_file.close()
   
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--dataset', type=str, required=True)
  parser.add_argument('--output-path', type=str, required=True)
  parser.add_argument('--patch-size', type=int, default=33)
  parser.add_argument('--stride', type=int, default=14)
  parser.add_argument('--upscale-factor', type=int, default=3)
  parser.add_argument('--validation', type=bool, default=False)
  parser.add_argument('--testing', type=bool, default=False)
  args = parser.parse_args()
  
  main(args)
