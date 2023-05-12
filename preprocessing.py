#!/usr/local/bin/python3
import numpy as np
import cv2
import patchify
from tqdm import tqdm

import os


PATH = 'datasets/T91/sub'
SIZE = 33
STRIDE = 1


def get_image_files(path):
  files = [os.path.join(path, image_name) for image_name in os.listdir(path)]
  return files


def main(upscale_factor):
  image_files = get_image_files(path='./datasets/T91/')
  for image in tqdm(image_files):
    if image.split('/')[-1] == 'sub':
      continue
    imread = cv2.imread(image, cv2.IMREAD_UNCHANGED).astype(np.float32)
    patches = patchify.patchify(np.array(imread), (SIZE, SIZE, 3), STRIDE)
    counter = 0
    for i in range(patches.shape[0]):
      for j in range(patches.shape[1]):
        patch = patches[i, j, 0, :, :, :]
        patch = cv2.cvtColor(patch, cv2.COLOR_RGB2BGR)
        W, H, _ = patch.shape
        HR_W, HR_H = (W//upscale_factor)*upscale_factor, (H//upscale_factor)*upscale_factor
        lr = cv2.resize(patch, (W // upscale_factor, H // upscale_factor), interpolation=cv2.INTER_CUBIC)
        hr = cv2.resize(lr, (HR_W, HR_H), interpolation=cv2.INTER_CUBIC)
        lr = cv2.resize(lr, (lr.shape[1] * upscale_factor, lr.shape[0] * upscale_factor), interpolation=cv2.INTER_CUBIC) 
        os.makedirs(os.path.join(PATH, 'lr_patches'), exist_ok=True)
        os.makedirs(os.path.join(PATH, 'hr_patches'), exist_ok=True)
        lr_path = f"{os.path.join(PATH, 'lr_patches', str(counter))}.jpg"
        hr_path = f"{os.path.join(PATH, 'hr_patches', str(counter))}.jpg"
        cv2.imwrite(lr_path, lr)
        cv2.imwrite(hr_path, hr)
        counter += 1
    
if __name__ == '__main__':
  os.makedirs(PATH, exist_ok=True)
  main(upscale_factor=2)
