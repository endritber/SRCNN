#!/usr/local/bin/python3
import argparse
import numpy as np
import cv2

import torch

import PIL.Image as Image
from model import SRCNN
from utils import psnr, convert_rgb_to_ycbcr, convert_ycbcr_to_rgb

def main(args, device):
  model = SRCNN().to(device)
  state_dict = model.state_dict()
  for n, p in torch.load(args.weights, map_location=lambda storage, loc: storage).items():
    if n in state_dict.keys():
      state_dict[n].copy_(p)
    else:
      raise KeyError(n)

  model.eval()
  
  image = cv2.imread(args.image_file, cv2.IMREAD_UNCHANGED).astype(np.float32)
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  H, W, _ = image.shape
  dim = ((W // args.upscale_factor) * args.upscale_factor, (H // args.upscale_factor) * args.upscale_factor)
  image = cv2.resize(image, dim, interpolation=cv2.INTER_CUBIC)
  cv2.imwrite(args.image_file.replace('.', '_bicubic_x{}.'.format(args.upscale_factor)), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
   
  ycbcr = convert_rgb_to_ycbcr(image)
  y = ycbcr/255.
  y = torch.from_numpy(y).to(device)
  y = y.unsqueeze(0)
  
  with torch.no_grad():
      preds = model(y).clamp(0.0, 1.0)

  PSNR = psnr(y.permute(0, 3, 1, 2), preds)
  print('PSNR: {:.2f}'.format(PSNR))

  preds = preds.mul(255.0).cpu().numpy().squeeze(0)
  output = np.clip(convert_ycbcr_to_rgb(preds.transpose(1, 2, 0)), 0.0, 255.0).astype(np.uint8)
  cv2.imwrite(args.image_file.replace('.', '_srcnn_x{}.'.format(args.upscale_factor)), cv2.cvtColor(output, cv2.COLOR_RGB2BGR))  

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--weights', type=str, required=True)
  parser.add_argument('--image-file', type=str, required=True)
  parser.add_argument('--upscale-factor', type=int, default=3)
  args = parser.parse_args()
  
  device = torch.device('mps') if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else torch.device('cpu')
  main(args, device)
  