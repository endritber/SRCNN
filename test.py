#!/usr/local/bin/python3
import argparse
import numpy as np

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
  
  #TODO: create it with opencv
  image = Image.open(args.image_file).convert('RGB')
  image_width = (image.width//args.upscale_factor)*args.upscale_factor 
  image_height = (image.height//args.upscale_factor)*args.upscale_factor 
  image = image.resize((image_width, image_height), resample=Image.BICUBIC)
  # image = image.resize((image.width // args.upscale_factor, image.height // args.upscale_factor), resample=Image.BICUBIC)
  # image = image.resize((image.width * args.upscale_factor, image.height * args.upscale_factor), resample=Image.BICUBIC)
  image.save(args.image_file.replace('.', '_bicubic_x{}.'.format(args.upscale_factor)))

  image = np.array(image).astype(np.float32)
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
  output = Image.fromarray(output)
  output.save(args.image_file.replace('.', '_srcnn_x{}.'.format(args.upscale_factor)))
  

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--weights', type=str, required=True)
  parser.add_argument('--image-file', type=str, required=True)
  parser.add_argument('--upscale-factor', type=int, default=3)
  args = parser.parse_args()
  
  device = torch.device('mps') if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else torch.device('cpu')
  main(args, device)
  