#!/usr/local/bin/python3
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model import SRCNN
from data import SuperResolutionDataset
from utils import psnr

from tqdm import tqdm
import argparse
import os
import time


def trainer(args, device):
  model = SRCNN().to(device)
  loss = nn.MSELoss()
  optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)
  
  train_dataset = SuperResolutionDataset(f'./datasets/{args.train_dataset}_{args.upscale_factor}.h5')
  train_dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=args.batch_size,
    shuffle=True
  )
  
  validation_dataset = SuperResolutionDataset(f'./datasets/{args.validation_dataset}_{args.upscale_factor}.h5', validation=True)
  validation_dataloader = DataLoader(
    dataset=validation_dataset,
    batch_size=1,
  )
  
  
  print(f'Training SRCNN with batch_size {args.batch_size}, learning_rate {args.learning_rate} upscale_factor {args.upscale_factor}')
  for epoch in range(args.num_epochs):
    print(f'Validating on {args.validation_dataset}...')
    model.eval()
    for data in tqdm(validation_dataloader):
      inputs, labels = data
      inputs = inputs.to(device)
      labels = labels.to(device) 
      
      with torch.no_grad():
        # print(inputs.shape)
        outs = model(inputs)
      
    print(f'validation PSNR: {psnr(outs.squeeze(1), labels.permute(0, 3, 1, 2)):.4f}')
    
    timer = int(time.monotonic())
    model.train()
    # model.load_state_dict(torch.load(os.path.join(args.model_path, '')))
    with tqdm(total=len(train_dataset) - len(train_dataset) % 128) as t:
      t.set_description(f'epoch: {epoch+1}/{args.num_epochs}')
      
      for data in train_dataloader:
        optimizer.zero_grad()
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        outs = model(inputs)
        cost = loss(outs.squeeze(1), labels.permute(0, 3, 1, 2))
        
        cost.backward()
        optimizer.step()
        
        t.set_postfix(MSE=f'{cost.item():.4f}', PSNR=f'{psnr(outs.squeeze(1), labels.permute(0, 3, 1, 2)):.4f}')
        t.update(len(inputs))

    torch.save(model.state_dict(), os.path.join(args.model_path, f'{timer}_{epoch}.pth'))
    
 

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--model-path', type=str, required=True)
  parser.add_argument('--train-dataset', type=str, required=True)
  parser.add_argument('--validation-dataset', type=str, required=True)
  parser.add_argument('--upscale_factor', type=int, default=3)
  parser.add_argument('--learning-rate', type=float, default=1e-3)
  parser.add_argument('--batch-size', type=int, default=16)
  parser.add_argument('--num-epochs', type=int, default=300)
  args = parser.parse_args()
  
  device = torch.device('mps') if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else torch.device('cpu')
  trainer(args, device)
        