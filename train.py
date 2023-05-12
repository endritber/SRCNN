#!/usr/local/bin/python3
import wandb
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

WAN = os.getenv('WAN') != None


def trainer(args, device):
  
  if WAN:
    wandb.login()
    wandb.init(
    project="SRCNN",
    # Track hyperparameters and run metadata
    config={
        "learning_rate": args.learning_rate,
        "epochs": args.num_epochs,
    })
  
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
  
  
  print(f'Training SRCNN with batch_size: {args.batch_size}, learning_rate: {args.learning_rate}, upscale_factor: {args.upscale_factor}, validating: {args.validation_dataset}')
  for epoch in range(args.num_epochs):
    
    model.eval()
    for data in tqdm(validation_dataloader):
      inputs, labels = data
      inputs = inputs.to(device)
      labels = labels.to(device) 
      
      with torch.no_grad():
        # print(inputs.shape)
        outs = model(inputs)
      
    PSNR = psnr(outs.squeeze(1), labels.permute(0, 3, 1, 2))
    print(f'{args.validation_dataset} PSNR: {PSNR:.4f}')
    if WAN:
      wandb.log({f"{args.validation_dataset} - Validation PSNR": PSNR})
    
    timer = int(time.monotonic())
    model.train()
    # model.load_state_dict(torch.load(os.path.join(args.model_path, '63845_5.pth')))
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
        
        PSNR = psnr(outs.squeeze(1), labels.permute(0, 3, 1, 2))
        t.set_postfix(MSE=f'{cost.item():.4f}', PSNR=f'{PSNR:.4f}')
        t.update(len(inputs))
        
        if WAN:
          wandb.log({'MSE': cost.item(), "PSNR": PSNR})

    if epoch+1 % 30 == 0:
      torch.save(model.state_dict(), os.path.join(args.model_path, f'{timer}_{epoch+1}.pth'))
    
 

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--model-path', type=str, required=True)
  parser.add_argument('--train-dataset', type=str, required=True)
  parser.add_argument('--validation-dataset', type=str, required=True)
  parser.add_argument('--upscale-factor', type=int, default=3)
  parser.add_argument('--learning-rate', type=float, default=1e-3)
  parser.add_argument('--batch-size', type=int, default=16)
  parser.add_argument('--num-epochs', type=int, default=300)
  args = parser.parse_args()
  
  device = torch.device('mps') if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else torch.device('cpu')
  trainer(args, device)
        