#!/usr/local/bin/python3
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model import SRCNN
from data import SuperResolutionDataset
from tqdm import tqdm


def calc_psnr(img1, img2):
    return 10. * torch.log10(1. / torch.mean((img1 - img2) ** 2))


if __name__ == '__main__':
  mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
  if mps_available:
    device = torch.device('mps')
  else:
    device = torch.device('cpu')
    
  model = SRCNN().to(device)
  criterion = nn.MSELoss()
  optimizer = optim.SGD(model.parameters(), lr=1e-2)
  
  train_dataset = SuperResolutionDataset('./datasets/T91.h5')
  train_dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=128,
    shuffle=True,
    # num_workers=4,
    # pin_memory=True
  )
  
  num_epochs=1000
  for epoch in range(num_epochs):
    model.train()
    with tqdm(total=len(train_dataset) - len(train_dataset) % 128) as t:
      t.set_description(f'epoch: {epoch+1}/{num_epochs}')
      
      for data in train_dataloader:
        inputs, labels = data
        
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        outs = model(inputs)
        
        # print(outs.shape, labels.permute(0, 3, 1, 2).shape, outs.squeeze(1).shape)
        loss = criterion(outs.squeeze(1), labels.squeeze(1).permute(0, 3, 1, 2))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        t.set_postfix(loss=f'{loss.item():.4f}', psnr=f'{calc_psnr(outs.squeeze(1), labels.squeeze(1).permute(0, 3, 1, 2)):.4f}')
        t.update(len(inputs))
        