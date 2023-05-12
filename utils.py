import torch

def psnr(output, label):
    return 10. * torch.log10(1. / torch.mean((output - label) ** 2))
