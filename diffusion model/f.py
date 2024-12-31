from inspect import isfunction
import torch.nn as nn
from einops.layers.torch import Rearrange
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torchvision.transforms import Compose, Lambda, ToPILImage
import numpy as np



def exists(x):
    return x is not None

def default(val,d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def Upsample(dim, dim_out=None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding=1),
    )

def Downsample(dim, dim_out=None):
    return nn.Sequential(
        Rearrange("b c (h p1) (w p2) -> b (c p1 p2) h w", p1=2, p2=2),
        nn.Conv2d(dim * 4, default(dim_out, dim), 1),
    )

def linear_beta_schedule(time_steps):

    min_beta = 0.0001
    max_beta = 0.02

    return torch.linspace(min_beta, max_beta, time_steps)

def constant_beta_schedule(time_steps):

    beta = 0.001

    return torch.tensotr((beta,)*time_steps)

def quadratic_beta_schedule(time_steps):

    min_beta = 0.0001
    max_beta = 0.02

    return torch.linspace(min_beta**0.5, max_beta**0.5, time_steps)**2

class Alpha():
    def __init__(self, beta_schedule):
        self.alphas = 1. - beta_schedule
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.betas = beta_schedule
        self.sqrt_alphas = torch.sqrt(self.alphas_cumprod)
        self.one_minus_alphas = 1 - self.alphas_cumprod
        self.one_minus_alphas_prev = 1 - F.pad(self.alphas_cumprod[:-1], (1,0), value=1.0)
        self.posterior_variance = beta_schedule * self.one_minus_alphas_prev / self.one_minus_alphas

    def sqrt_alphas_extract(self, t):
        batch_size = t.shape[0]
        alphas = self.sqrt_alphas.gather(-1, t.cpu()-1)

        return torch.reshape(alphas, (batch_size, 1, 1, 1)).to(t.device)


    def sqrt_one_minus_alphas_extract(self, t):
        batch_size = t.shape[0]
        alphas = self.one_minus_alphas.gather(-1, t.cpu()-1)

        return torch.reshape(torch.sqrt(alphas), (batch_size, 1, 1, 1)).to(t.device)

    def alphas_extract(self,t):
        batch_size = t.shape[0]
        alphas = self.alphas.gather(-1, t.cpu()-1)

        return torch.reshape(alphas, (batch_size, 1, 1, 1)).to(t.device)

    def posterior_variance_extract(self,t):
        batch_size = t.shape[0]
        alphas = self.posterior_variance.gather(-1, t.cpu()-1)

        return torch.reshape(alphas, (batch_size, 1, 1, 1)).to(t.device)

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num & divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
        return arr

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def image_show(image_file):

    reverse_transform = Compose([
        Lambda(lambda t: (t + 1).clamp(-1, 1) / 2),  # [-1,1] -> [0,1], clamp로 안전 처리
        Lambda(lambda t: (t * 255).clamp(0, 255)),  # [0,1] -> [0,255]
        Lambda(lambda t: t.permute(1, 2, 0)),  # (C,H,W)->(H,W,C)
        Lambda(lambda t: t.numpy().astype(np.uint8)),
        ToPILImage(),
    ])

    loaded_image = torch.load(image_file).to('cpu')
    random_index = 8
    image = reverse_transform(loaded_image[random_index])
    image.show()
