import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import os
#from google.colab import drive
from model import diffusion_loss, UNet, EMA, sample
from f import image_show

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2470, 0.2435, 0.2616)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(cifar10_mean, cifar10_std)
])

train_dataset = datasets.CIFAR10(
    root='./data',
    train=True,
    transform=transform,
    download=True,
)

test_dataset = datasets.CIFAR10(
    root='./data',
    train=False,  # 테스트용 데이터
    transform=transform,
    download = True,
)

# 데이터 로더 생성
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, drop_last=True)


device = 'cuda' if torch.cuda.is_available() else "cpu"

model = UNet()
model.to(device)
ema = EMA(model)
ema.to(device)

#drive.mount('/content/drive')

epochs = 300

optimizer = Adam(model.parameters(), lr=2e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

best_loss = float('inf')

for epoch in range(epochs):
    for step, data in tqdm(enumerate(train_loader),desc = 'training', total = len(train_loader)):

        images, _ = data
        images = images.to(device)
        optimizer.zero_grad()

        batch_size = images.shape[0]

        t = torch.randint(1, 1001, (batch_size,), device=device).long().to(device)

        loss = diffusion_loss(diffusion_model = model, x = images, t=t, loss_type='l2')

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_loss_file = 'mini_batch_loss.pth'
            torch.save(model.state_dict(), best_loss_file)

        loss.backward()
        optimizer.step()

        ema.update()

    print("\nLoss\n:", loss.item())
    ema_shadow_file = "ema_shadow.pth"
    torch.save(ema.shadow, ema_shadow_file)

    if epoch % 10 == 9:

        file_path = '/content/drive/MyDrive/'

        if os.path.exists(file_path+ema_shadow_file):
            os.remove(file_path+ema_shadow_file)
        if os.path.exists(file_path+best_loss_file):
            os.remove(file_path+best_loss_file)

        #!cp ema_shadow.pth /content/drive/MyDrive/
        #!cp mini_batch_loss.pth /content/drive/MyDrive/

    scheduler.step()



import numpy as np
from torchvision.transforms import Compose, Lambda, ToPILImage
from PIL import Image

'''
model.eval()
ema.shadow = torch.load("ema_shadow.pth", weights_only=True)
ema.apply_shadow()
imgs = sample(model, 32)
'''

image_show('image_step_1.pt')

