import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import os
from google.colab import drive
from torch.optim.lr_scheduler import LambdaLR
import math

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

model = UNet(dim = 64, dim_advanced=(1,2,4,8))
model.to(device)
ema = EMA(model)
ema.to(device)

drive.mount('/content/drive')

epochs = 300


optimizer = Adam(model.parameters(), lr=1e-4)

def warmup_cosine_decay_scheduler(optimizer, warmup_steps, total_steps, eta_max=1e-4, eta_min=1e-5):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            # Warmup 단계
            return current_step / warmup_steps
        else:
            # Cosine Decay 단계
            progress = (current_step - warmup_steps) / (total_steps - warmup_steps)
            return eta_min / eta_max + (1 - eta_min / eta_max) * 0.5 * (1 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)

# Scheduler 생성
total_steps = len(train_loader) * 300
warmup_steps = int(0.1 * total_steps)
scheduler = warmup_cosine_decay_scheduler(optimizer, warmup_steps, total_steps)

best_loss = float('inf')

for epoch in range(epochs):
    for step, data in tqdm(enumerate(train_loader),desc = 'training',total = len(train_loader)):

        images, _ = data
        images = images.to(device)
        optimizer.zero_grad()

        batch_size = images.shape[0]

        t = torch.randint(1, 1001, (batch_size,), device=device).long().to(device)

        loss = diffusion_loss(diffusion_model = model, x = images, t=t, loss_type='l2')
        print(loss)

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_loss_file = 'mini_batch_loss.pth'
            torch.save(model.state_dict(), best_loss_file)

        loss.backward()
        total_norm = torch.norm(torch.stack([p.grad.norm(2) for p in model.parameters() if p.grad is not None]), 2)
        if total_norm > 1.0:
            print(total_norm)
        optimizer.step()
        scheduler.step()
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

        !cp ema_shadow.pth /content/drive/MyDrive/
        !cp mini_batch_loss.pth /content/drive/MyDrive/

    if step % 50 == 49:

        checkpoint_path = "checkpoint.pth"
        torch.save({
                'epoch': epoch,
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'ema_shadow': ema.shadow,
                'best_loss': best_loss,
            }, checkpoint_path)

        !cp checkpoint.pth /content/drive/MyDrive/
        print(f"Checkpoint 저장 완료! Epoch: {epoch}, Step: {step}")

    scheduler.step()


'''
best_loss_file = 'mini_batch_loss.pth'
for epoch in range(start_epoch, epochs):  # 시작 에폭부터 진행
    for step, data in tqdm(enumerate(train_loader, start=start_step if epoch == start_epoch else 0),desc = 'training',
                           total = len(train_loader)):

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

    print("\nLoss :", loss.item())
    ema_shadow_file = "ema_shadow.pth"
    torch.save(ema.shadow, ema_shadow_file)

    if epoch % 10 == 9:

        file_path = '/content/drive/MyDrive/'

        if os.path.exists(file_path+ema_shadow_file):
            os.remove(file_path+ema_shadow_file)
        if os.path.exists(file_path+best_loss_file):
            os.remove(file_path+best_loss_file)

        !cp ema_shadow.pth /content/drive/MyDrive/
        !cp mini_batch_loss.pth /content/drive/MyDrive/

    if step % 100 == 0:  # 100 스텝마다 임시 저장
            torch.save({
                'epoch': epoch,
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'ema_shadow': ema.shadow,
                'best_loss': best_loss,
            }, checkpoint_path)


            print(f"Checkpoint 저장 완료! Epoch: {epoch}, Step: {step}")

    scheduler.step()


'''


