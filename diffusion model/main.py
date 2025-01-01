import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import UNet, EMA, sample, f_sample
from f import image_show, image_save
import numpy as np
from torchvision.transforms import Compose, Lambda, ToPILImage
from PIL import Image


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


# 데이터 로더 생성
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True)


device = 'cuda' if torch.cuda.is_available() else "cpu"

model = UNet(dim = 128, dim_schedule=(1,2,2,4))
model.to(device)
ema = EMA(model)
ema.to(device)

def warmup_cosine_decay_scheduler(
    optimizer,
    warmup_steps,
    total_steps,
    eta_start=1e-6,
    eta_max=1e-4,
    eta_min=1e-5,
):

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return (
                (eta_start / eta_max) +
                (1 - eta_start / eta_max) * (current_step / warmup_steps)
            )
        else:
            progress = (current_step - warmup_steps) / (total_steps - warmup_steps)
            return (
                (eta_min / eta_max) +
                (1 - eta_min / eta_max) * 0.5 * (1 + math.cos(math.pi * progress))
            )

    return LambdaLR(optimizer, lr_lambda)

drive.mount('/content/drive')

epochs = 300

optimizer = Adam(model.parameters(), lr=1e-4)
total_steps = len(train_loader) * 300
warmup_steps = int(0.1 * total_steps)
scheduler = warmup_cosine_decay_scheduler(
    optimizer,
    warmup_steps=warmup_steps,
    total_steps=total_steps,
    eta_start=1e-6,
    eta_max=1e-4,
    eta_min=1e-5,
)

restart = 1

start_epoch, start_step, best_loss = check_point_return(model, ema, scheduler, optimizer, restart=restart)

clip = 0
for epoch in range(start_epoch, epochs):
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
            best_loss_file = 'model_parameter/mini_batch_loss.pth'
            torch.save(model.state_dict(), best_loss_file)

        loss.backward()

        total_norm = torch.norm(torch.stack([p.grad.norm(2) for p in model.parameters() if p.grad is not None]), 2)
        if total_norm > 2:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            clip += 1
        optimizer.step()
        scheduler.step()

        ema.update()

    print(f"epoch : {epoch+1}/{epochs} Loss:{loss.item()}")
    print(f"cliping : {clip / len(train_loader)*100}%")
    clip = 0
    ema_shadow_file = "model_parameter/ema_shadow.pth"
    torch.save(ema.shadow, ema_shadow_file)

    if epoch % 10 == 9:

        file_path = '/content/drive/MyDrive/'
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



model.eval()
ema.shadow = torch.load("model_parameter/ema_shadow.pth", weights_only=True, map_location='cpu')
ema.apply_shadow()


test = iter(test_loader)
test_sample = next(test)
sources = f_sample(images=test_sample[0], file_num=1)
imgs = sample(model, file_num= 1, image= sources)


