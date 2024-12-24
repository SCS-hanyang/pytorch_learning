from model import UNet, diffusion_loss
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim import Adam

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2470, 0.2435, 0.2616)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(cifar10_mean, cifar10_std)
])

train_dataset = datasets.CIFAR10(
    root='./data',
    train=True,
    transform=transform
)

test_dataset = datasets.CIFAR10(
    root='./data',
    train=False,  # 테스트용 데이터
    transform=transform
)

# 데이터 로더 생성
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, drop_last=True)

'''
a = iter(train_loader)
image, label = next(a)
print(image.shape)


img, label = train_dataset[0]
img.shape = (64, 3, 32, 32)
label.shape = (64,)
'''

device = 'cuda' if torch.cuda.is_available() else "cpu"

model = UNet()
optimizer = Adam(model.parameters(), lr=1e-3)

epochs = 6

best_loss = float('inf')

for epoch in range(epochs):
    for step, data in tqdm(enumerate(train_loader),desc = 'training', total = len(train_loader)):
        
        images, _ = data
        images = images.to(device)
        optimizer.zero_grad()

        batch_size = images.shape[0]

        t = torch.randint(1, 1001, (batch_size,), device=device).long()

        loss = diffusion_loss(diffusion_model = model, x = images, t=t, loss_type='l2')

        if step % 100 == 0:
            print("\nLoss:", loss.item())

        if loss.item() < best_loss:
            torch.save(model.state_dict(), 'unet_weights.pth')

        loss.backward()
        optimizer.step()

