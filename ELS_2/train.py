from ELS_2.out_dated.compared_ddpm_version1 import Unet, GaussianDiffusion, Trainer
import torchvision
from torchvision import transforms


folder = 'results_60000_32'
transform = transforms.Compose([
            transforms.ToTensor(),
        ])

ds = torchvision.datasets.MNIST(root="/home/dataset/mnist", train=True, transform=transform, download=True)

unet = Unet(dim = 8, dim_mults = (1,2,4), channels=1)
model = GaussianDiffusion(unet, image_size=28, sampling_timesteps=50)
trainer = Trainer(model, folder, ds = ds, train_num_steps = 450000, save_and_sample_every = 5000, train_batch_size=128)
trainer.train()
