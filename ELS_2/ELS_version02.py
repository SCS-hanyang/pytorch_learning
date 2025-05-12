import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import math
import os
import matplotlib.pyplot as plt
from torch.nn import Module
from tqdm import tqdm
from pathlib import Path
from functools import partial


# 데이터 변환 정의 (텐서 변환 및 정규화)
transform = transforms.Compose([
    transforms.ToTensor(),
])

# MNIST 데이터셋 다운로드
train_dataset = torchvision.datasets.MNIST(root="./data", train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root="./data", train=False, transform=transform, download=True)

# 데이터로더 생성
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=10000, shuffle=True)

def loader(dl):
    while True:
        for dataset in dl:
            yield dataset

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def linear_beta_schedule(timesteps):
    """
    linear schedule, proposed in original ddpm paper
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def gamma_cosine(t, lambda_val=2):
    """
    Cosine schedule 기반 gamma_t 계산
    t: 연속 시간 (0~1 사이)
    lambda_val: Cosine decay의 steepness 조절 파라미터
    """
    s_t = (torch.pi / 2) * t ** lambda_val  # s(t) = (pi/2) * t^lambda
    sin_2s = torch.sin(2 * s_t)
    cos2_s = torch.cos(s_t) ** 2
    gamma_t = (torch.pi / 4) * lambda_val * t ** (lambda_val - 1) * (sin_2s / cos2_s)

    return gamma_t

class Schedule(Module):
    def __init__(self):
        super().__init__()
        betas = linear_beta_schedule(1000).to(torch.float64)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)
        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))


class HierarchicalSoftmax:
    def __init__(self, num_classes):
        self.tree_depth = int(torch.ceil(torch.log2(torch.tensor(num_classes))))  # 트리 깊이

    def forward(self, x):
        prob = 1.0
        for i in range(self.tree_depth):  # 트리 깊이만큼 Softmax 수행
            logits = x[:, 2**i : 2**(i+1)]  # 현재 레벨의 logits
            softmax_out = F.softmax(logits, dim=-1)
            prob *= softmax_out  # 확률을 누적
        return prob

class where_patches():
    def __init__(self, image_t, patch_size):
        # zero padding시 가장자리에서 발생하는 breakin equivariance 방지를 위해, 가장자리 patch는 가장자리 patch에서만 참조하게
        _, c, h, w = image_t.shape
        b = 64  # 이후 수정
        device = image_t.device

        self.h = h

        thres = patch_size // 2

        self.position_tensor = torch.zeros((h, w)).to(device)
        self.position_tensor[:thres, :thres] = 1
        self.position_tensor[:thres, thres:w-thres] = 2
        self.position_tensor[:thres, w-thres:w] = 3
        self.position_tensor[thres:h-thres, :thres] = 4
        self.position_tensor[thres:h-thres, w-thres:w] = 5
        self.position_tensor[h-thres:h, :thres] = 6
        self.position_tensor[h-thres:h, thres:w-thres] = 7
        self.position_tensor[h-thres:h, w-thres:w] = 8

        self.thres = thres

    def find_dataset(self, idx, dataset):
        b, c, ph, pw = dataset.shape
        b = int(b / (self.h * self.h))

        i = idx // self.h
        j = idx % self.h

        h, w = self.h, self.h

        thres = self.thres
        dataset = dataset.view(b, self.h, self.h, c, patch_size, patch_size)

        if self.position_tensor[i,j].item() == 1:
            return dataset[:, thres, :thres, :, :, :].reshape(-1, c, patch_size, patch_size)
        elif self.position_tensor[i,j].item() == 2:
            return dataset[:, :thres, thres:w-thres, :, :, :].reshape(-1, c, patch_size, patch_size)
        elif self.position_tensor[i,j].item() == 3:
            return dataset[:,:thres, w-thres:w, :, :, :].reshape(-1, c, patch_size, patch_size)
        elif self.position_tensor[i,j].item() == 4:
            return dataset[:, thres:h-thres, :thres, :, :, :].reshape(-1, c, patch_size, patch_size)
        elif self.position_tensor[i,j].item() == 5:
            return dataset[:, thres:h-thres, w-thres:w, :, :, :].reshape(-1, c, patch_size, patch_size)
        elif self.position_tensor[i,j].item() == 6:
            return dataset[:, h-thres:h, :thres, :, :, :].reshape(-1, c, patch_size, patch_size)
        elif self.position_tensor[i,j].item() == 7:
            return dataset[:, h-thres:h, thres:w-thres, :, :, :].reshape(-1, c, patch_size, patch_size)
        elif self.position_tensor[i,j].item() == 8:
            return dataset[:, h-thres:h, w-thres:w, :, :, :].reshape(-1, c, patch_size, patch_size)
        else:
            return dataset[:, thres:h-thres, thres:w-thres, :, :, :].reshape(-1, c, patch_size, patch_size)


def partial_image(dataset, patch_size):
    # 각 step에서의 imgs를 patch 단위로 분해 후 재조합
    _, c, _, _ = dataset.shape
    pad_size = (patch_size // 2,) * 4
    dataset = F.pad(dataset, pad=pad_size, mode='constant', value=0)

    dataset = dataset.unfold(2, patch_size, 1).unfold(3, patch_size, 1)
    dataset = dataset.permute(0, 2, 3, 1, 4, 5).contiguous()
    dataset = dataset.view(-1, c, patch_size, patch_size).to(device)

    return dataset

def weight(image_t, dataset, sqrt_at, sqrt_one_minus_at):
    # weight 계산, eucildian 거리의 제곱을 softmax 함수로
    mse = (image_t - sqrt_at * dataset) ** 2
    mse = mse.view(dataset.shape[0], -1).sum(dim=1)
    mse *= -1. / (2 * sqrt_one_minus_at ** 2)

    log_weights = mse - torch.logsumexp(mse, dim=0)
    return torch.exp(log_weights)

def mx(image_t, dataset, time_step, patch_size, schedule):
    # time step t에서 score 계산
    _, c, h, w = (64, 1, 28, 28)    # 임시로 논 거여서 큰 상관 없음(h, c 정보 주려고)

    image_t_partial = partial_image(image_t, patch_size)
    target_partial = partial_image(dataset, patch_size)

    where = where_patches(image_t, patch_size)

    sqrt_at = schedule.sqrt_alphas_cumprod[t]
    sqrt_one_minus_at = schedule.sqrt_one_minus_alphas_cumprod[t].squeeze()

    # Vectorizing `where_patches` computation
    weight_values = []
    image_t_patches = image_t_partial[:, :, patch_size // 2, patch_size // 2]

    max_weight = 0
    idx = 0

    for idx in tqdm(range(h * w), desc="Computing Mx"):

        target_patches = where.find_dataset(idx, target_partial)
        weights = weight(image_t_partial[idx], target_patches, sqrt_at, sqrt_one_minus_at)

        if torch.max(weights) > max_weight:
            max_weight = torch.max(weights)

        target_0 = target_patches[:, :, patch_size // 2, patch_size // 2].squeeze()
        score = (sqrt_at * target_0 - image_t_patches[idx]) * weights
        score = score.sum(dim=0) / (sqrt_one_minus_at ** 2)

        weight_values.append(score)

    score = torch.tensor(weight_values).view(c, h, w).to(device)
    HMax = (torch.argmax(score) + 1) // 28
    WMax = (torch.argmax(score) + 1) % 28

    HMin = (torch.argmin(score) + 1) // 28
    WMin = (torch.argmin(score) + 1) % 28

    print(f"max score is {torch.max(score)}  //  min score is {torch.min(score)}  //  avg score is {torch.abs(score).mean()}\n")
    print(f"max location is ({HMax}, {WMax}) // min location is ({HMin}, {WMin})")
    return score, max_weight


def ddim_sample(schedule, img, device, return_all_timesteps = True):
    sampling_timesteps = 20
    #batch = 1
    eta = 0.
    total_timesteps = 50

    #batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[0], self.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

    times = torch.linspace(-1, total_timesteps - 1, steps = sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
    times = list(reversed(times.int().tolist()))
    time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

    #img = torch.randn(shape, device = device)
    imgs = [img]

    x_start = None

    for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
        #time_cond = torch.full((batch,), time, device = device, dtype = torch.long)

        pred_noise, x_start, max_weight = model_predictions(img, time, schedule, patch_size, device, clip_x_start = True, rederive_pred_noise = True)

        if time_next < 0:
            img = x_start
            imgs.append(img)
            continue

        alpha = schedule.alphas_cumprod[time]
        alpha_next = schedule.alphas_cumprod[time_next]

        sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
        c = (1 - alpha_next - sigma ** 2).sqrt()

        noise = torch.randn_like(img)

        img = x_start * alpha_next.sqrt() + \
              c * pred_noise + \
              sigma * noise

        imgs.append(img)
        img_show = img.squeeze()
        plt.imshow(img_show.cpu().numpy(), cmap='gray')
        plt.axis('off')
        file_name = Path(results_folder) / f"restored_image-{time}.png"
        plt.savefig(file_name, bbox_inches="tight", pad_inches=0)

        print(f"max img pixel : {torch.max(img)} //  min img pixel : {torch.min(img)}")

        print(f"Time step {time} complete. max weight is {max_weight}")

    ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1)

    return ret

def model_predictions(x, t, schedule, patch_size, device, clip_x_start = False, rederive_pred_noise = False):
    maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity
    pred_noise, max_weight = mx(x, dataset, t, patch_size, schedule)
    x_start = predict_start_from_noise(x, torch.tensor([t], device = device), pred_noise, schedule)
    #x_start = maybe_clip(x_start)

    #if clip_x_start and rederive_pred_noise:
    #    pred_noise = predict_noise_from_start(x, t, x_start, schedule)

    return pred_noise, x_start, max_weight

def predict_start_from_noise(x_t, t, noise, schedule):
    return (
        extract(schedule.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
        extract(schedule.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
    )

def predict_noise_from_start(x_t, t, x0, schedule):
    return (
        (extract(schedule.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
        extract(schedule.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
    )

def identity(t, *args, **kwargs):
    return t


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

schedule = Schedule()
schedule = schedule.to(device)

# Main loop
img = torch.randn((1, 1, 28, 28), device=device, dtype = torch.float64)
dl = loader(train_loader)
dataset, labels = next(dl)
dataset = dataset.to(device)
patch_size = 13

dir = os.getcwd()
results_folder = "ELS_2/results_ELS_1"

results_folder = Path(dir) / results_folder

img = ddim_sample(schedule, img, device)






'''for time_step in reversed(range(1000)):
    score, max_weight, H, W = mx(img, dataset, time_step, patch_size, schedule)
    
    
    # linear schedule에서 gamma_t 계산
    gamma_t = torch.min(schedule.betas) + (torch.max(schedule.betas) - torch.min(schedule.betas)) * time_step / 1000
    #gamma_t = gamma_cosine(torch.tensor(time_step / 1000, device=device))

    # deterministic flow 구현
    gamma_t = gamma_t.to(device)
    drift = - gamma_t * (img + score)

    img = img + drift
    print(f"max img pixel : {torch.max(img)} //  min img pixel : {torch.min(img)}")
    # t step에서의 이미지 시각화
    mean, std = 0.1307, 0.3081
    restored_image = img * std + mean

    plt.imshow(restored_image.squeeze().cpu().numpy(), cmap="gray")
    plt.axis("off")

    file_name = Path(results_folder) / f"restored_image-{time_step}.png"
    plt.savefig(file_name, bbox_inches="tight", pad_inches=0)

    print(f"Time step {time_step} complete. max weight is {max_weight}")'''
