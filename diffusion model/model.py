import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, einsum
from tqdm import tqdm


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.GroupNorm(8, dim)
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(self.norm(x), *args, **kwargs)

class Residual(nn.Module):
    def __init__(self, fn):
        super(Residual, self).__init__()
        self.fn = fn
    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class PositionalEmbeddings(nn.Module):
    def __init__(self, dim):
        super(PositionalEmbeddings, self).__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device

        half_dim = self.dim // 2
        emb = torch.log(torch.tensor(10000)) / (half_dim - 1)
        emb = torch.exp(-emb * torch.arange(half_dim, device = device))
        emb = t[:,None] * emb[None,:]
        emb = torch.cat((torch.sin(emb), torch.cos(emb)), dim = -1)

        return emb


class WideResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim=None, kernel_size=3, num_groups = 8, drop_out = 0.1):
        super(WideResidualBlock, self).__init__()
        self.widen_factor = 1
        self.skip_connection = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.group_norm1 = nn.GroupNorm(num_groups, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels*self.widen_factor, kernel_size, padding=kernel_size//2)
        self.group_norm2 = nn.GroupNorm(num_groups, out_channels * self.widen_factor)
        self.conv2 = nn.Conv2d(out_channels*self.widen_factor,out_channels,  kernel_size, padding=kernel_size//2)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(drop_out)

        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, self.widen_factor*out_channels * 2))
            if exists(time_emb_dim)
            else None
        )

    def forward(self, x, time_emb=None):

        residual = self.skip_connection(x)
        out = self.group_norm1(x)
        out = self.act(out)
        out = self.conv1(out)
        if exists(self.mlp) and exists(time_emb):
            scale_shift = self.mlp(time_emb)
            scale, shift = rearrange(scale_shift, 'b c -> b c 1 1').chunk(2, dim=1)
            out = out * (scale + 1) + shift

        out = self.group_norm2(out)
        out = self.act(out)
        out = self.dropout(out)
        out = self.conv2(out)

        return out + residual


class Attention(nn.Module):
    def __init__(self, channel, resolution, drop_out = 0.1):
        super(Attention, self).__init__()

        if resolution == 16:
            head = 4
            dim_head = 32
        elif resolution == 4:       # 계산 시간 너무 오래 걸리면 16 32로 바꾸기
            head = 8
            dim_head = 64
        else :
            raise Exception("The attention layer is not appropriately positioned.")

        self.scale = dim_head ** -0.5
        self.head = head
        hidden_dim = head * dim_head
        self.to_qkv = nn.Conv2d(channel, 3*hidden_dim, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, channel, 1)
        self.dropout = nn.Dropout(drop_out)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t:rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.head), qkv
        )

        q = q*self.scale
        sim = einsum(q, k, "b h d i, b h d j -> b h i j")
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)
        attn = self.dropout(attn)

        out = einsum(attn, v, 'b h i j, b h d j -> b h i d')
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=h, y=w)

        return self.to_out(out)

class BottleneckResidualBlock(nn.Module):
    def __init__(self, channels, time_emb_dim = None ,num_groups=8):
        super(BottleneckResidualBlock, self).__init__()
        self.group_norm1 = nn.GroupNorm(num_groups, channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.group_norm2 = nn.GroupNorm(num_groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.act = nn.ReLU()

        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, channels * 2))
            if exists(time_emb_dim)
            else None
        )

    def forward(self, x, time_emb):
        shortcut = x
        out = self.group_norm1(x)
        out = self.act(out)
        out = self.conv1(out)

        if exists(self.mlp) and exists(time_emb):
            scale_shift = self.mlp(time_emb)
            scale, shift = rearrange(scale_shift, 'b c -> b c 1 1').chunk(2, dim=1)
            out = out * (scale + 1) + shift

        out = self.group_norm2(out)
        out = self.act(out)
        out = self.conv2(out)

        return out + shortcut

class UNet(nn.Module):
    def __init__(
            self,
            dim = 64,
            channel = 3,
            init_channel = 64,
            out_dim = 3,
            self_condition = False,
            drop_out = 0.1,
    ):
        super(UNet, self).__init__()

        self.channel = channel
        self.self_condition = self_condition
        input_channel = channel * (2 if self_condition else 1)

        self.dims = (64, 128, 256, 256)
        self.resolutions = (32, 16, 8, 4)

        self.init_conv = nn.Conv2d(input_channel, init_channel, 3, padding=1)

        time_dim = dim * 4

        # positional embedding 을 비선형 변환하여 표현력 올리고 이 후 다른 layer에 전달될 때 더 잘 통홥될 수 있도록

        self.time_mlp = nn.Sequential(
            PositionalEmbeddings(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Dropout(drop_out),
            nn.Linear(time_dim, time_dim),
        )

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])

        for idx in range(len(self.dims)-1):
            dim_in = self.dims[idx]
            dim_out = self.dims[idx+1]

            self.downs.append(
                nn.ModuleList([
                    WideResidualBlock(in_channels=dim_in, out_channels=dim_in, time_emb_dim=time_dim),
                    WideResidualBlock(in_channels=dim_in, out_channels=dim_in, time_emb_dim=time_dim),
                    PreNorm(dim_in, Attention(dim_in, self.resolutions[idx]))
                    if self.resolutions[idx] == 16 else nn.Identity(),
                    Downsample(dim_in, dim_out),
                ])
            )

        self.downs.append(
            nn.ModuleList([
                WideResidualBlock(in_channels=self.dims[-1], out_channels=self.dims[-1], time_emb_dim=time_dim),
                WideResidualBlock(in_channels=self.dims[-1], out_channels=self.dims[-1], time_emb_dim=time_dim),
                nn.Identity(),
                nn.Conv2d(self.dims[-1], self.dims[-1], kernel_size=3, padding=1)
            ])
        )

        self.bottle_necks = nn.ModuleList([
            BottleneckResidualBlock(self.dims[-1], time_emb_dim=time_dim),
            PreNorm(self.dims[-1], Attention(self.dims[-1], self.resolutions[-1])),
            BottleneckResidualBlock(self.dims[-1], time_emb_dim=time_dim)
        ])

        for idx in range(len(self.dims)-1):
            dim_in = self.dims[-(idx+1)]
            dim_out = self.dims[-(idx+2)]

            self.ups.append(
                nn.ModuleList([
                    WideResidualBlock(in_channels=dim_in*2, out_channels=dim_in, time_emb_dim=time_dim),
                    WideResidualBlock(in_channels=dim_in*2, out_channels=dim_in, time_emb_dim=time_dim),
                    PreNorm(dim_in, Attention(dim_in, self.resolutions[-(idx+1)]))
                    if self.resolutions[-(idx+1)] == 16 else nn.Identity(),
                    Upsample(dim_in, dim_out),
                ])
            )

        self.ups.append(
            nn.ModuleList([
                WideResidualBlock(in_channels=self.dims[0]*2, out_channels=self.dims[0], time_emb_dim=time_dim),
                WideResidualBlock(in_channels=self.dims[0]*2, out_channels=self.dims[0], time_emb_dim=time_dim),
                nn.Identity(),
                nn.Conv2d(self.dims[0], self.dims[0], kernel_size=3, padding=1)

            ])
        )

        self.out_dim = default(out_dim, channel)

        self.final_res_block = WideResidualBlock(self.dims[0]*2, self.dims[0], time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    def forward(self, x, time, x_self_cond=None):

        if self.self_condition:
            x_self_cond = default(x_self_cond, torch.zeros(x))
            x = torch.cat([x_self_cond, x], dim=1)

        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.bottle_necks[0](x, t)
        x = self.bottle_necks[1](x)
        x = self.bottle_necks[2](x, t)
        a = 0

        for block1, block2, attn, upsample in self.ups:
            a+=1
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((r, x), dim=1)

        x = self.final_res_block(x, t)

        return self.final_conv(x)

def forward_sampling(x, t, alphas=None, noise=None):
    if noise is None:
        noise = torch.randn_like(x)

    if alphas == None:
        alphas = Alpha(linear_beta_schedule(1000))

    return x * alphas.sqrt_alphas_extract(t) + noise * alphas.sqrt_one_minus_alphas_extract(t)

def diffusion_loss(diffusion_model, x, t, noise=None, loss_type = 'l2'):
    if noise is None:
        noise = torch.randn_like(x)

    x_noisy = forward_sampling(x, t, noise=noise)
    predict_loss = diffusion_model(x_noisy, t)

    if loss_type == "l1":
        loss = F.l1_loss(predict_loss, noise)
    elif loss_type == "l2":
        loss = F.mse_loss(predict_loss, noise)
    elif loss_type == 'huber':
        loss = F.smooth_l1_loss(predict_loss, noise)

    return loss

def reverse_sampling(model, x, t, t_index, alpha = None):
    if alpha is None:
        alpha = Alpha(linear_beta_schedule(1000))

    alphas = alpha.alphas_extract(t)

    sqrt_recip_alphas = 1 / torch.sqrt(alphas)

    sqrt_one_minus_alphas = alpha.sqrt_one_minus_alphas_extract(t)

    mean = sqrt_recip_alphas * (x - (1 - alphas) * model(x, t) / sqrt_one_minus_alphas)

    if t_index == 1:
        return mean

    else:
        noise = torch.randn_like(x)
        posterior_variance = alpha.posterior_variance_extract(t)

        return mean + torch.sqrt(posterior_variance) * noise


def reverse_process(model, shape, timesteps = 1000):
    device = next(model.parameters()).device

    batch_size = shape[0]

    img = torch.randn(shape, device=device)
    imgs = []

    with torch.no_grad():
        for t in tqdm(reversed(range(1, 1001)), desc='sampling loop time step', total=timesteps):
            img = reverse_sampling(model, img, torch.full((batch_size,),t, dtype=torch.long, device=device), t)
            imgs.append(img.to('cpu'))
            torch.save(img.to('cpu'), 'results/image_step_'+str(t)+'.pt')

    return img


def sample(model, image_size, batch_size=16, channels=3):
    return reverse_process(model, shape=(batch_size, channels, image_size, image_size))


class EMA():
    def __init__(self, model, decay=0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {name: param.clone().detach() for name, param in model.named_parameters() if param.requires_grad}

    def to(self, device):
        for name in self.shadow:
            self.shadow[name] = self.shadow[name].to(device)

    def update(self):
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    self.shadow[name].mul_(self.decay).add_(param.data, alpha=1 - self.decay)

    def apply_shadow(self):
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    param.data.copy_(self.shadow[name])

    def reset(self):
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    self.shadow[name].copy_(param.data)