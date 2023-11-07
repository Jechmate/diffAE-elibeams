import torch
import torch.nn as nn
import numpy as np
import torch_dct
from utils import *
from modules import *
from tqdm import tqdm
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
import logging
from torchsummary import summary
import copy

class DCTBlur(nn.Module): # TODO use pytorch instead of np

    def __init__(self, img_width, img_height, device, noise_steps):
        super(DCTBlur, self).__init__()
        self.device = device
        self.noise_steps = noise_steps
        freqs_hor = np.pi*torch.linspace(0, img_width-1,img_width).to(device)/img_width
        freqs_ver = np.pi*torch.linspace(0, img_height-1,img_height).to(device)/img_height
        self.frequencies_squared = freqs_hor[None, :]**2 + freqs_ver[:, None]**2 # swapped None and :, sizes didnt match

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def prepare_blur_schedule(self, blur_sigma_max, blur_sigma_min):
        self.blur_schedule = torch.Tensor().to(self.device)
        self.blur_schedule = np.exp(np.linspace(np.log(blur_sigma_min), np.log(blur_sigma_max), self.noise_steps))
        self.blur_schedule = torch.Tensor(np.array([0] + list(self.blur_schedule))).to(self.device)  # Add the k=0 timestep

    def forward(self, x, t):
        sigmas = self.blur_schedule[t][:, None, None, None]
        t = sigmas**2/2
        dct_coefs = torch_dct.dct_2d(x, norm='ortho')
        dct_coefs = dct_coefs * torch.exp(-self.frequencies_squared * t)
        return torch_dct.idct_2d(dct_coefs, norm='ortho')
    
    def get_initial_sample(self, trainloader, device):
        """Take a draw from the prior p(u_K)"""
        initial_sample = next(iter(trainloader))['image'].to(device)
        # original_images = initial_sample.clone()
        initial_sample = self.forward(initial_sample, (self.noise_steps * torch.ones(initial_sample.shape[0]).long()).to(device))
        return initial_sample # , original_images
    
    def sample(self, trainloader, device, model, delta, settings, cfg_scale=3, resize=None):
        initial_sample = self.get_initial_sample(trainloader, device)
        with torch.no_grad():
            u = initial_sample.to(device).float()
            for i in tqdm(range(self.noise_steps, 0, -1)):
                vec_fwd_steps = torch.ones(
                    initial_sample.shape[0], device=device, dtype=torch.long) * i
                # Predict less blurry mean
                u_mean = model(u, vec_fwd_steps, settings) + u
                if cfg_scale > 0: # TODO this may be bs
                    uncond_mean = model(u, vec_fwd_steps, None) + u
                    u_mean = torch.lerp(uncond_mean, u_mean, cfg_scale)
                # Sampling step
                noise = torch.randn_like(u)
                u = u_mean + noise*delta
            u_mean = (u_mean.clamp(-1, 1) + 1) / 2
            u_mean = (u_mean * 255).type(torch.uint8)
            if resize:
                u_mean = f.resize(u_mean, resize, antialias=True)
            return u_mean
    

def train_IHD(args, model=None):
    # dist.init_process_group(backend='nccl', init_method='env://', rank=torch.cuda.device_count(), world_size=1)
    setup_logging(args.run_name)
    device = args.device
    dataloader = get_data(args)
    gradient_acc = args.grad_acc
    steps_per_epoch = len(dataloader) / gradient_acc
    if not model:
        model = UNet_conditional(img_height=args.image_height, img_width=args.image_width, device=device, feat_num=len(args.features)).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    else:
        optimizer = optim.AdamW([
                {"params": model.inc.parameters(), "lr": 1e-3},
                {"params": model.down1.maxpool_conv.parameters(), "lr": 1e-3},
                {"params": model.down2.maxpool_conv.parameters(), "lr": 1e-4},
                {"params": model.down3.maxpool_conv.parameters(), "lr": 1e-6},
                {"params": model.bot1.parameters(), "lr": 1e-6},
                {"params": model.bot2.parameters(), "lr": 1e-6},
                {"params": model.bot3.parameters(), "lr": 1e-6},
                {"params": model.up1.conv.parameters(), "lr": 1e-6},
                {"params": model.up2.conv.parameters(), "lr": 1e-4},
                {"params": model.up3.conv.parameters(), "lr": 1e-3},
                {"params": model.outc.parameters(), "lr": 1e-3},
            ], lr=args.lr,
        )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs*steps_per_epoch)
    mse = nn.MSELoss()
    logger = SummaryWriter(os.path.join("runs", args.run_name))
    l = len(dataloader)
    ema = EMA(0.995)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)
    sigma = 0.01
    delta = 0.0125
    blur_sigma_max = 64
    blur_sigma_min = 0.5
    heat_forward_module = DCTBlur(img_width=args.image_width, img_height=args.image_height, device=args.device, noise_steps=args.noise_steps)
    heat_forward_module.prepare_blur_schedule(blur_sigma_max, blur_sigma_min)

    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)
        for i, data in enumerate(pbar):
            images = data['image'].to(device)
            settings = data['settings'].to(device)
            t = heat_forward_module.sample_timesteps(images.shape[0]).to(device)
            blurred_batch = heat_forward_module(images, t).float()
            less_blurred_batch = heat_forward_module(images, t-1).float()
            noise = torch.randn_like(blurred_batch) * sigma
            perturbed_data = noise + blurred_batch
            if epoch == 0 and i == 0:
                summary(model, perturbed_data, t, settings, device=device)
            diff = model(perturbed_data, t, settings)
            prediction = perturbed_data + diff
            loss = mse(less_blurred_batch, prediction)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema.step_ema(ema_model, model)
            scheduler.step()
            pbar.set_postfix(MSE=loss.item())
            logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)

        if epoch % 10 == 0:
            settings = torch.Tensor([13.,15,20.]).to(device).unsqueeze(0)
            ema_sampled_images = heat_forward_module.sample(trainloader=get_data(args), device=args.device, model=ema_model, delta=delta, settings=settings, resize=(256, 512))
            save_images(ema_sampled_images, os.path.join("results", args.run_name, f"{epoch}_ema.jpg"))
            torch.save(ema_model.state_dict(), os.path.join("models", args.run_name, f"ema_ckpt.pt"))
            torch.save(optimizer.state_dict(), os.path.join("models", args.run_name, f"optim.pt"))