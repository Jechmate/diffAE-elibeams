import os
import copy
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch import optim
from utils import *
from modules import UNet_conditional, EMA
import torch_dct
import logging
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchsummary import summary
import torchvision.transforms.functional as f

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")


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


class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_height=120, img_width=300, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        self.img_height = img_height
        self.img_width = img_width
        self.device = device

    def prepare_noise_schedule(self):
        t = torch.linspace(0, 1, self.noise_steps)
        return self.beta_end + 0.5 * (self.beta_start - self.beta_end) * (1 + torch.cos(t * torch.pi))

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        eps = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * eps, eps

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, n, settings, cfg_scale=3, resize=None):
        logging.info(f"Sampling {n} new images....")
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 1, self.img_height, self.img_width)).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t, settings)
                if cfg_scale > 0:
                    uncond_predicted_noise = model(x, t, None)
                    predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        if resize:
            x = f.resize(x, resize, antialias=True)
        return x

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
    blur_sigma_max = 128
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
            # if epoch == 0 and i == 0:
            #     print("summary:")
            #     summary(model, perturbed_data, t, settings, device=device)
            #     print("After summary")
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
            

def train(args, model=None):
    setup_logging(args.run_name)
    device = args.device
    dataloader = get_data(args)
    gradient_acc = args.grad_acc
    steps_per_epoch = len(dataloader) / gradient_acc
    if not model:
        print("Training from scratch")
        model = UNet_conditional(img_height=args.image_height, img_width=args.image_width, device=args.device, feat_num=len(args.features)).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    else:
        optimizer = optim.AdamW([
                {"params": model.inc.parameters(), "lr": 1e-3},
                {"params": model.down1.maxpool_conv.parameters(), "lr": 1e-3},
                {"params": model.down2.maxpool_conv.parameters(), "lr": 1e-4},
                {"params": model.down3.maxpool_conv.parameters(), "lr": 1e-6},
                {"params": model.down4.maxpool_conv.parameters(), "lr": 1e-6},
                {"params": model.bot1.parameters(), "lr": 1e-6},
                {"params": model.bot2.parameters(), "lr": 1e-6},
                {"params": model.bot3.parameters(), "lr": 1e-6},
                {"params": model.up1.conv.parameters(), "lr": 1e-6},
                {"params": model.up2.conv.parameters(), "lr": 1e-6},
                {"params": model.up3.conv.parameters(), "lr": 1e-4},
                {"params": model.up4.conv.parameters(), "lr": 1e-3},
                {"params": model.outc.parameters(), "lr": 1e-3},
            ], lr=args.lr,
        )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs*steps_per_epoch)
    mse = nn.MSELoss()
    diffusion = Diffusion(img_height=args.image_height, img_width=args.image_width, device=device, noise_steps=args.noise_steps, beta_end=args.beta_end)
    logger = SummaryWriter(os.path.join("runs", args.run_name))
    l = len(dataloader)
    ema = EMA(0.8)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False).to(device)

    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)
        for i, data in enumerate(pbar):
            images = data['image'].to(device)
            settings = data['settings'].to(device)
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(images, t)
            if epoch == 0 and i == 0:
                summary(model, x_t, t, settings, device=device)
            if np.random.random() < 0.1:
                settings = None
            predicted_noise = model(x_t, t, settings)
            loss = mse(noise, predicted_noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema.step_ema(ema_model, model)
            scheduler.step()
            # if (i+1) % gradient_acc == 0:

            pbar.set_postfix(MSE=loss.item())
            logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)

        if epoch % 10 == 0:# and epoch > 0:
            settings = torch.Tensor([13.,15.,20.]).to(device).unsqueeze(0)
            # sampled_images = diffusion.sample(model, n=args.batch_size, settings=settings)
            ema_sampled_images = diffusion.sample(ema_model, n=args.batch_size, settings=settings, resize=(256, 512))
            # plot_images(sampled_images)
            # save_images(sampled_images, os.path.join("results", args.run_name, f"{epoch}.jpg"))
            save_images(ema_sampled_images, os.path.join("results", args.run_name, f"{epoch}_ema.jpg"))
            # torch.save(model.state_dict(), os.path.join("models", args.run_name, f"ckpt.pt"))
            torch.save(ema_model.state_dict(), os.path.join("models", args.run_name, f"ema_ckpt.pt"))
            torch.save(optimizer.state_dict(), os.path.join("models", args.run_name, f"optim.pt"))


def launch():
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = "classic_4blocks_bend003"
    args.epochs = 301
    args.noise_steps = 700
    args.beta_end = 0.003
    args.batch_size = 5
    args.image_height = 64
    args.image_width = 128
    args.features = ["E","P","ms"]
    args.dataset_path = r"with_gain"
    args.csv_path = "params.csv"
    args.device = "cuda:0"
    args.lr = 1e-3
    args.exclude = []# ['train/19']
    args.grad_acc = 1

    model = UNet_conditional(img_width=128, img_height=64, feat_num=3, device=args.device).to(args.device)
    ckpt = torch.load("models/transfered_4block.pt")
    model.load_state_dict(ckpt)
    train(args, model)


if __name__ == '__main__':
    launch()
