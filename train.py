import os
import copy
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch import optim
from src.utils import *
from src.modules import UNet_conditional, EMA
import logging
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchsummary import summary
import torchvision.transforms.functional as f
from src.diffusion import *

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")


def sigmoid(x: int,
            scaling: float = 2,
            shift: float = 0,
            ) -> float:
    return 1 - (1 + torch.exp(-x*scaling + shift))**-1


def sigmoid_loss(x: torch.Tensor, el_pointing=64, pixel_in_mm=0.137, device='cpu') -> torch.Tensor:
    distance = torch.arange(-el_pointing, x.shape[-1]-el_pointing, device=device) * pixel_in_mm
    sig = sigmoid(distance).to(device)
    return x*sig


def sigmoid_schedule(step, max_steps=850, k=0.08):
    x_0 = max_steps/10
    return 20 - (20 / (1 + torch.exp(-k * (step - x_0))))



def weighted_mse_loss(input, target, weight):
        return (weight * (input - target) ** 2).mean()


def weighted_mean(input, weight):
    return (weight * input).mean()


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
                {"params": model.down3.maxpool_conv.parameters(), "lr": 1e-4},
                {"params": model.bot1.parameters(), "lr": 1e-5},
                {"params": model.bot2.parameters(), "lr": 1e-5},
                {"params": model.bot3.parameters(), "lr": 1e-5},
                {"params": model.up1.conv.parameters(), "lr": 1e-4},
                {"params": model.up2.conv.parameters(), "lr": 1e-4},
                {"params": model.up3.conv.parameters(), "lr": 1e-3},
                {"params": model.outc.parameters(), "lr": 1e-3},
            ], lr=args.lr,
        )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs*steps_per_epoch)
    mse = nn.MSELoss()
    betas = prepare_noise_schedule(args.noise_steps, args.beta_start, args.beta_end)
    diffusion = GaussianDiffusion(betas=betas, noise_steps=args.noise_steps, img_height=args.image_height, img_width=args.image_width, device=device)
    sampler = SpacedDiffusion(beta_start=args.beta_start, beta_end=args.beta_end, section_counts=[40], noise_steps=args.noise_steps, img_height=args.image_height, img_width=args.image_width, device=device)
    logger = SummaryWriter(os.path.join("runs", args.run_name))
    l = len(dataloader)
    ema = EMA(0.995)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False).to(device)

    el_pointing_adjusted = int(args.electron_pointing_pixel/(args.real_size[1]/args.image_width))
    pixel_in_mm_adjusted = 0.137*(args.real_size[1]/args.image_width)
    fing_x = int(10/(args.real_size[1]/args.image_width))
    fing_y = int(8/(args.real_size[0]/args.image_height))

    # deflection_MeV = deflection_calc(args.batch_size, args.real_size[1], args.electron_pointing_pixel).to(device)
    deflection_MeV = deflection_biexp_calc(args.batch_size, args.real_size[1], args.electron_pointing_pixel, pixel_in_mm_adjusted)[0].to(device)

    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)
        for i, data in enumerate(pbar):
            images = data['image'].to(device)
            settings = data['settings'].to(device)
            acq_time = settings[:, 2]
            t = diffusion.sample_timesteps(images.shape[0], all_same=False).to(device)
            x_t, noise = diffusion.noise_images(images, t)
            if epoch == 0 and i == 0:
                summary(model, x_t, t, settings, device=device)
            if np.random.random() < 0.1:
                settings = None
            predicted_noise = model(x_t, t, settings)
            loss1 = mse(noise, predicted_noise)
            # pred, _ = diffusion.noise_images(images, t, predicted_noise)
            # _, x_t_spectr = calc_spec(((x_t.clamp(-1, 1) + 1) / 2).to(device), 
            #                             args.electron_pointing_pixel, 
            #                             deflection_MeV, 
            #                             acquisition_time_ms=acq_time, 
            #                             resize=args.real_size,
            #                             image_gain=50,
            #                             device=device)
            # _, pred_spectr = calc_spec(((pred.clamp(-1, 1) + 1) / 2).to(device), 
            #                             args.electron_pointing_pixel, 
            #                             deflection_MeV, 
            #                             acquisition_time_ms=acq_time, 
            #                             resize=args.real_size,
            #                             image_gain=50,
            #                             device=device)
            # concatenated = torch.cat((x_t_spectr, pred_spectr), dim=-1)
            # max_val = torch.max(concatenated)
            # min_val = torch.min(concatenated)
            # x_t_spectr_norm = (x_t_spectr - min_val) / ((max_val - min_val) / 2) - 1
            # pred_spectr_norm = (pred_spectr - min_val) / ((max_val - min_val) / 2) - 1
            # pred_norm = (pred.clamp(-1, 1) + 1) / 2
            # pred_norm[:, :, :fing_y, :fing_x] = 0
            # phys_weight = sigmoid_schedule(t, max_steps=args.noise_steps).unsqueeze(1).unsqueeze(2)
            # loss2 = weighted_mse_loss(x_t_spectr_norm, pred_spectr_norm, phys_weight)
            # loss3 = weighted_mean(sigmoid_loss(pred_norm, el_pointing=el_pointing_adjusted, pixel_in_mm=pixel_in_mm_adjusted, device=device), phys_weight)
            loss = loss1# + loss2 + loss3
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema.step_ema(ema_model, model)
            scheduler.step()
            pbar.set_postfix({"_MSE": "{:.4f}".format(loss.item())})#, "BASIC": "{:.4f}".format(loss1.item()), "SPECTR": "{:.4f}".format(loss2.item()), "BEAMPOS": "{:.4f}".format(loss3.item())})
            logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)

        if args.sample_freq and epoch % args.sample_freq == 0:# and epoch > 0:
            settings = torch.Tensor(args.sample_settings).to(device).unsqueeze(0)
            ema_sampled_images = sampler.ddim_sample_loop(model=ema_model, y=settings, cfg_scale=3, device=device, eta=1, n=args.batch_size, resize=args.real_size)
            save_images(ema_sampled_images, os.path.join("results", args.run_name, f"{epoch}_ema.jpg"))
            torch.save(ema_model.state_dict(), os.path.join("models", args.run_name, f"ema_ckpt.pt"))
            torch.save(optimizer.state_dict(), os.path.join("models", args.run_name, f"optim.pt"))
    
    if not args.sample_freq:
        if args.sample_size:
            settings = torch.Tensor(args.sample_settings).to(device).unsqueeze(0)
            ema_sampled_images = sampler.ddim_sample_loop(model=ema_model, y=settings, cfg_scale=3, device=device, eta=1, n=args.batch_size, resize=args.real_size)
            save_samples(ema_sampled_images, os.path.join("results", args.run_name))
        torch.save(ema_model.state_dict(), os.path.join("models", args.run_name, f"ema_ckpt.pt"))
        torch.save(optimizer.state_dict(), os.path.join("models", args.run_name, f"optim.pt"))

def launch():
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = "test"
    args.epochs = 301
    args.noise_steps = 850
    args.physinf_thresh = args.noise_steps // 10 # original has // 10
    args.beta_start = 1e-4
    args.beta_end = 0.02
    args.batch_size = 2
    args.image_height = 64
    args.image_width = 128
    args.real_size = (256, 512)
    args.features = ["E","P","ms"]
    args.dataset_path = r"data/gain50"
    args.csv_path = "data/params.csv"
    args.device = "cuda:0"
    args.lr = 1e-3
    args.exclude = []# ['train/19']
    args.grad_acc = 1
    args.sample_freq = 0
    args.sample_settings = [13.,15.,20.]
    args.sample_size = 8
    args.electron_pointing_pixel = 64

    model = UNet_conditional(img_width=128, img_height=64, feat_num=3, device=args.device).to(args.device)
    ckpt = torch.load("models/transfered.pt", map_location=args.device)
    model.load_state_dict(ckpt)
    train(args, model)


if __name__ == '__main__':
    launch()
