import os
import copy
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch import optim
from src.utils import *
from src.modules import UNet_conditional, EMA, SemEncoder, MLPSkipNet
import logging
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchsummary import summary
import torchvision.transforms.functional as f
from src.diffusion import *

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")
            

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
    diffusion = GaussianDiffusion(img_height=args.image_height, img_width=args.image_width, device=device, noise_steps=args.noise_steps, beta_end=args.beta_end)
    logger = SummaryWriter(os.path.join("runs", args.run_name))
    l = len(dataloader)
    ema = EMA(0.9)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False).to(device)
    deflection_MeV = deflection_calc(args.batch_size, args.real_size[1], args.electron_pointing_pixel).to(device)

    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)
        for i, data in enumerate(pbar):
            images = data['image'].to(device)
            settings = data['settings'].to(device)
            acq_time = settings[:, 2]
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(images, t)
            if epoch == 0 and i == 0:
                summary(model, x_t, t, settings, device=device)
            if np.random.random() < 0.1:
                settings = None
            predicted_noise = model(x_t, t, settings)
            loss1 = mse(noise, predicted_noise)

            if False: # t[0].item() < args.noise_steps*0.1:
                pred, _ = diffusion.noise_images(images, t, predicted_noise)
                _, x_t_spectr = calc_spec(((x_t.clamp(-1, 1) + 1) / 2).to(device), 
                                          args.electron_pointing_pixel, 
                                          deflection_MeV, 
                                          acquisition_time_ms=acq_time, 
                                          resize=args.real_size,
                                          device=device)
                _, pred_spectr = calc_spec(((pred.clamp(-1, 1) + 1) / 2).to(device), 
                                           args.electron_pointing_pixel, 
                                           deflection_MeV, 
                                           acquisition_time_ms=acq_time, 
                                           resize=args.real_size,
                                           device=device)
                concatenated = torch.cat((x_t_spectr, pred_spectr), dim=-1) # TODO normalizes over batch, would be better image by image
                max_val = torch.max(concatenated)
                min_val = torch.min(concatenated)
                x_t_spectr_norm = (x_t_spectr - min_val) / ((max_val - min_val) / 2) - 1
                pred_spectr_norm = (pred_spectr - min_val) / ((max_val - min_val) / 2) - 1
                x_t_spectr_norm = x_t_spectr_norm.to(device)
                pred_spectr_norm = pred_spectr_norm.to(device)
                loss2 = mse(x_t_spectr_norm, pred_spectr_norm) * 10
                loss2.requires_grad = True # TODO why is this necessary? Without it it doesnt have a grad_fn which feels wrong
                el_pointing_adjusted = int(args.electron_pointing_pixel/(args.real_size[1]/args.image_width))
                pred_norm = (pred.clamp(-1, 1) + 1) / 2
                loss3 = pred_norm[:, :, :, :el_pointing_adjusted].mean(dim=(0, -2, -1))
                print(f"Basic: {loss1}, 1D: {loss2}, Beam_pos: {loss3}")
                loss = loss1 + loss2 + loss3
            else:
                loss = loss1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema.step_ema(ema_model, model)
            scheduler.step()
            # if (i+1) % gradient_acc == 0:

            pbar.set_postfix(MSE=loss.item())
            logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)

        if args.sample_freq and epoch % args.sample_freq == 0:# and epoch > 0:
            settings = torch.Tensor(args.sample_settings).to(device).unsqueeze(0)
            ema_sampled_images = diffusion.sample(ema_model, n=args.sample_size, settings=settings, resize=(256, 512))
            save_images(ema_sampled_images, os.path.join("results", args.run_name, f"{epoch}_ema.jpg"))
            torch.save(ema_model.state_dict(), os.path.join("models", args.run_name, f"ema_ckpt.pt"))
            torch.save(optimizer.state_dict(), os.path.join("models", args.run_name, f"optim.pt"))
    
    if not args.sample_freq:
        if args.sample_size:
            settings = torch.Tensor(args.sample_settings).to(device).unsqueeze(0)
            ema_sampled_images = diffusion.sample(ema_model, n=args.sample_size, settings=settings, resize=(256, 512))
            save_samples(ema_sampled_images, os.path.join("results", args.run_name))
        torch.save(ema_model.state_dict(), os.path.join("models", args.run_name, f"ema_ckpt.pt"))
        torch.save(optimizer.state_dict(), os.path.join("models", args.run_name, f"optim.pt"))


def train_enc_test(args):
    setup_logging(args.run_name)
    device = args.device
    dataloader = get_data(args)
    gradient_acc = args.grad_acc
    steps_per_epoch = len(dataloader) / gradient_acc
    model = Encoder(img_height=args.image_height, img_width=args.image_width, device=args.device, feat_num=len(args.features)).to(args.device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs*steps_per_epoch)
    mse = nn.MSELoss()
    diffusion = SpacedDiffusion(beta_start=1e-4, beta_end=0.02, noise_steps=args.noise_steps, section_counts=[10, 20, 10], img_height=64, img_width=128, device=device, rescale_timesteps=False)
    logger = SummaryWriter(os.path.join("runs", args.run_name))
    l = len(dataloader)
    ema = EMA(0.9)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False).to(device)
    deflection_MeV = deflection_calc(args.batch_size, args.real_size[1], args.electron_pointing_pixel).to(device)

    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)
        for i, data in enumerate(pbar):
            images = data['image'].to(device)
            settings = data['settings'].to(device)
            if epoch == 0 and i == 0:
                summary(model, images, settings, device=device)
            # if np.random.random() < 0.1:
            #     settings = None
            predicted_noise = model(images, settings)
            loss = mse(predicted_noise, predicted_noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema.step_ema(ema_model, model)
            scheduler.step()
            # if (i+1) % gradient_acc == 0:

            pbar.set_postfix(MSE=loss.item())
            logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)


def train_AE(args, model_stoch=None, model_sem=None):
    setup_logging(args.run_name)
    device = args.device
    dataloader = get_data(args)
    gradient_acc = args.grad_acc
    steps_per_epoch = len(dataloader) / gradient_acc
    if not model_stoch:
        print("Training stoch from scratch")
        model_stoch = UNet_conditional(img_height=args.image_height, img_width=args.image_width, device=args.device, feat_num=args.latent_dim).to(device)
    if not model_sem:
        print("Training sem from scratch")
        model_sem = SemEncoder(img_height=args.image_height, img_width=args.image_width, device=args.device, feat_num=len(args.features)).to(device)
    
    optimizer = optim.AdamW([
                {"params": model_stoch.inc.parameters(), "lr": 1e-3},
                {"params": model_stoch.down1.maxpool_conv.parameters(), "lr": 1e-3},
                {"params": model_stoch.down2.maxpool_conv.parameters(), "lr": 1e-4},
                {"params": model_stoch.down3.maxpool_conv.parameters(), "lr": 1e-6},
                {"params": model_stoch.bot1.parameters(), "lr": 1e-6},
                {"params": model_stoch.bot2.parameters(), "lr": 1e-6},
                {"params": model_stoch.bot3.parameters(), "lr": 1e-6},
                {"params": model_stoch.up1.conv.parameters(), "lr": 1e-6},
                {"params": model_stoch.up2.conv.parameters(), "lr": 1e-4},
                {"params": model_stoch.up3.conv.parameters(), "lr": 1e-3},
                {"params": model_stoch.outc.parameters(), "lr": 1e-3},
                {"params": model_sem.inc.parameters(), "lr": 1e-3},
                {"params": model_sem.down1.maxpool_conv.parameters(), "lr": 1e-3},
                {"params": model_sem.down2.maxpool_conv.parameters(), "lr": 1e-4},
                {"params": model_sem.down3.maxpool_conv.parameters(), "lr": 1e-6}
            ], lr=args.lr, # TODO does this work the way I hope or does it exclude any parameters that I didnt list here?
        )
    
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs*steps_per_epoch)
    mse = nn.MSELoss()
    diffusion = SpacedDiffusion(img_height=args.image_height, img_width=args.image_width, device=device, noise_steps=args.noise_steps, beta_start=args.beta_start, beta_end=args.beta_end, section_counts=[20])
    logger = SummaryWriter(os.path.join("runs", args.run_name))
    l = len(dataloader)
    ema_stoch = EMA(0.9)
    ema_sem = EMA(0.9)
    ema_model_stoch = copy.deepcopy(model_stoch).eval().requires_grad_(False).to(device)
    ema_model_sem = copy.deepcopy(model_sem).eval().requires_grad_(False).to(device)
    # deflection_MeV = deflection_calc(args.batch_size, args.real_size[1], args.electron_pointing_pixel).to(device)

    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)
        for i, data in enumerate(pbar):
            images = data['image'].to(device)
            settings = data['settings'].to(device)
            # acq_time = settings[:, 2]
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(images, t)
            sem_vec = model_sem(images, settings)
            predicted_noise = model_stoch(x_t, t, sem_vec)
            loss1 = mse(noise, predicted_noise)
            loss = loss1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema_stoch.step_ema(ema_model_stoch, model_stoch)
            ema_sem.step_ema(ema_model_sem, model_sem)
            scheduler.step()
            # if (i+1) % gradient_acc == 0:

            pbar.set_postfix(MSE=loss.item())
            logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)

        if args.sample_freq and epoch % args.sample_freq == 0:# and epoch > 0:
            # settings = torch.Tensor(args.sample_settings).to(device).unsqueeze(0)
            # ema_sampled_images = diffusion.sample(ema_model, n=args.sample_size, settings=settings, resize=(256, 512))
            # save_images(ema_sampled_images, os.path.join("results", args.run_name, f"{epoch}_ema.jpg"))
            torch.save(ema_model_stoch.state_dict(), os.path.join("models", args.run_name, f"ema_stoch_ckpt.pt"))
            torch.save(ema_model_sem.state_dict(), os.path.join("models", args.run_name, f"ema_sem_ckpt.pt"))
            torch.save(optimizer.state_dict(), os.path.join("models", args.run_name, f"optim.pt"))
    
    if not args.sample_freq:
        # if args.sample_size:
        #     settings = torch.Tensor(args.sample_settings).to(device).unsqueeze(0)
        #     ema_sampled_images = diffusion.sample(ema_model, n=args.sample_size, settings=settings, resize=(256, 512))
        #     save_samples(ema_sampled_images, os.path.join("results", args.run_name))
        torch.save(ema_model_stoch.state_dict(), os.path.join("models", args.run_name, f"ema_stoch_ckpt.pt"))
        torch.save(ema_model_sem.state_dict(), os.path.join("models", args.run_name, f"ema_sem_ckpt.pt"))
        torch.save(optimizer.state_dict(), os.path.join("models", args.run_name, f"optim.pt"))


def train_latent(args, model_sem):
    setup_logging(args.run_name)
    device = args.device
    dataloader = get_data(args)
    gradient_acc = args.grad_acc
    steps_per_epoch = len(dataloader) / gradient_acc
    
    model_sem.eval()
    model_lat = MLPSkipNet(device=device)

    optimizer = optim.AdamW(model_lat.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs*steps_per_epoch)
    lss = nn.L1Loss()
    betas = prepare_noise_schedule(args.noise_steps, beta_start=args.beta_start, beta_end=args.beta_end)
    diffusion = GaussianDiffusion(img_height=1, img_width=128, device=device, noise_steps=args.noise_steps, betas=betas)
    logger = SummaryWriter(os.path.join("runs", args.run_name))
    l = len(dataloader)
    ema_lat = EMA(0.9)

    ema_model_lat = copy.deepcopy(model_lat).eval().requires_grad_(False).to(device)
    # deflection_MeV = deflection_calc(args.batch_size, args.real_size[1], args.electron_pointing_pixel).to(device)

    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)
        for i, data in enumerate(pbar):
            images = data['image'].to(device)
            settings = data['settings'].to(device)
            # acq_time = settings[:, 2]
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            sem_vec = model_sem(images, settings).to(device)
            x_t, noise = diffusion.noise_sem(sem_vec, t)
            predicted_noise = model_lat(x_t, t, settings)
            loss1 = lss(noise, predicted_noise)
            loss = loss1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema_lat.step_ema(ema_model_lat, model_lat)
            scheduler.step()
            # if (i+1) % gradient_acc == 0:

            pbar.set_postfix(L1=loss.item())
            logger.add_scalar("L1", loss.item(), global_step=epoch * l + i)

        if args.sample_freq and epoch % args.sample_freq == 0:# and epoch > 0:
            # settings = torch.Tensor(args.sample_settings).to(device).unsqueeze(0)
            # ema_sampled_images = diffusion.sample(ema_model, n=args.sample_size, settings=settings, resize=(256, 512))
            # save_images(ema_sampled_images, os.path.join("results", args.run_name, f"{epoch}_ema.jpg"))
            torch.save(ema_model_lat.state_dict(), os.path.join("models", args.run_name, f"ema_lat_ckpt.pt"))
            torch.save(optimizer.state_dict(), os.path.join("models", args.run_name, f"optim_lat.pt"))
    
    if not args.sample_freq:
        # if args.sample_size:
        #     settings = torch.Tensor(args.sample_settings).to(device).unsqueeze(0)
        #     ema_sampled_images = diffusion.sample(ema_model, n=args.sample_size, settings=settings, resize=(256, 512))
        #     save_samples(ema_sampled_images, os.path.join("results", args.run_name))
        torch.save(ema_model_lat.state_dict(), os.path.join("models", args.run_name, f"ema_lat_ckpt.pt"))
        torch.save(optimizer.state_dict(), os.path.join("models", args.run_name, f"optim_lat.pt"))



def launch():
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = "diffAE_new_out"
    args.epochs = 301
    args.noise_steps = 700
    args.beta_start = 1e-4
    args.beta_end = 0.02
    args.batch_size = 6
    args.image_height = 64
    args.image_width = 128
    args.real_size = (256, 512)
    args.features = ["E","P","ms"]
    args.dataset_path = r"data/with_gain"
    args.csv_path = "data/params.csv"
    args.device = "cuda:3"
    args.lr = 1e-3
    args.exclude = []# ['train/19']
    args.grad_acc = 1
    args.sample_freq = 10
    args.sample_settings = [13.,15.,20.]
    args.sample_size = 8
    args.electron_pointing_pixel = 62
    args.latent_dim = 128
    model_sem = SemEncoder(img_width=128, img_height=64, feat_num=3, device=args.device).to(args.device)
    ckpt = torch.load("models/transfered_sem.pt", map_location=args.device)
    model_sem.load_state_dict(ckpt)
    # train_latent(args, model_sem)

    model = UNet_conditional(img_width=128, img_height=64, feat_num=128, device=args.device).to(args.device)
    ckpt = torch.load("models/transfered_3block.pt", map_location=args.device)
    model.load_state_dict(ckpt)
    train_AE(args, model, model_sem)


if __name__ == '__main__':
    launch()
