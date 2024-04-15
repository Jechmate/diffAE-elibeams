import pandas as pd
import os
import argparse
import torch
import torch.nn as nn
import numpy as np
from src.modules import UNet_conditional
from src.diffusion import SpacedDiffusion
from tqdm import tqdm
from PIL import Image
from src.utils import save_samples
from train import train
from pathlib import Path
from src.dataset import get_1d
from torch.nn.functional import mse_loss, normalize
from dtaidistance import dtw
from pytorch_msssim import ssim

class weighted_MSELoss(nn.Module):
    def __init__ (self):
        super().__init__ ()
    def forward (self, input, target, weight):
        return ((input - target)**2) * weight


def compare_avg(dir1, dir2, save=False):
    # Get all image files in the directories
    files1 = [os.path.join(dir1, f) for f in os.listdir(dir1) if f.endswith('.png')]
    files2 = [os.path.join(dir2, f) for f in os.listdir(dir2) if f.endswith('.png')]

    # Compute the average image for each directory
    avg1 = np.mean([np.array(Image.open(f)) for f in files1], axis=0).astype(np.uint8)
    avg2 = np.mean([np.array(Image.open(f)) for f in files2], axis=0).astype(np.uint8)

    # Save and show the average images
    if save:
        Image.fromarray(avg1).save('average1.jpg')
        Image.fromarray(avg2).save('average2.jpg')
        print('Average images saved as average1.jpg and average2.jpg')
    return avg1, avg2


def calculate_pixelwise_variance(folder_path):
    image_list = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.png'):  # Check for image file extensions as needed
            image_path = os.path.join(folder_path, file_name)
            image = np.array(Image.open(image_path))  # Convert to grayscale
            image_list.append(image)

    # Stack images and calculate variance along the new axis
    stacked_images = np.stack(image_list, axis=2)
    variance_per_pixel = np.var(stacked_images, axis=2)
    return variance_per_pixel


def compare_spectra(train_dir, valid_dir, exp_num):
    train_files = [os.path.join(train_dir, f) for f in os.listdir(train_dir) if f.endswith('.png')]
    valid_files = [os.path.join(valid_dir, f) for f in os.listdir(valid_dir) if f.endswith('.png')]

    # Compute the average image for each directory
    train_avg = np.mean([np.array(Image.open(f)) for f in train_files], axis=0).astype(np.uint8)/255
    valid_avg = np.mean([np.array(Image.open(f)) for f in valid_files], axis=0).astype(np.uint8)/255

    ssim_val = ssim(torch.from_numpy(train_avg).unsqueeze(0).unsqueeze(0), torch.from_numpy(valid_avg).unsqueeze(0).unsqueeze(0), data_range=1.0)

    settings = pd.read_csv("data/params.csv", engine='python')[["E","P","ms"]]
    # exp_num = int(valid_dir.split('_')[-1])
    ms = settings.loc[exp_num - 1]['ms']

    _, spectr_train = get_1d(train_avg, electron_pointing_pixel=62, acquisition_time_ms=ms)
    _, spectr_valid = get_1d(valid_avg, electron_pointing_pixel=62, acquisition_time_ms=ms)
    spectr_train = torch.Tensor(spectr_train)
    spectr_valid = torch.Tensor(spectr_valid)

    spectr_train = torch.where(torch.isnan(spectr_train), torch.zeros_like(spectr_train), spectr_train)
    spectr_valid = torch.where(torch.isnan(spectr_valid), torch.zeros_like(spectr_valid), spectr_valid)

    train_norm = normalize(spectr_train, dim=0)
    valid_norm = normalize(spectr_valid, dim=0)

    mse = mse_loss(spectr_train, spectr_valid)
    mse_norm = mse_loss(train_norm, valid_norm)

    variance_train = np.mean(calculate_pixelwise_variance(train_dir))
    variance_valid = np.mean(calculate_pixelwise_variance(valid_dir))

    var_diff = np.abs(variance_train - variance_valid)

    return {'mse': mse, 'mse_norm' : mse_norm, 'ssim' : ssim_val, 'var_diff' : var_diff}, {'spectr_train' : spectr_train, 'spectr_valid' : spectr_valid}


def sample_all(root="models", result_dir="results/transfer_withgain_512_valid", device='cuda:2', n=8, dataset=Path("data/with_gain"), model_prefix='no_', load_model=True, section_counts=[30], cfg_scale=3, ns=700, model=None):
    if load_model:
        dir_list = [x for x in os.listdir(root) if x.startswith(model_prefix)]
    else:
        dir_list = os.listdir(dataset)
    settings = pd.read_csv("data/params.csv", engine='python')[["E","P","ms"]]
    for subdir in tqdm(dir_list):
        if load_model:
            exp_number = int(subdir.split('_')[1])
        else:
            exp_number = int(subdir)
        original_size = len(os.listdir(Path.joinpath(dataset, str(exp_number))))
        E = settings.loc[exp_number - 1]['E']
        P = settings.loc[exp_number - 1]['P']
        ms = settings.loc[exp_number - 1]['ms']
        if load_model:
            model = UNet_conditional(img_width=128, img_height=64, feat_num=3, device=device).to(device)
            ckpt = torch.load(os.path.join(root, subdir, 'ema_ckpt.pt'), map_location=device)
            model.load_state_dict(ckpt)
            model.eval()
        diffusion = SpacedDiffusion(beta_start=1e-4, beta_end=0.02, section_counts=section_counts, noise_steps=ns, img_width=128, img_height=64, device=device)
        y = torch.Tensor([E,P,ms]).to(device).float().unsqueeze(0) # parameter vector
        res_path = os.path.join(result_dir, str(exp_number))
        os.makedirs(res_path, exist_ok=True)
        total = len([f for f in os.listdir(res_path) if f.endswith('.png')])
        while total < original_size:
            if total + n > original_size:
                add = original_size - total
            else:
                add = n
            x = diffusion.ddim_sample_loop(model, y, cfg_scale=cfg_scale, resize=[256, 512], n=add, eta=1, device=device, gain=50)
            if len(x.shape) == 2:
                x = x.unsqueeze(0)
            res_path = os.path.join(result_dir, str(exp_number))
            save_samples(x, res_path, start_index=total)
            total += add


def main(validate_on = []):
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.epochs = 301
    args.noise_steps = 1000
    args.physinf_thresh = args.noise_steps // 10 # original has // 10
    args.beta_start = 1e-4
    args.beta_end = 0.02
    args.batch_size = 4
    args.image_height = 64
    args.image_width = 128
    args.real_size = (256, 512)
    args.features = ["E","P","ms"]
    args.dataset_path = r"data/gain50"
    args.csv_path = "data/params.csv"
    args.device = "cuda:2"
    args.lr = 1e-3
    args.exclude = []# ['train/19']
    args.grad_acc = 1
    args.sample_freq = 0
    args.sample_settings = [13.,15.,20.]
    args.sample_size = 8
    args.electron_pointing_pixel = 64

    settings = pd.read_csv(args.csv_path, engine='python')[args.features]

    if validate_on:
        experiments = validate_on
    else:
        experiments = os.listdir(args.dataset_path)

    for experiment in sorted(experiments, key=lambda x: int(x)):
        args.exclude = [os.path.join(args.dataset_path, experiment)]
        args.run_name = "valid_nophys_1000_bs4_" + experiment
        row = settings.loc[[int(experiment) - 1], args.features]
        args.sample_settings = row.values.tolist()[0]

        model = UNet_conditional(img_width=128, img_height=64, feat_num=3, device=args.device).to(args.device)
        ckpt = torch.load("models/transfered.pt", map_location=args.device)
        model.load_state_dict(ckpt)
        train(args, model)
# [1, 1, 1, 1, 1, 1, 1, 1, 1, 6] 1x9plus6
# [2, 2, 2, 2, 2, 2, 2, 2, 2, 7] 2x9plus7

if __name__ == "__main__":
    main(validate_on=['3', '8', '11', '19', '21'])
    # validate on: 3, 8, 11, 19, 21
    # device = "cuda:1"
    # # model = UNet_conditional(img_width=128, img_height=64, feat_num=3, device=device).to(device)
    # # ckpt = torch.load('models/nophys_850steps/ema_ckpt.pt', map_location=device)
    # # model.load_state_dict(ckpt)
    # # model.eval()
    # name = 'valid_physsched_850'
    # cfg = 1
    # section_counts = [2, 2, 2, 2, 2, 2, 2, 2, 2, 7]
    # sample_all(load_model=True, root="models/" + name,
    #             result_dir='results/' + name + '_sec' + '18plus7' + '_cfg' + str(cfg),
    #             device=device, ns=850, section_counts=section_counts, n=16, cfg_scale=cfg)
    # cfg_values = [1, 3, 5, 6, 7]
    # section_counts_list = [
    #     [15],
    #     [25],
    #     [45],
    #     [1, 1, 1, 1, 1, 1, 1, 1, 1, 6],
    #     [2, 2, 2, 2, 2, 2, 2, 2, 2, 7]
    # ]
# 
    # for cfg in cfg_values:
    #     for section_counts in section_counts_list:
    #         # Determine the section_count string representation
    #         if len(section_counts) == 1:
    #             section_count_str = str(section_counts[0])
    #         elif section_counts[-1] == 6:
    #             section_count_str = "9plus6"
    #         elif section_counts[-1] == 7:
    #             section_count_str = "18plus7"
    #         else:
    #             section_count_str = "custom"
    #         result_dir = f'results/{name}_sec{section_count_str}_cfg{cfg}'
    #         sample_all(
    #             load_model=True,
    #             root=f"models/{name}",
    #             result_dir=result_dir,
    #             device=device,
    #             ns=700,
    #             section_counts=section_counts,
    #             n=25,
    #             cfg_scale=cfg
    #         )

# not with uniform 50 gain
# results/valid_1st_sig_beamloss_sec18plus7_cfg1
# Average FID: 101.57641512426819600000
# Maximum FID: 158.98257544864143 (in subdirectory 8)
# Minimum FID: 75.1090963337465 (in subdirectory 19)
# MSE: 9914.153369140626 Var_diff: 66.54356847025922

# with uniform 50 gain
# results/valid_noscaleloss3_850_sec2x9plus7_cfg1
# Average FID: 125.93109201228355400000
# Maximum FID: 197.45190346779944 (in subdirectory 8)
# Minimum FID: 74.58281204934039 (in subdirectory 3)
# Variance of FID: 2162.74495532811917761544


# physsched_850


# valid_nophys_1000_bs4_ on eli3
# valid_phys_1000_bs4_ on eli2
# valid_phys_850_bs4 on eli