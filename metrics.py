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
    dist = dtw.distance(spectr_train.numpy(), spectr_valid.numpy())
    return {'mse': mse, 'mse_norm' : mse_norm, 'dtw_dist' : dist, 'ssim' : ssim_val}, {'spectr_train' : spectr_train, 'spectr_valid' : spectr_valid}


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
        total = 0
        while total != original_size:
            if total + n > original_size:
                add = original_size - total
            else:
                add = n
            x = diffusion.ddim_sample_loop(model, y, cfg_scale=cfg_scale, resize=[256, 512], n=add, eta=1, device=device)
            res_path = os.path.join(result_dir, str(exp_number))
            os.makedirs(res_path, exist_ok=True)
            save_samples(x, res_path, start_index=total)
            total += add


def main(validate_on = []):
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    # args.run_name = "physinf_tenth_1000ns"
    args.epochs = 301
    args.noise_steps = 700
    args.physinf_thresh = 0 # args.noise_steps // 10 # original has // 10
    args.beta_start = 1e-4
    args.beta_end = 0.02
    args.batch_size = 8
    args.image_height = 64
    args.image_width = 128
    args.real_size = (256, 512)
    args.features = ["E","P","ms"]
    args.dataset_path = r"data/with_gain"
    args.csv_path = "data/params.csv"
    args.device = "cuda:3"
    args.lr = 1e-3
    # args.exclude = []# ['train/19']
    args.grad_acc = 1
    args.sample_freq = 0
    args.sample_settings = [13.,15.,20.]
    args.sample_size = 8
    args.electron_pointing_pixel = 62

    settings = pd.read_csv(args.csv_path, engine='python')[args.features]

    if validate_on:
        experiments = validate_on
    else:
        experiments = os.listdir(args.dataset_path)

    for experiment in sorted(experiments, key=lambda x: int(x)):
        args.exclude = [os.path.join(args.dataset_path, experiment)]
        args.run_name = "valid_nophys_700ns_no_" + experiment
        row = settings.loc[[int(experiment) - 1], args.features]
        args.sample_settings = row.values.tolist()[0]

        model = UNet_conditional(img_width=128, img_height=64, feat_num=3, device=args.device).to(args.device)
        ckpt = torch.load("models/transfered.pt", map_location=args.device)
        model.load_state_dict(ckpt)
        train(args, model)


if __name__ == "__main__":
    # main(validate_on=['3', '8', '11', '19', '21'])
    # validate on: 3, 8, 11, 19, 21
    device = "cuda:1"
    # model = UNet_conditional(img_width=128, img_height=64, feat_num=3, device=device).to(device)
    # ckpt = torch.load('models/nophys_850steps/ema_ckpt.pt', map_location=device)
    # model.load_state_dict(ckpt)
    # model.eval()
    sample_all(load_model=True, root="models/valid_phys10th_850ns", result_dir='results/valid_phys10th_850ns_sec1x9plus6_cfg6', device=device, ns=850, section_counts=[1, 1, 1, 1, 1, 1, 1, 1, 1, 6], n=30, cfg_scale=6)


# results/valid_nophys_1000ns_sec25_cfg3
# Average FID: 119.05659711519518200000
# Maximum FID: 153.38061429650202 (in subdirectory 8)
# Minimum FID: 80.31072614235177 (in subdirectory 21)


# results/valid_nophys_1000ns_sec25_cfg5
# Average FID: 122.66234388646795200000
# Maximum FID: 162.45260141287113 (in subdirectory 8)
# Minimum FID: 89.06457458686222 (in subdirectory 3)


# results/valid_nophys_1000ns_sec25_cfg1
# Average FID: 118.95029222594557800000
# Maximum FID: 160.36977156427324 (in subdirectory 8)
# Minimum FID: 87.18477539802609 (in subdirectory 21)


# results/valid_nophys_1000ns_sec15_cfg5
# Average FID: 123.69727537219869400000
# Maximum FID: 153.08863307888478 (in subdirectory 19)
# Minimum FID: 98.59411944839258 (in subdirectory 3)


# results/valid_nophys_1000ns_sec15_cfg3
# Average FID: 119.17753343017793800000
# Maximum FID: 153.2653922356985 (in subdirectory 19)
# Minimum FID: 91.48876159365918 (in subdirectory 3)


# results/valid_nophys_1000ns_sec15_cfg1
# Average FID: 120.61537912189690000000
# Maximum FID: 156.4156695391676 (in subdirectory 8)
# Minimum FID: 101.34343741393124 (in subdirectory 3)


# results/valid_nophys_1000ns_sec45_cfg1
# Average FID: 118.02157381786009800000
# Maximum FID: 161.67525935193527 (in subdirectory 8)
# Minimum FID: 92.62521276506443 (in subdirectory 3)


# results/valid_nophys_1000ns_sec45_cfg3
# Average FID: 116.65401054124966000000
# Maximum FID: 152.86376756881324 (in subdirectory 19)
# Minimum FID: 75.44730194382555 (in subdirectory 21)


# results/valid_nophys_1000ns_sec45_cfg5
# Average FID: 116.79243957971613800000
# Maximum FID: 163.38900325378333 (in subdirectory 8)
# Minimum FID: 74.9725797521477 (in subdirectory 21)


# results/valid_phys10_1000ns_sec25_cfg3
# Average FID: 117.11994622087477800000
# Maximum FID: 152.55426121455272 (in subdirectory 19)
# Minimum FID: 67.85619006362464 (in subdirectory 21)
    

# results/valid_phys10_850ns_sec25_cfg3
# Average FID: 116.64258999892782200000
# Maximum FID: 158.34709055470682 (in subdirectory 8)
# Minimum FID: 74.48594131933876 (in subdirectory 21)
    

# results/valid_nophys_850ns_sec25_cfg3
# Average FID: 121.01403280795305800000
# Maximum FID: 158.0101492488743 (in subdirectory 8)
# Minimum FID: 66.07767391397675 (in subdirectory 21)
# 
    

# results/valid_nophys_700ns_sec25_cfg3
# Average FID: 126.68570900910456200000
# Maximum FID: 152.91827941960412 (in subdirectory 19)
# Minimum FID: 85.9735640502186 (in subdirectory 21)


# results/valid_phys10th_700ns_sec25_cfg3
# Average FID: 121.44642767851204200000
# Maximum FID: 157.03153395959282 (in subdirectory 8)
# Minimum FID: 68.88302720806232 (in subdirectory 21)
    

# results/valid_phys10th_850ns_sec45_cfg3
# Average FID: 121.32844268069473400000
# Maximum FID: 166.5842583564771 (in subdirectory 8)
# Minimum FID: 80.04399699214429 (in subdirectory 21)
    

# results/valid_phys10th_850ns_sec2x9plus7_cfg3
# Average FID: 120.12624281349848000000
# Maximum FID: 149.85396754629986 (in subdirectory 8)
# Minimum FID: 104.36485274546402 (in subdirectory 21)
    

# results/valid_phys10th_850ns_sec1x9plus6_cfg3
# Average FID: 119.38068350324149200000
# Maximum FID: 141.93252471336214 (in subdirectory 8)
# Minimum FID: 106.90714367898956 (in subdirectory 19)
# Variance of FID: 204.39459466774626973953
    

# results/valid_phys10th_850ns_sec15_cfg3
# Average FID: 127.24349146203107600000
# Maximum FID: 173.59695644953578 (in subdirectory 8)
# Minimum FID: 96.65418349280148 (in subdirectory 21)
# Variance of FID: 963.29571732171304586978


# results/valid_nophys_850ns_sec2x9plus7_cfg3
# Average FID: 127.39009629653249800000
# Maximum FID: 149.29243630939771 (in subdirectory 19)
# Minimum FID: 94.02488295847405 (in subdirectory 21)
# Variance of FID: 501.54382244412537611716
    

# results/valid_phys10th_850ns_sec2x9plus22_cfg3
# Average FID: 120.79870209555850800000
# Maximum FID: 138.88988199600928 (in subdirectory 8)
# Minimum FID: 101.34328345845154 (in subdirectory 21)
# Variance of FID: 256.98616015699719554552
    

# results/valid_phys10th_850ns_sec1x9plus6_cfg5
# Average FID: 109.37514232358729600000
# Maximum FID: 141.26033079610346 (in subdirectory 8)
# Minimum FID: 79.70565502076501 (in subdirectory 19)
# Variance of FID: 493.67966054827874012330


# results/valid_phys10th_850ns_sec15_cfg5
# Average FID: 118.99652598175237200000
# Maximum FID: 177.99923588262186 (in subdirectory 8)
# Minimum FID: 88.52929483622614 (in subdirectory 3)
# Variance of FID: 1320.97261127278712923768
    

# results/valid_phys10th_850ns_sec1x9plus6_cfg7
# Average FID: 110.02264303761085000000
# Maximum FID: 139.9549688561478 (in subdirectory 8)
# Minimum FID: 94.59962608688411 (in subdirectory 19)
# Variance of FID: 369.13008468275739898155
    

# results/valid_phys10th_850ns_sec1x9plus6_cfg6
# Average FID: 107.83916760386820400000
# Maximum FID: 144.69223531468563 (in subdirectory 8)
# Minimum FID: 80.75562093081973 (in subdirectory 19)
# Variance of FID: 609.20258366116697385077