import torch
from torchvision.utils import save_image
from src.diffusion import GaussianDiffusion, prepare_noise_schedule, DCTBlur
from src.utils import get_data
import argparse
import random


def gaussian_test():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.batch_size = 1  # 5
    args.image_width = 512
    args.image_height = 256
    args.dataset_path = r"data/with_gain"
    args.csv_path = "data/params.csv"
    args.features = ["E","P","ms"]
    args.exclude = []# ['train/19']

    dataloader = get_data(args)

    betas = prepare_noise_schedule(noise_steps=700, beta_end=0.02, beta_start=1e-4)
    diff = GaussianDiffusion(device="cpu", img_height=args.image_height, img_width=args.image_width, betas=betas)


    image = next(iter(dataloader))['image']
    t = torch.Tensor([0, 99, 199, 299, 399, 499, 599, 699]).long()

    noised_image, _ = diff.noise_images(image, t)
    save_image(noised_image.add(1).mul(0.5), "noise.png")


def ihd_test():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.batch_size = 1  # 5
    args.image_width = 512
    args.image_height = 256
    args.dataset_path = r"data/with_gain"
    args.csv_path = "data/params.csv"
    args.features = ["E","P","ms"]
    args.exclude = []# ['train/19']
    dataloader = get_data(args)
    diff = DCTBlur(img_height=256, img_width=512, noise_steps=700, device='cpu')
    sigma = 0.01
    delta = 0.0125
    blur_sigma_max = 64
    blur_sigma_min = 0.5
    diff.prepare_blur_schedule(blur_sigma_max, blur_sigma_min)
    dataset_length = len(dataloader.dataset)
    random_index = random.randint(0, dataset_length - 1)
    random_data_sample = dataloader.dataset[random_index]
    image = random_data_sample['image']
    # image = next(iter(dataloader))['image']
    t = torch.Tensor([0, 99, 199, 299, 399, 499, 599, 699]).long()
    blurred_batch = diff(image, t).float()
    save_image(blurred_batch.add(1).mul(0.5), "blur.png")
    


if __name__ == '__main__':
    ihd_test()