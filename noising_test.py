import torch
from torchvision.utils import save_image
from ddpm_conditional import Diffusion
from utils import get_data
import argparse

parser = argparse.ArgumentParser()
args = parser.parse_args()
args.batch_size = 1  # 5
args.image_width = 128
args.image_height = 64
args.dataset_path = r"with_gain"
args.csv_path = "params.csv"
args.features = ["E","P","ms"]
args.exclude = []# ['train/19']

dataloader = get_data(args)

diff = Diffusion(device="cpu", img_height=args.image_height, img_width=args.image_width, noise_steps=700, beta_end=0.003, beta_start=0.0001)


image = next(iter(dataloader))['image']
t = torch.Tensor([0, 99, 199, 299, 399, 499, 599, 699]).long()

noised_image, _ = diff.noise_images(image, t)
save_image(noised_image.add(1).mul(0.5), "noise.jpg")
