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

args.dataset_path = r"train"

dataloader = get_data(args)

diff = Diffusion(device="cpu", img_height=args.image_height, img_width=args.image_width, noise_steps=200)


image = next(iter(dataloader))[0]
t = torch.Tensor([24, 49, 74, 99, 124, 149, 174, 199]).long()

noised_image, _ = diff.noise_images(image, t)
save_image(noised_image.add(1).mul(0.5), "noise.jpg")
