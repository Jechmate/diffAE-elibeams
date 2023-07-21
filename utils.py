import os
import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
import cv2


def plot_images(images):
    n = len(images)
    rows = (n + 4) // 5
    cols = min(n, 5)

    plt.figure(figsize=(32, 32))
    plt.subplots_adjust(wspace=0.1, hspace=0.1)

    for i in range(n):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(images[i].cpu().permute(1, 2, 0).numpy())
        plt.axis("off")
        plt.title(f"{i}", size=12)

    plt.show()


def save_samples(images, folder="samples"):
    ndarr = images.permute(0, 2, 3, 1).to('cpu').numpy()
    for i, im in enumerate(ndarr):
        cv2.imwrite(folder + "/" + str(i) + ".png", im)


def save_images(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    im = Image.fromarray(ndarr)
    im.save(path)


def get_data(args):
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Grayscale(), # TODO hope this doesnt do any funny business with the data - ImageFolder loads 3 channels by default
        torchvision.transforms.Resize((args.image_height, args.image_width)),  # args.image_size + 1/4 *args.image_size
        # torchvision.transforms.RandomResizedCrop((args.image_height, args.image_width), scale=(0.8, 1.0)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(0.5, 0.5)
    ])
    dataset = torchvision.datasets.ImageFolder(args.dataset_path, transform=transforms)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    return dataloader


def setup_logging(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)
