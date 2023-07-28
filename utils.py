import os
import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
import matplotlib as mpl
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import cv2
import glob

class ExperimentDataset(Dataset):
    """Face settings dataset."""

    def __init__(self, csv_file="params.csv", root_dir="train", transform=None, features=["E","perc_N","P","gain","ms"]):
        """
        Arguments:
            csv_file (string): Path to the csv file with settings.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.features = features
        self.settings = pd.read_csv(csv_file, engine='python')[features]
        self.root_dir = root_dir
        self.file_list = self.get_list_of_img()
        self.transform = transform

    def get_list_of_img(self, regex="*.png"):
        files = []
        for dirpath, _, _ in os.walk(self.root_dir):
            type_files = glob.glob(os.path.join(dirpath, regex))
            files += type_files
        return sorted(files)
    
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        filename = self.file_list[idx]
        image = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
        exp_index = int(filename.split('/')[1])
        settings = self.settings.iloc[exp_index-1, 0:]
        settings = np.array([settings])
        settings = settings.astype('float32').reshape(-1, len(self.features))
        if self.transform:
            image = self.transform(image)
        to_tens = torchvision.transforms.ToTensor()
        settings = to_tens(settings).squeeze()
        sample = {'image': image, 'settings': settings}

        return sample


def load_images_from_dir(path, num_images):
    images = []
    for i in range(num_images):
        image_path = os.path.join(path, f"{i}.png")
        image = Image.open(image_path)
        images.append(image)
    return images


def plot_images_from_dir(path, num_images):
    images = load_images_from_dir(path, num_images)
    n = len(images)
    rows = (n + 4) // 5
    cols = min(n, 5)

    plt.figure(figsize=(32, 32))
    plt.subplots_adjust(wspace=0.1, hspace=0.1)

    for i in range(n):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(images[i], vmin=0, vmax=255, cmap=mpl.colormaps['viridis'])
        plt.axis("off")
        plt.title(f"{i}", size=12)

    plt.show()


def plot_images(images):
    n = len(images)
    rows = 2
    cols = 4

    plt.figure(figsize=(64, 32))
    plt.subplots_adjust(wspace=0.1, hspace=0.1)

    for i in range(n):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(images[i].cpu().permute(1, 2, 0).numpy(), vmin=0, vmax=255, cmap=mpl.colormaps['viridis'])
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
        # torchvision.transforms.Grayscale(), # TODO hope this doesnt do any funny business with the data
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize((args.image_height, args.image_width)),  # args.image_size + 1/4 *args.image_size
    ])
    dataset = ExperimentDataset(args.csv_path, args.dataset_path, transform=transforms, features=args.features)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    return dataloader


def setup_logging(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)
