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
import src.dataset as dataset
import scipy
import torchvision.transforms.functional as f

class ExperimentDataset(Dataset):
    def __init__(self, csv_file="params.csv", root_dir="train", transform=None, features=["E","perc_N","P","gain","ms"], exclude=[]):
        """
        Arguments:
            csv_file (string): Path to the csv file with settings.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.features = features
        self.settings = pd.read_csv(csv_file, engine='python')[features]# .apply(self.min_max_norm)
        self.root_dir = root_dir
        self.exclude = exclude
        self.file_list = self.get_list_of_img()
        self.transform = transform

    def min_max_norm(self, col):
        return (col - col.min()) / (col.max() - col.min())

    def get_list_of_img(self, regex="*.png"):
        files = []
        for dirpath, _, _ in os.walk(self.root_dir):
            if dirpath in self.exclude:
                print("Excluding " + dirpath)
                continue
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
        exp_index = int(filename.split('/')[-2])
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
    
    
def plot_average_image_pairs(root_folder, acquisition_time_ms, electron_pointing_pixel=62):
    subfolders = sorted([f.path for f in os.scandir(root_folder) if f.is_dir()])
    n = len(subfolders)
    fig, axs = plt.subplots(n, 2, figsize=(15, 4*n))
    fig.subplots_adjust(hspace=0.35)  # Increase the space between rows
    fig.subplots_adjust(wspace=0.1)  # Decrease the space between columns
    for i, subfolder in enumerate(subfolders):
        images = []
        for filename in os.listdir(subfolder):
            if filename.endswith(".png"):
                im = cv2.imread(os.path.join(subfolder, filename), cv2.IMREAD_UNCHANGED)
                images.append(im)
        avg_im = np.mean(images, axis=0)
        deflection_MeV, spectrum_calibrated = dataset.get_1d(avg_im/255, acquisition_time_ms, electron_pointing_pixel=electron_pointing_pixel)

        axs[i, 1].plot(deflection_MeV, spectrum_calibrated)  # plot without fit
        axs[i, 1].set_title('Reconstructed Spectrum')
        axs[i, 1].set_ylabel('Spectral Intensity (pA/MeV)')
        axs[i, 1].set_xlabel('Energy (MeV)')
        axs[i, 1].set_xlim([2, 20])
        axs[i, 0].imshow(avg_im, vmin=0, vmax=255)
        axs[i, 0].set_title(os.path.basename(subfolder))
    plt.show()


def find_ticks(deflection_MeV, beam_point_x, beam_point_y, pixel_in_mrad, energy_levels, ranges):
    # Find the index of the first occurrence within each energy level range
    ticks = {}
    for energy, (low, high) in zip(energy_levels, ranges):
        ticks[f'tick{energy}MeV'] = next((i for i, val in enumerate(deflection_MeV[beam_point_x:], start=beam_point_x) if low < val < high), None)
    # Calculate y-ticks
    ticks['tick_10mrad_px'] = beam_point_y - round(10 / pixel_in_mrad)
    ticks['tick0mrad_px'] = beam_point_y
    ticks['tick10mrad_px'] = beam_point_y + round(10 / pixel_in_mrad)
    return ticks


def plot_image_pairs(images, acquisition_time_ms, beam_point_x, beam_point_y, energy, pressure, xlim=[2, 20], model=1):
    def get_y_lims_within_xlim(x, y, xlim):
        """Find the min and max y-values within the specified x-limits."""
        within_xlim = (x >= xlim[0]) & (x <= xlim[1])
        y_within_xlim = y[within_xlim]
        return [np.min(y_within_xlim), np.max(y_within_xlim)] if y_within_xlim.size > 0 else [np.min(y), np.max(y)]

    n = len(images)
    pixel_in_mrad = 0.3653
    energy_levels = [100, 30, 15, 10, 8, 5, 3]  # Removed 40 and 20
    ranges = [(70, 101), (20, 31), (12, 15.5), (8, 10.5), (6, 8.2), (4.8, 5.2), (2.9, 3.2)]  # Adjusted ranges
    
    fig, axs = plt.subplots(n, 2, figsize=(15, 4*n))
    fig.subplots_adjust(hspace=0.35, wspace=0.15, top=0.98)
    title = fig.suptitle(f"Energy: {energy} mJ, Pressure: {pressure} bar, Acquisition time: {acquisition_time_ms} ms, Model: {model}",  fontsize=16)
    title.set_position([0.5, 1])

    for i in range(n):
        im = images[i].cpu().permute(1, 2, 0).numpy()
        deflection_MeV, spectrum_calibrated = dataset.get_1d(im/255, acquisition_time_ms, electron_pointing_pixel=beam_point_x)  # Using a local function
        
        # Find ticks for the current image
        ticks = find_ticks(deflection_MeV, beam_point_x, beam_point_y, pixel_in_mrad, energy_levels, ranges)
        
        # Plot the spectrum
        axs[i, 1].plot(deflection_MeV, spectrum_calibrated)
        axs[i, 1].set_title('Reconstructed Spectrum', fontsize=12)
        axs[i, 1].set_ylabel('Spectral Intensity (pA/MeV)', fontsize=12)
        axs[i, 1].set_xlabel('Energy [MeV]', fontsize=12)
        axs[i, 1].set_xlim(xlim)
        y_lims = get_y_lims_within_xlim(deflection_MeV, spectrum_calibrated, xlim)
        axs[i, 1].set_ylim(y_lims)

        # Plot the image
        axs[i, 0].imshow(im, vmin=0, vmax=255, cmap='inferno')
        axs[i, 0].set_title(f"Image {i}")

        # Set y-axis ticks for mrad values
        axs[i, 0].set_yticks([ticks['tick_10mrad_px'], ticks['tick0mrad_px'], ticks['tick10mrad_px']])
        axs[i, 0].set_yticklabels(['-10', '0', '10'])
        axs[i, 0].set_ylabel('Angle [mrad]')

        # Set x-axis ticks for MeV values
        mev_ticks = [tick for key, tick in ticks.items() if 'MeV' in key and tick is not None]
        axs[i, 0].set_xticks(mev_ticks)
        axs[i, 0].set_xticklabels([key.split('tick')[1].replace('MeV', '') for key in ticks if 'MeV' in key and ticks[key] is not None])
        axs[i, 0].set_xlabel('Energy [MeV]')

    plt.show()




def stitch_images(directory):
    # Get the list of image files in the directory
    image_files = sorted([os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.png') or f.endswith('.jpg')])
    image_files.sort(key=lambda f: int(os.path.basename(f).split('_')[0]))

    # Open the images and get their sizes
    images = [Image.open(f) for f in image_files]
    widths, heights = zip(*(i.size for i in images))

    # Create a new image with the combined height of all images
    total_height = sum(heights)
    max_width = max(widths)
    stitched_image = Image.new('RGB', (max_width, total_height))

    # Paste the images into the stitched image
    y_offset = 0
    for img in images:
        stitched_image.paste(img, (0, y_offset))
        y_offset += img.size[1]

    # Save the stitched image
    stitched_image.save(os.path.join(directory, 'stitched_image.png'))


def save_samples(images, folder="samples", start_index=0):
    ndarr = images.permute(0, 2, 3, 1).to('cpu').numpy()
    indexes = range(start_index, start_index + len(ndarr))
    for i, im in zip(indexes, ndarr):
        cv2.imwrite(folder + "/" + str(i) + ".png", im)


def save_images(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    im = Image.fromarray(ndarr)
    im.save(path)


def get_data(args):
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize((args.image_height, args.image_width), antialias=True),  # args.image_size + 1/4 *args.image_size
        torchvision.transforms.Normalize(0.5, 0.5)
    ])
    dataset = ExperimentDataset(args.csv_path, args.dataset_path, transform=transforms, features=args.features, exclude=args.exclude)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    return dataloader


def setup_logging(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)


def deflection_calc(batch_size, hor_image_size, electron_pointing_pixel):
    pixel_in_mm = 0.137 
    deflection_MeV = torch.zeros((batch_size, hor_image_size))
    deflection_mm = torch.zeros((batch_size, hor_image_size))
    mat = scipy.io.loadmat('data/Deflection_curve_Mixture_Feb28.mat')
    for i in range(hor_image_size):
        if i <= electron_pointing_pixel:
            deflection_mm[:, i] = 0
        else:
            deflection_mm[:, i] = (i - electron_pointing_pixel) * pixel_in_mm
            
    for i in range(electron_pointing_pixel, hor_image_size):
        xq = deflection_mm[:, i]
        mask = xq > 1
        if mask.any():
            deflection_MeV[mask, i] = torch.from_numpy(scipy.interpolate.interp1d(mat['deflection_curve_mm'][:, 0],
                                                           mat['deflection_curve_MeV'][:, 0],
                                                           kind='linear',
                                                           assume_sorted=False,
                                                           bounds_error=False)(xq[mask]).astype(np.float32))
    return deflection_MeV


def calc_spec(image, electron_pointing_pixel, deflection_MeV, acquisition_time_ms, image_gain=0, resize=None, noise=False, device='cpu'):
    if resize:
        image = f.resize(image, resize, antialias=True)
    image_gain /= 32
    if noise:
        noise = torch.median(torch.stack([image[:, :, int(image.shape[1]*0.9), int(image.shape[2]*0.05)],
                        image[:, :, int(image.shape[1]*0.9), int(image.shape[2]*0.9)],
                        image[:, :, int(image.shape[1]*0.1), int(image.shape[2]*0.9)]], dim=0), dim=(1, 2))
        noise = noise.unsqueeze(1).unsqueeze(2)
        image[image <= noise] = 0
    # acquisition_time_ms = 10
    hor_image_size = image.shape[3]
    batch_size = image.shape[0]
    horizontal_profile = torch.sum(image, dim=(1, 2)).to(device)
    spectrum_in_pixel = torch.zeros((batch_size, hor_image_size)).to(device)
    spectrum_in_MeV = torch.zeros((batch_size, hor_image_size)).to(device)
            
    for j in range(electron_pointing_pixel, hor_image_size):
        spectrum_in_pixel[:, j] = horizontal_profile[:,j]
        with torch.no_grad():
            mask = (deflection_MeV[:, j-1] - deflection_MeV[:, j]) != 0
            spectrum_in_MeV[mask, j] = spectrum_in_pixel[mask, j] / (deflection_MeV[mask, j-1] - deflection_MeV[mask, j])
            spectrum_in_MeV[~torch.isfinite(spectrum_in_MeV)] = 0

    acquisition_time_ms = acquisition_time_ms.reshape(batch_size, 1).repeat(1, hor_image_size).to(device)
    spectrum_calibrated = (spectrum_in_MeV * 3.706) / (acquisition_time_ms*image_gain) if image_gain else (spectrum_in_MeV * 3.706) / acquisition_time_ms
    return deflection_MeV, spectrum_calibrated