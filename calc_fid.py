import os
import re
import csv
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from scipy import linalg
from tqdm import tqdm
from vae import SimpleVAE

# Define transformations: Normalize images from [0, 255] to [-1, 1]
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=0.5, std=0.5)
])

# Dataset class to load images from a folder
class ImageFolderDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        self.image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.png')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('L')  # Convert to grayscale; adjust if images are RGB
        if self.transform:
            image = self.transform(image)
        return image

# Function to calculate FID
def calculate_fid(mu1, sigma1, mu2, sigma2):
    """Compute the Frechet Distance between two Gaussian distributions."""
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid

# Function to extract latent features using VAE
def get_latent_features(model, dataloader, device):
    model.eval()
    latent_features = []
    with torch.no_grad():
        for images in tqdm(dataloader, desc="Extracting Latent Features"):
            images = images.to(device)
            mu, _ = model.encode(images)
            latent_features.append(mu.cpu().numpy())
    latent_features = np.concatenate(latent_features, axis=0)
    return latent_features

# Function to parse folder name to extract parameters
def parse_folder_name(folder_name):
    """
    Parse the folder name to extract the schedule, sections, and cfg_scale.
    Handles patterns like:
    - 'cossched_sec10_cfg1'
    - 'cossched_seccos10_cfg1'
    """
    # Updated pattern to handle 'secX' or 'seccosX'
    pattern = r'(\w+)_sec(cos)?(\d+)_cfg(\d+)'
    match = re.match(pattern, folder_name)
    if match:
        physics, cos_flag, sections, cfg_scale = match.groups()
        if cos_flag:  # If 'cos' is present in the section name
            sections = 'cos' + sections
        return {
            'physics': physics,
            'sections': sections,
            'cfg_scale': cfg_scale
        }
    return None

# Main function to calculate the average FID across all subfolders and save to CSV
def calculate_average_fid(vae_model, results_root, validation_root, device, csv_filename):
    # Prepare to write to CSV file
    with open(csv_filename, mode='w', newline='') as csv_file:
        fieldnames = ['Physics', 'Sections', 'CFG', 'FID']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()  # Write the header

        # List all folders in results root
        results_folders = [f for f in os.listdir(results_root) if os.path.isdir(os.path.join(results_root, f))]

        for result_folder in results_folders:
            result_path = os.path.join(results_root, result_folder)
            params = parse_folder_name(result_folder)

            if not params:
                print(f"Skipping folder {result_folder}, invalid format")
                continue
            
            # Iterate through subfolders (3, 8, 11, 19, 21)
            subfolder_fids = []
            for subfolder in ['3', '8', '11', '19', '21']:
                result_subfolder = os.path.join(result_path, subfolder)
                validation_subfolder = os.path.join(validation_root, subfolder)

                if not os.path.exists(result_subfolder) or not os.path.exists(validation_subfolder):
                    print(f"Skipping comparison for {subfolder}, folder not found")
                    continue

                # Load the datasets
                result_dataset = ImageFolderDataset(result_subfolder, transform=transform)
                validation_dataset = ImageFolderDataset(validation_subfolder, transform=transform)

                result_loader = DataLoader(result_dataset, batch_size=64, shuffle=False)
                validation_loader = DataLoader(validation_dataset, batch_size=64, shuffle=False)

                # Extract latent features from the VAE
                result_features = get_latent_features(vae_model, result_loader, device)
                validation_features = get_latent_features(vae_model, validation_loader, device)

                # Compute FID
                mu1, sigma1 = np.mean(result_features, axis=0), np.cov(result_features, rowvar=False)
                mu2, sigma2 = np.mean(validation_features, axis=0), np.cov(validation_features, rowvar=False)
                fid = calculate_fid(mu1, sigma1, mu2, sigma2)
                subfolder_fids.append(fid)

            if subfolder_fids:
                avg_fid = np.mean(subfolder_fids)
                print(f"Folder: {result_folder}, Avg FID: {avg_fid}, Params: {params}")
                
                # Write the results to the CSV file
                writer.writerow({
                    'Physics': params['physics'],
                    'Sections': params['sections'],
                    'CFG': params['cfg_scale'],
                    'FID': avg_fid
                })

# Assuming your trained VAE is loaded here
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load your VAE model
    vae = SimpleVAE(latent_dim=64).to(device)
    vae.load_state_dict(torch.load('models/vae_final.pth', map_location=device))

    # Paths to results and validation folders
    results_root = 'results_gaindata_batch4_600e'  # Replace with actual path
    validation_root = 'data/with_gain'  # Replace with actual path
    csv_filename = 'fid__paper_results.csv'  # CSV file to store the results

    # Calculate average FID and save to CSV
    calculate_average_fid(vae, results_root, validation_root, device, csv_filename)