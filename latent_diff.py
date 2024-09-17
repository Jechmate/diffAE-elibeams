import argparse
import torch
from src.utils import get_data
from vqvae import VQVAE, make_latent_dataset

# Argument parsing and configuration
parser = argparse.ArgumentParser()
args = parser.parse_args()
args.run_name = "vqvae"
args.epochs = 100
args.noise_steps = 1000
args.phys = True
args.beta_start = 1e-4
args.beta_end = 0.02
args.batch_size = 4
args.image_height = 256
args.image_width = 512
args.real_size = (256, 512)
args.features = ["E", "P", "ms"]
args.dataset_path = r"data/with_gain"
args.csv_path = "data/params.csv"
args.device = "cuda:1" if torch.cuda.is_available() else "cpu"
args.lr = 1e-3
args.exclude = []  # Example: ['train/19']
args.grad_acc = 1
args.sample_freq = 0
args.sample_settings = [32., 15., 15.]
args.sample_size = 8
args.electron_pointing_pixel = 62
args.seed = 42
args.split = False

# Get dataloader
save_path = "data/latent_dataset_vqvae_phys.pth"

# Load the latent dataset from the file
train_loader = torch.load(save_path)

# Initialize model and load state dictionary
device = args.device
model = VQVAE().to(device)
model.load_state_dict(torch.load("final_128_vqvae_model_phys.pth", map_location=device))

# Save the latent dataset to disk
save_path = "data/latent_phys/pth"
make_latent_dataset(model, train_loader, save_path)

model = VQVAE().to(device)
model.load_state_dict(torch.load("final_128_vqvae_model.pth", map_location=device))

# Save the latent dataset to disk
save_path = "data/latent_nophys/pth"
make_latent_dataset(model, train_loader, save_path)
