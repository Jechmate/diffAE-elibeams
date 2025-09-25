import numpy as np
from src.spectrum_dataset import SpectrumDataset

# Set paths to your data directory and params file
DATA_DIR = 'data/spectra'
PARAMS_FILE = 'data/params.csv'

# Load the dataset
print('Loading dataset...')
dataset = SpectrumDataset(DATA_DIR, PARAMS_FILE)

all_intensities = []

print('Collecting intensity values...')
for i in range(len(dataset)):
    sample = dataset[i]
    # sample['intensity'] is a torch tensor, convert to numpy
    intensities = sample['intensity'].numpy()
    all_intensities.append(intensities)

# Concatenate all intensities into a single array
all_intensities = np.concatenate(all_intensities)

# Compute statistics
min_intensity = np.min(all_intensities)
max_intensity = np.max(all_intensities)
std_intensity = np.std(all_intensities)
percentiles = [1, 5, 25, 50, 75, 95, 99]
percentile_values = np.percentile(all_intensities, percentiles)

# Print results
print(f"\nIntensity statistics for the entire dataset:")
print(f"  Min: {min_intensity}")
print(f"  Max: {max_intensity}")
print(f"  Std: {std_intensity}")
for p, v in zip(percentiles, percentile_values):
    print(f"  {p}th percentile: {v}") 