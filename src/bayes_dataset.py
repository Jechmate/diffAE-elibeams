import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

class BayesDataset(Dataset):
    def __init__(self, data_dir, params_file, transform=None, normalize=False, features=["n_e (1e19 cm^-3)", "tau_0 (fs)", "w_0 (um)", "a_0"]):
        """
        Dataset for loading 1D intensity data from .npy files with experimental parameters.
        
        Args:
            data_dir (str): Directory containing the .npy files
            params_file (str): Path to the io.csv file containing experimental parameters
            transform (callable, optional): Optional transform to be applied to the data
            normalize (bool): Whether to normalize the intensity values to [0, 1]
        """
        self.data_dir = data_dir
        self.transform = transform
        self.normalize = normalize
        
        # Load parameters from CSV
        self.params_df = pd.read_csv(params_file, sep=';')[features]
        
        # Load energy axis (same for all data points)
        self.energy = np.load(os.path.join(data_dir, 'ene.npy'))
        
        # Get list of intensity files
        self.file_list = [f'a{i}.npy' for i in range(1, 19)]  # a1.npy to a18.npy
        
        # Calculate normalization values if needed
        if normalize:
            max_intensity = float('-inf')
            for file in self.file_list:
                intensity = np.load(os.path.join(data_dir, file))
                max_intensity = max(max_intensity, intensity.max())
            self.max_intensity = max_intensity
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Load intensity data
        filename = self.file_list[idx]
        intensity = np.load(os.path.join(self.data_dir, filename))
        
        # Convert to tensor
        intensity = torch.tensor(intensity, dtype=torch.float32)
        energy = torch.tensor(self.energy, dtype=torch.float32)
        
        # Get corresponding parameters
        params = self.params_df.iloc[idx]
        settings = torch.tensor(self.features, dtype=torch.float32)
        
        # Normalize intensity if requested
        if self.normalize:
            intensity = intensity / self.max_intensity
            
        # Apply any additional transforms
        if self.transform:
            intensity = self.transform(intensity)
            
        return {
            'intensity': intensity,
            'energy': energy,
            'settings': settings
        }

def get_bayes_dataloader(
    data_dir,
    params_file,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    transform=None,
    normalize=False,
):
    """
    Creates a DataLoader for the Bayes dataset.
    
    Args:
        data_dir (str): Directory containing the .npy files
        params_file (str): Path to the io.csv file
        batch_size (int): Batch size for the dataloader
        shuffle (bool): Whether to shuffle the data
        num_workers (int): Number of worker processes for data loading
        transform (callable, optional): Optional transform to be applied
        normalize (bool): Whether to normalize the intensity values
        
    Returns:
        DataLoader: PyTorch DataLoader for the dataset
    """
    dataset = BayesDataset(
        data_dir=data_dir,
        params_file=params_file,
        transform=transform,
        normalize=normalize
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )

if __name__ == "__main__":
    # Example usage
    data_dir = "data/bayes_data"
    params_file = "data/bayes_data/io.csv"
    
    # Create dataset
    dataset = BayesDataset(data_dir, params_file)
    print(f"Dataset size: {len(dataset)}")
    
    # Get a sample
    sample = dataset[0]
    print(f"Sample intensity shape: {sample['intensity'].shape}")
    print(f"Sample energy shape: {sample['energy'].shape}")
    print(f"Sample settings shape: {sample['settings'].shape}")
    
    # Create dataloader
    dataloader = get_bayes_dataloader(data_dir, params_file, batch_size=4)
    
    # Example iteration
    for batch in dataloader:
        print(f"Batch intensity shape: {batch['intensity'].shape}")
        print(f"Batch energy shape: {batch['energy'].shape}")
        print(f"Batch settings shape: {batch['settings'].shape}")
        break
