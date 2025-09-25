import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append('.')

# Import your modules
from src.modules_1d import EDMPrecond
from src.diffusion import EdmSampler
from optimize_conditional import create_energy_axis
from scipy import stats

def test_single_exclusion_model(model_name="edm_4kepochs_exclude_3", device='cuda'):
    """Test evaluation on a single exclusion model."""
    
    print(f"Testing model: {model_name}")
    print(f"Using device: {device}")
    
    # Configuration
    config = {
        'device': device,
        'resolution': 256,
        'settings_dim': 3,
        'num_steps': 30,
        'cfg_scale': 3.0,
    }
    
    # Extract excluded experiment number
    excluded_exp = int(model_name.split('_')[-1])
    print(f"Excluded experiment: {excluded_exp}")
    
    # Load model
    model_path = f"models/{model_name}"
    print(f"Loading model from: {model_path}")
    
    # Initialize model
    model = EDMPrecond(
        resolution=config['resolution'],
        settings_dim=config['settings_dim'],
        sigma_min=0,
        sigma_max=float('inf'),
        sigma_data=0.112,
        model_type='UNet_conditional',
        device=config['device']
    ).to(config['device'])
    
    # Load checkpoint
    checkpoint_path = f"{model_path}/ema_ckpt_final.pt"
    if not os.path.exists(checkpoint_path):
        # Try alternative checkpoint names
        alternatives = ["ema_ckpt.pt"]
        epoch_files = [f for f in os.listdir(model_path) if f.startswith('ema_ckpt_epoch_')]
        if epoch_files:
            epoch_files.sort(key=lambda x: int(x.split('_')[3].split('.')[0]))
            alternatives.insert(0, epoch_files[-1])
        
        for alt in alternatives:
            alt_path = f"{model_path}/{alt}"
            if os.path.exists(alt_path):
                checkpoint_path = alt_path
                break
    
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=config['device'])
    model.load_state_dict(checkpoint)
    model.eval()
    
    # Create sampler
    sampler = EdmSampler(net=model, num_steps=config['num_steps'])
    
    # Get experiment settings
    params_df = pd.read_csv("data/params.csv")
    exp_row = params_df[params_df['experiment'] == excluded_exp].iloc[0]
    experiment_settings = [float(exp_row['E']), float(exp_row['P']), float(exp_row['ms'])]
    print(f"Experiment settings: E={experiment_settings[0]}, P={experiment_settings[1]}, ms={experiment_settings[2]}")
    
    # Load original spectra
    original_data_path = f"data/spectra/{excluded_exp}"
    csv_files = list(Path(original_data_path).glob("*.csv"))
    original_spectra = []
    
    print(f"Loading original spectra from {original_data_path}")
    for csv_file in csv_files[:5]:  # Load only first 5 for testing
        df = pd.read_csv(csv_file)
        original_spectra.append(df['intensity'].values)
    
    original_spectra = np.array(original_spectra)
    print(f"Loaded {len(original_spectra)} original spectra, shape: {original_spectra.shape}")
    
    # Generate same number of samples
    print(f"Generating {len(original_spectra)} samples...")
    
    with torch.no_grad():
        settings_tensor = torch.tensor(experiment_settings, dtype=torch.float32).reshape(1, -1).to(config['device'])
        
        samples = sampler.sample(
            resolution=config['resolution'],
            device=config['device'],
            settings=settings_tensor,
            n_samples=len(original_spectra),
            cfg_scale=config['cfg_scale'],
            settings_dim=config['settings_dim'],
            smooth_output=True,
            smooth_kernel_size=9,
            smooth_sigma=2.0
        )
        
        generated_spectra = samples.cpu().numpy()
        if generated_spectra.ndim == 3:
            generated_spectra = generated_spectra[:, 0, :]
    
    print(f"Generated spectra shape: {generated_spectra.shape}")
    
    # Calculate Wasserstein distance bin-by-bin
    print("Calculating Wasserstein distances...")
    
    resolution = min(original_spectra.shape[1], generated_spectra.shape[1])
    distances = []
    
    for bin_idx in range(0, resolution, 10):  # Sample every 10th bin for speed
        original_bin = original_spectra[:, bin_idx]
        generated_bin = generated_spectra[:, bin_idx]
        
        # Skip very low intensity bins
        if np.mean(original_bin) < 0.01 and np.mean(generated_bin) < 0.01:
            continue
        
        wasserstein_dist = stats.wasserstein_distance(original_bin, generated_bin)
        distances.append(wasserstein_dist)
        
        if len(distances) <= 5:  # Print first few
            print(f"  Bin {bin_idx}: Wasserstein distance = {wasserstein_dist:.6f}")
    
    avg_distance = np.mean(distances)
    print(f"\nAverage Wasserstein distance (sampled bins): {avg_distance:.6f}")
    
    # Create a simple comparison plot
    energy_axis = create_energy_axis(resolution)
    
    plt.figure(figsize=(12, 8))
    
    # Plot average spectra
    plt.subplot(2, 2, 1)
    orig_mean = np.mean(original_spectra, axis=0)
    gen_mean = np.mean(generated_spectra, axis=0)
    
    plt.plot(energy_axis, orig_mean, 'b-', label='Original (avg)', linewidth=2)
    plt.plot(energy_axis, gen_mean, 'r--', label='Generated (avg)', linewidth=2)
    plt.xlabel('Energy (MeV)')
    plt.ylabel('Intensity')
    plt.title(f'Average Spectra - Exp {excluded_exp}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot individual samples
    plt.subplot(2, 2, 2)
    for i in range(min(3, len(original_spectra))):
        plt.plot(energy_axis, original_spectra[i], 'b-', alpha=0.6, linewidth=1)
        plt.plot(energy_axis, generated_spectra[i], 'r--', alpha=0.6, linewidth=1)
    plt.xlabel('Energy (MeV)')
    plt.ylabel('Intensity')
    plt.title('Sample Comparisons')
    plt.grid(True, alpha=0.3)
    
    # Plot intensity distributions
    plt.subplot(2, 2, 3)
    plt.hist(original_spectra.flatten(), bins=30, alpha=0.6, label='Original', density=True, color='blue')
    plt.hist(generated_spectra.flatten(), bins=30, alpha=0.6, label='Generated', density=True, color='red')
    plt.xlabel('Intensity')
    plt.ylabel('Density')
    plt.title('Intensity Distributions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot standard deviations
    plt.subplot(2, 2, 4)
    orig_std = np.std(original_spectra, axis=0)
    gen_std = np.std(generated_spectra, axis=0)
    
    plt.plot(energy_axis, orig_std, 'b-', label='Original std', linewidth=2)
    plt.plot(energy_axis, gen_std, 'r--', label='Generated std', linewidth=2)
    plt.xlabel('Energy (MeV)')
    plt.ylabel('Standard Deviation')
    plt.title('Variability Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.suptitle(f'Model: {model_name} - Test Results', fontsize=14)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(f"test_{model_name}_results.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\nTest completed successfully!")
    print(f"Results saved as test_{model_name}_results.png")
    
    return avg_distance

if __name__ == "__main__":
    # Test with one model first
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # You can change this to test different models
    test_model = "edm_4kepochs_exclude_3"
    
    print("Testing single exclusion model evaluation...")
    try:
        distance = test_single_exclusion_model(test_model, device)
        print(f"Success! Average Wasserstein distance: {distance:.6f}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc() 