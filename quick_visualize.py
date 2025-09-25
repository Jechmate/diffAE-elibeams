import numpy as np
import matplotlib.pyplot as plt
import os
import glob

def quick_plot_samples(run_name="edm_1d_spectrum_256pts", results_dir="results"):
    """
    Quick visualization of the latest samples from a training run.
    """
    # Look for sample files
    pattern = os.path.join(results_dir, run_name, "*.npy")
    sample_files = glob.glob(pattern)
    
    if not sample_files:
        print(f"No .npy files found in {os.path.join(results_dir, run_name)}")
        return
    
    print(f"Found {len(sample_files)} sample files:")
    for f in sorted(sample_files):
        print(f"  - {os.path.basename(f)}")
    
    # Load the final sample or the latest one
    final_sample_path = os.path.join(results_dir, run_name, "final_sample.npy")
    if os.path.exists(final_sample_path):
        latest_file = final_sample_path
        print(f"\nUsing final sample: {os.path.basename(latest_file)}")
    else:
        # Sort by epoch number and take the latest
        epoch_files = [f for f in sample_files if "epoch" in f]
        if epoch_files:
            latest_file = sorted(epoch_files, 
                               key=lambda x: int(x.split('_')[-1].split('.')[0]))[-1]
            print(f"\nUsing latest epoch sample: {os.path.basename(latest_file)}")
        else:
            latest_file = sample_files[0]
            print(f"\nUsing: {os.path.basename(latest_file)}")
    
    # Load and analyze the samples
    samples = np.load(latest_file)
    print(f"\nSample shape: {samples.shape}")
    
    # Handle different tensor shapes
    if samples.ndim == 3:  # (n_samples, channels, length)
        n_samples, n_channels, length = samples.shape
        print(f"Found {n_samples} samples with {n_channels} channels and {length} points each")
        spectra = samples[:, 0, :]  # Take first channel
    elif samples.ndim == 2:  # (n_samples, length) or (channels, length)
        if samples.shape[0] <= 10:  # Assume first dim is samples
            n_samples, length = samples.shape
            spectra = samples
        else:  # Assume (channels, length)
            n_channels, length = samples.shape
            n_samples = 1
            spectra = samples[0:1, :]
        print(f"Found {n_samples} samples with {length} points each")
    else:  # 1D array
        length = len(samples)
        n_samples = 1
        spectra = samples.reshape(1, -1)
        print(f"Found 1 sample with {length} points")
    
    # Create energy axis (you may want to adjust this based on your actual data)
    energy_axis = np.linspace(0, 100, length)  # Example: 0-100 eV
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Generated Spectra Analysis - {run_name}', fontsize=16)
    
    # Plot 1: All samples overlaid
    ax1 = axes[0, 0]
    for i in range(min(n_samples, 5)):  # Limit to 5 samples for clarity
        ax1.plot(energy_axis, spectra[i], linewidth=1.5, label=f'Sample {i+1}', alpha=0.8)
    ax1.set_xlabel('Energy (eV)')
    ax1.set_ylabel('Intensity')
    ax1.set_title(f'All Samples (showing first {min(n_samples, 5)})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: First sample detailed
    ax2 = axes[0, 1]
    ax2.plot(energy_axis, spectra[0], 'b-', linewidth=2)
    ax2.set_xlabel('Energy (eV)')
    ax2.set_ylabel('Intensity')
    ax2.set_title('First Sample (Detailed)')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Intensity distribution
    ax3 = axes[1, 0]
    all_intensities = spectra.flatten()
    ax3.hist(all_intensities, bins=50, alpha=0.7, edgecolor='black')
    ax3.set_xlabel('Intensity')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Intensity Distribution')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Mean and std across samples (if multiple samples)
    ax4 = axes[1, 1]
    if n_samples > 1:
        mean_spectrum = np.mean(spectra, axis=0)
        std_spectrum = np.std(spectra, axis=0)
        
        ax4.plot(energy_axis, mean_spectrum, 'r-', linewidth=2, label='Mean')
        ax4.fill_between(energy_axis, 
                        mean_spectrum - std_spectrum, 
                        mean_spectrum + std_spectrum, 
                        alpha=0.3, label='±1 std')
        ax4.set_xlabel('Energy (eV)')
        ax4.set_ylabel('Intensity')
        ax4.set_title(f'Mean ± Std ({n_samples} samples)')
        ax4.legend()
    else:
        ax4.plot(energy_axis, spectra[0], 'g-', linewidth=2)
        ax4.set_xlabel('Energy (eV)')
        ax4.set_ylabel('Intensity')
        ax4.set_title('Single Sample')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Print statistics
    print(f"\nSpectrum Statistics:")
    print(f"  Intensity range: [{np.min(spectra):.4f}, {np.max(spectra):.4f}]")
    print(f"  Mean intensity: {np.mean(spectra):.4f}")
    print(f"  Std intensity: {np.std(spectra):.4f}")
    if n_samples > 1:
        print(f"  Inter-sample std: {np.mean(np.std(spectra, axis=0)):.4f}")
    
    plt.show()
    
    return samples, energy_axis

def plot_training_evolution(run_name="edm_1d_spectrum_256pts", results_dir="results"):
    """
    Plot how the generated spectra evolve during training.
    """
    # Find all epoch samples
    pattern = os.path.join(results_dir, run_name, "sample_epoch_*.npy")
    epoch_files = glob.glob(pattern)
    
    if len(epoch_files) < 2:
        print(f"Need at least 2 epoch samples to show evolution. Found {len(epoch_files)}")
        return
    
    # Sort by epoch number
    epoch_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    
    # Select a subset of files for visualization (max 6)
    if len(epoch_files) > 6:
        indices = np.linspace(0, len(epoch_files)-1, 6, dtype=int)
        selected_files = [epoch_files[i] for i in indices]
    else:
        selected_files = epoch_files
    
    print(f"Plotting evolution using {len(selected_files)} epochs")
    
    # Create subplot
    n_files = len(selected_files)
    cols = 3
    rows = (n_files + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
    if rows == 1:
        axes = axes.reshape(1, -1) if n_files > 1 else [axes]
    
    for i, file_path in enumerate(selected_files):
        row = i // cols
        col = i % cols
        ax = axes[row, col] if rows > 1 else axes[col]
        
        # Extract epoch number
        epoch = int(file_path.split('_')[-1].split('.')[0])
        
        # Load sample
        sample = np.load(file_path)
        
        # Handle shape
        if sample.ndim == 3:
            spectrum = sample[0, 0, :]  # First sample, first channel
        elif sample.ndim == 2:
            spectrum = sample[0, :] if sample.shape[0] <= 4 else sample[0, :]
        else:
            spectrum = sample
        
        # Create energy axis
        energy_axis = np.linspace(0, 100, len(spectrum))
        
        ax.plot(energy_axis, spectrum, 'b-', linewidth=1.5)
        ax.set_title(f'Epoch {epoch}')
        ax.set_xlabel('Energy (eV)')
        ax.set_ylabel('Intensity')
        ax.grid(True, alpha=0.3)
        
        # Set consistent y-limits for comparison
        ax.set_ylim([spectrum.min() - 0.1*abs(spectrum.min()), 
                    spectrum.max() + 0.1*abs(spectrum.max())])
    
    # Hide empty subplots
    for i in range(n_files, rows * cols):
        row = i // cols
        col = i % cols
        ax = axes[row, col] if rows > 1 else axes[col]
        ax.set_visible(False)
    
    plt.suptitle(f'Training Evolution - {run_name}', fontsize=16)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("Quick Spectrum Visualization")
    print("============================")
    
    # Default visualization
    samples, energy = quick_plot_samples()
    
    # Ask if user wants to see training evolution
    response = input("\nShow training evolution? (y/n): ").lower().strip()
    if response in ['y', 'yes']:
        plot_training_evolution()
