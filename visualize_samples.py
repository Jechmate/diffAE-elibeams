import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from pathlib import Path
import argparse

def load_energy_axis(length=256):
    """
    Create a synthetic energy axis for the spectra.
    In practice, you might want to load the actual energy values from your data.
    """
    # Create a synthetic energy range (you can adjust this based on your actual data)
    return np.linspace(0, 100, length)  # Example: 0-100 eV range

def plot_single_spectrum(spectrum, energy_axis=None, title="Generated Spectrum", figsize=(10, 6)):
    """
    Plot a single spectrum.
    
    Args:
        spectrum: 1D numpy array of intensity values
        energy_axis: 1D numpy array of energy values (optional)
        title: Title for the plot
        figsize: Figure size tuple
    """
    if energy_axis is None:
        energy_axis = load_energy_axis(len(spectrum))
    
    plt.figure(figsize=figsize)
    plt.plot(energy_axis, spectrum, 'b-', linewidth=1.5)
    plt.xlabel('Energy (eV)')
    plt.ylabel('Intensity')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return plt.gcf()

def plot_multiple_spectra(spectra, energy_axis=None, title="Generated Spectra", figsize=(12, 8)):
    """
    Plot multiple spectra on the same figure.
    
    Args:
        spectra: 2D numpy array where each row is a spectrum
        energy_axis: 1D numpy array of energy values (optional)
        title: Title for the plot
        figsize: Figure size tuple
    """
    if energy_axis is None:
        energy_axis = load_energy_axis(spectra.shape[-1])
    
    plt.figure(figsize=figsize)
    
    for i, spectrum in enumerate(spectra):
        plt.plot(energy_axis, spectrum, linewidth=1.5, label=f'Sample {i+1}', alpha=0.8)
    
    plt.xlabel('Energy (eV)')
    plt.ylabel('Intensity')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return plt.gcf()

def plot_training_progression(results_dir, run_name, figsize=(15, 10)):
    """
    Plot spectra from different training epochs to show progression.
    
    Args:
        results_dir: Path to results directory
        run_name: Name of the training run
        figsize: Figure size tuple
    """
    sample_files = glob.glob(os.path.join(results_dir, run_name, "sample_epoch_*.npy"))
    sample_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))  # Sort by epoch number
    
    if not sample_files:
        print(f"No sample files found in {os.path.join(results_dir, run_name)}")
        return None
    
    # Load energy axis
    first_sample = np.load(sample_files[0])
    if first_sample.ndim == 3:  # (n_samples, channels, length)
        spectrum_length = first_sample.shape[-1]
    elif first_sample.ndim == 2:  # (channels, length)
        spectrum_length = first_sample.shape[-1]
    else:
        spectrum_length = len(first_sample)
    
    energy_axis = load_energy_axis(spectrum_length)
    
    # Create subplot grid
    n_files = len(sample_files)
    cols = min(4, n_files)
    rows = (n_files + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if rows == 1:
        axes = axes.reshape(1, -1) if n_files > 1 else [axes]
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    
    for i, file_path in enumerate(sample_files):
        row = i // cols
        col = i % cols
        ax = axes[row, col] if rows > 1 else axes[col]
        
        # Extract epoch number from filename
        epoch = int(file_path.split('_')[-1].split('.')[0])
        
        # Load and plot spectrum
        sample = np.load(file_path)
        
        # Handle different sample shapes
        if sample.ndim == 3:  # (n_samples, channels, length)
            spectrum = sample[0, 0, :]  # Take first sample, first channel
        elif sample.ndim == 2:  # (channels, length) or (n_samples, length)
            spectrum = sample[0, :] if sample.shape[0] <= 4 else sample[0, :]  # Assume first dim is samples if small
        else:  # 1D
            spectrum = sample
        
        ax.plot(energy_axis, spectrum, 'b-', linewidth=1.5)
        ax.set_title(f'Epoch {epoch}')
        ax.set_xlabel('Energy (eV)')
        ax.set_ylabel('Intensity')
        ax.grid(True, alpha=0.3)
    
    # Hide empty subplots
    for i in range(n_files, rows * cols):
        row = i // cols
        col = i % cols
        ax = axes[row, col] if rows > 1 else axes[col]
        ax.set_visible(False)
    
    plt.suptitle(f'Training Progression - {run_name}', fontsize=16)
    plt.tight_layout()
    return fig

def plot_sample_comparison(file_path, settings=None, figsize=(12, 8)):
    """
    Plot all samples from a single .npy file for comparison.
    
    Args:
        file_path: Path to the .npy file
        settings: Optional list of experimental settings for the samples
        figsize: Figure size tuple
    """
    samples = np.load(file_path)
    
    # Handle different sample shapes
    if samples.ndim == 3:  # (n_samples, channels, length)
        n_samples = samples.shape[0]
        spectrum_length = samples.shape[-1]
        spectra = samples[:, 0, :]  # Take first channel
    elif samples.ndim == 2:  # Could be (n_samples, length) or (channels, length)
        if samples.shape[0] <= 10:  # Assume first dim is samples if reasonable number
            n_samples = samples.shape[0]
            spectrum_length = samples.shape[1]
            spectra = samples
        else:  # Assume (channels, length)
            n_samples = 1
            spectrum_length = samples.shape[1]
            spectra = samples[0:1, :]  # Take first channel
    else:  # 1D
        n_samples = 1
        spectrum_length = len(samples)
        spectra = samples.reshape(1, -1)
    
    energy_axis = load_energy_axis(spectrum_length)
    
    plt.figure(figsize=figsize)
    
    for i in range(n_samples):
        label = f'Sample {i+1}'
        if settings and i < len(settings):
            label += f' (Settings: {settings[i]})'
        
        plt.plot(energy_axis, spectra[i], linewidth=1.5, label=label, alpha=0.8)
    
    plt.xlabel('Energy (eV)')
    plt.ylabel('Intensity')
    plt.title(f'Generated Spectra Comparison\n{os.path.basename(file_path)}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return plt.gcf()

def analyze_spectrum_statistics(file_path):
    """
    Analyze and print statistics of the generated spectra.
    
    Args:
        file_path: Path to the .npy file
    """
    samples = np.load(file_path)
    
    # Handle different sample shapes
    if samples.ndim == 3:  # (n_samples, channels, length)
        spectra = samples[:, 0, :]  # Take first channel
    elif samples.ndim == 2:
        spectra = samples
    else:
        spectra = samples.reshape(1, -1)
    
    print(f"\nSpectrum Statistics for {os.path.basename(file_path)}:")
    print(f"Shape: {samples.shape}")
    print(f"Number of spectra: {spectra.shape[0]}")
    print(f"Spectrum length: {spectra.shape[1]}")
    print(f"Intensity range: [{np.min(spectra):.4f}, {np.max(spectra):.4f}]")
    print(f"Mean intensity: {np.mean(spectra):.4f}")
    print(f"Std intensity: {np.std(spectra):.4f}")
    
    return {
        'shape': samples.shape,
        'n_spectra': spectra.shape[0],
        'length': spectra.shape[1],
        'min_intensity': np.min(spectra),
        'max_intensity': np.max(spectra),
        'mean_intensity': np.mean(spectra),
        'std_intensity': np.std(spectra)
    }

def main():
    parser = argparse.ArgumentParser(description='Visualize generated spectrum samples')
    parser.add_argument('--results_dir', type=str, default='results', 
                        help='Path to results directory')
    parser.add_argument('--run_name', type=str, default='edm_1d_spectrum_256pts',
                        help='Name of the training run')
    parser.add_argument('--file_path', type=str, default=None,
                        help='Specific .npy file to visualize')
    parser.add_argument('--mode', type=str, default='progression',
                        choices=['progression', 'single', 'comparison', 'stats'],
                        help='Visualization mode')
    parser.add_argument('--save_plots', action='store_true',
                        help='Save plots instead of showing them')
    parser.add_argument('--output_dir', type=str, default='plots',
                        help='Directory to save plots')
    
    args = parser.parse_args()
    
    if args.save_plots:
        os.makedirs(args.output_dir, exist_ok=True)
    
    if args.mode == 'progression':
        # Plot training progression
        fig = plot_training_progression(args.results_dir, args.run_name)
        if fig:
            if args.save_plots:
                fig.savefig(os.path.join(args.output_dir, f'{args.run_name}_progression.png'), 
                           dpi=300, bbox_inches='tight')
                print(f"Saved progression plot to {args.output_dir}")
            else:
                plt.show()
        
    elif args.mode == 'single' and args.file_path:
        # Plot single spectrum file
        if os.path.exists(args.file_path):
            fig = plot_sample_comparison(args.file_path)
            if args.save_plots:
                filename = os.path.splitext(os.path.basename(args.file_path))[0]
                fig.savefig(os.path.join(args.output_dir, f'{filename}_comparison.png'), 
                           dpi=300, bbox_inches='tight')
                print(f"Saved comparison plot to {args.output_dir}")
            else:
                plt.show()
        else:
            print(f"File not found: {args.file_path}")
    
    elif args.mode == 'comparison':
        # Compare all sample files
        sample_files = glob.glob(os.path.join(args.results_dir, args.run_name, "*.npy"))
        for file_path in sorted(sample_files):
            fig = plot_sample_comparison(file_path)
            if args.save_plots:
                filename = os.path.splitext(os.path.basename(file_path))[0]
                fig.savefig(os.path.join(args.output_dir, f'{filename}_comparison.png'), 
                           dpi=300, bbox_inches='tight')
            else:
                plt.show()
        if args.save_plots:
            print(f"Saved all comparison plots to {args.output_dir}")
    
    elif args.mode == 'stats':
        # Print statistics for all files
        sample_files = glob.glob(os.path.join(args.results_dir, args.run_name, "*.npy"))
        for file_path in sorted(sample_files):
            analyze_spectrum_statistics(file_path)

if __name__ == "__main__":
    # Example usage when run directly
    if len(os.sys.argv) == 1:
        print("Example usage:")
        print("python visualize_samples.py --mode progression")
        print("python visualize_samples.py --mode single --file_path results/edm_1d_spectrum_256pts/final_sample.npy")
        print("python visualize_samples.py --mode comparison --save_plots")
        print("python visualize_samples.py --mode stats")
    else:
        main() 