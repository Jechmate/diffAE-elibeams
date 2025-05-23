import os
import numpy as np
from scipy import stats
import torch
import pandas as pd
from pathlib import Path
import re
from collections import defaultdict
from src.utils import deflection_biexp_calc
from tqdm import tqdm
import cv2
import torchvision.transforms.functional as f
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def calc_spec(image, electron_pointing_pixel, deflection_MeV, acquisition_time_ms, device='cpu'):
    # Ensure image is on the correct device
    image = image.to(device)
    # print(deflection_MeV.shape)
    # Get image dimensions
    hor_image_size = image.shape[1]  # Width
    
    # Calculate horizontal profile by summing along height dimension
    horizontal_profile = torch.sum(image, dim=0).to(device)
    
    # Initialize spectrum arrays
    spectrum_in_pixel = torch.zeros(hor_image_size).to(device)
    spectrum_in_MeV = torch.zeros(hor_image_size).to(device)
    
    # Fill spectrum_in_pixel
    spectrum_in_pixel[electron_pointing_pixel:] = horizontal_profile[electron_pointing_pixel:]
    
    # Prepare for derivative calculation
    shifts = -1
    deflection_MeV_shifted = torch.roll(deflection_MeV, shifts=shifts)
    
    # Pad with zeros where necessary
    if shifts < 0:
        # For left shift, zero pad on the right
        deflection_MeV_shifted[-shifts:] = 0
    else:
        # For right shift, zero pad on the left
        deflection_MeV_shifted[:shifts] = 0
    
    # Calculate derivative
    derivative = deflection_MeV - deflection_MeV_shifted
    derivative = derivative.to(device)
    
    # Calculate spectrum in MeV, avoiding division by zero
    mask = derivative != 0
    spectrum_in_MeV[mask] = spectrum_in_pixel[mask] / derivative[mask]
    
    # Handle any infinities or NaNs
    spectrum_in_MeV[~torch.isfinite(spectrum_in_MeV)] = 0
    
    # Apply calibration factor and acquisition time
    spectrum_calibrated = spectrum_in_MeV * 3.706 / acquisition_time_ms
    
    return deflection_MeV, spectrum_calibrated

def load_acquisition_times(params_file):
    """
    Load acquisition times from params.csv.
    
    Parameters:
    - params_file: Path to params.csv
    
    Returns:
    - Dictionary mapping experiment number to acquisition time (ms)
    """
    params_df = pd.read_csv(params_file)
    return {int(row['experiment']): int(row['ms']) for _, row in params_df.iterrows()}

def perform_ks_test_bin_by_bin(validation_data, generated_data, alpha=0.05, min_avg_charge=5):
    """
    Perform a bin-by-bin two-sample Kolmogorov-Smirnov test.
    
    Parameters:
    - validation_data: List where each element is a numpy array of charge values for a specific energy bin
    - generated_data: List where each element is a numpy array of charge values for a specific energy bin
    - alpha: Significance level for the KS test (default: 0.05)
    - min_avg_charge: Minimum average charge value for a bin to be included in analysis (default: 5)
    
    Returns:
    - success_rate: Percentage of bins where p-value > alpha (null hypothesis not rejected)
    - results: Dictionary containing detailed results for each bin
    """
    n_bins = min(len(validation_data), len(generated_data))
    results = {}
    p_values = []
    success_count = 0
    total_tests = 0
    
    for i in range(n_bins):
        val_bin = validation_data[i]
        gen_bin = generated_data[i]
        
        # Skip bins with insufficient data
        if len(val_bin) == 0 or len(gen_bin) == 0:
            results[i] = {
                'error': 'Insufficient data',
                'included_in_analysis': False
            }
            continue
        
        # Calculate average charge values
        val_avg_charge = np.mean(val_bin)
        gen_avg_charge = np.mean(gen_bin)
        
        # Skip bins with average charge value <= 5
        if val_avg_charge <= min_avg_charge and gen_avg_charge <= min_avg_charge:
            results[i] = {
                'val_avg_charge': val_avg_charge,
                'gen_avg_charge': gen_avg_charge,
                'included_in_analysis': False,
                'reason': f'Average charge values ({val_avg_charge:.2f}, {gen_avg_charge:.2f}) below threshold {min_avg_charge}',
                'val_data': val_bin,
                'gen_data': gen_bin
            }
            continue
        
        # Perform KS test on bins that meet the criteria
        try:
            ks_statistic, p_value = stats.ks_2samp(val_bin, gen_bin)
            
            results[i] = {
                'ks_statistic': ks_statistic,
                'p_value': p_value,
                'success': p_value > alpha,
                'val_data': val_bin,
                'gen_data': gen_bin,
                'val_avg_charge': val_avg_charge,
                'gen_avg_charge': gen_avg_charge,
                'included_in_analysis': True
            }
            
            p_values.append(p_value)
            total_tests += 1
            if p_value > alpha:
                success_count += 1
        except Exception as e:
            results[i] = {
                'error': str(e),
                'included_in_analysis': False
            }
    
    # Calculate success rate
    success_rate = (success_count / total_tests * 100) if total_tests > 0 else 0
    
    return success_rate, results

def parse_model_folder_name(folder_name):
    """
    Parse model folder name to extract physics, sections, and CFG.
    
    Parameters:
    - folder_name: Name of the model folder (e.g., "cossched_sec10_cfg1" or "cossched_seccos10_cfg1")
    
    Returns:
    - tuple: (physics, sections, cfg)
    """
    # Updated pattern to handle both "sec" and "seccos" formats
    pattern = r"(.+)_sec(?:cos)?(\d+)_cfg(\d+)"
    match = re.match(pattern, folder_name)
    
    if match:
        physics = match.group(1)
        sections = match.group(2)
        cfg = int(match.group(3))
        return physics, sections, cfg
    else:
        return None, None, None

def process_folder_images(folder_path, calc_spec_function, params, image_extension=".png"):
    """
    Process all images in a folder and organize charge values by energy bin.
    
    Parameters:
    - folder_path: Path to folder containing images
    - calc_spec_function: Function to calculate spectrum
    - params: Dictionary containing parameters for calc_spec function
    - image_extension: File extension for image files
    
    Returns:
    - charge_by_bin: Dictionary with bin indices as keys and lists of charge values as values
    """
    folder_path = Path(folder_path)
    image_files = list(folder_path.glob(f"*{image_extension}"))
    charge_by_bin = defaultdict(list)
    
    for image_file in image_files:
        try:
            # Load image
            if image_extension == ".png":
                image = cv2.imread(str(image_file), cv2.IMREAD_GRAYSCALE)
                if image is None:
                    raise ValueError(f"Failed to load image: {image_file}")
                # Convert to tensor if needed
                image = torch.tensor(image, dtype=torch.float32, device=params.get('device', 'cpu'))
            else:
                # For .pt files or other tensor formats
                image = torch.load(image_file, map_location=params.get('device', 'cpu'))
            
            # Calculate spectrum using the provided function
            bins, values = calc_spec_function(
                image,
                params['electron_pointing_pixel'], 
                params['deflection_MeV'], 
                params['acquisition_time_ms'], 
                device=params.get('device', 'cpu')
            )
            energy = params['deflection_MeV'].squeeze().numpy()
            values = values.squeeze().numpy()
            bins = bins.squeeze().numpy()
            start_idx = next((i for i, e in enumerate(energy) if e < 100 and e > 0), 0)
            # print(start_idx)
            bins = bins[start_idx:]
            values = values[start_idx:]
            #print(len(bins), len(values))
            
            # Convert to numpy if they're tensors
            # if isinstance(bins, torch.Tensor):
            #     bins = bins.cpu().numpy()
            # if isinstance(values, torch.Tensor):
            #     values = values.cpu().numpy()
            
            # Store values by bin
            for i, value in enumerate(values):
                charge_by_bin[i].append(value)
                
        except Exception as e:
            print(f"Error processing {image_file}: {e}")
    return charge_by_bin

def analyze_energy_spectra(
    results_dir, 
    validation_dir, 
    calc_spec_function, 
    base_params,
    acquisition_times,
    alpha=0.05,
    min_avg_charge=5,
    image_extension=".png"
):
    """
    Analyze energy spectra data using KS tests across all subfolders.
    
    Parameters:
    - results_dir: Directory containing generated results (e.g., "results_gaindata_batch4_600e")
    - validation_dir: Directory containing validation data (e.g., "data/with_gain")
    - calc_spec_function: Function to calculate spectrum from images
    - base_params: Dictionary containing base parameters for calc_spec function (without acquisition_time_ms)
    - acquisition_times: Dictionary mapping experiment number (subfolder) to acquisition time in ms
    - alpha: Significance level for KS test (default: 0.05)
    - min_avg_charge: Minimum average charge value for a bin to be included in analysis (default: 5)
    - image_extension: File extension for image files (default: ".png")
    
    Returns:
    - Dictionary with detailed results for each model and subfolder
    """
    results_path = Path(results_dir)
    validation_base = Path(validation_dir)
    
    # Validation subfolders to process
    validation_subfolders = ["3", "8", "11", "19", "21"]
    
    # Get all model folders
    model_folders = [f for f in results_path.iterdir() if f.is_dir()]
    
    # Dictionary to store results
    model_results = {}
    
    for model_folder in tqdm(model_folders):
        model_name = model_folder.name
        physics, sections, cfg = parse_model_folder_name(model_name)
        
        if physics is None:
            print(f"Skipping folder {model_name}: could not parse name")
            continue
        
        model_key = f"{physics}_sec{sections}_cfg{cfg}"
        model_results[model_key] = {
            'physics': physics,
            'sections': sections,
            'cfg': cfg,
            'subfolder_results': {},
            'average_success_rate': 0.0
        }
        
        subfolder_success_rates = []
        
        for subfolder in validation_subfolders:
            val_folder = validation_base / subfolder
            gen_folder = model_folder / subfolder
            
            if not val_folder.exists() or not gen_folder.exists():
                continue
            
            # Get acquisition time for this experiment
            acq_time = acquisition_times.get(int(subfolder))
            if acq_time is None:
                continue
            
            # Create a copy of base_params and update with the correct acquisition time
            current_params = base_params.copy()
            current_params['acquisition_time_ms'] = acq_time
            
            # Process validation and generated images with the correct params
            val_charge_by_bin = process_folder_images(val_folder, calc_spec_function, current_params, image_extension)
            gen_charge_by_bin = process_folder_images(gen_folder, calc_spec_function, current_params, image_extension)
            
            # Prepare data for KS test
            num_bins = max(max(val_charge_by_bin.keys(), default=-1), max(gen_charge_by_bin.keys(), default=-1)) + 1
            validation_data = [np.array(val_charge_by_bin.get(i, [])) for i in range(num_bins)]
            generated_data = [np.array(gen_charge_by_bin.get(i, [])) for i in range(num_bins)]
            
            # Perform KS test bin by bin
            success_rate, bin_results = perform_ks_test_bin_by_bin(
                validation_data, 
                generated_data, 
                alpha,
                min_avg_charge
            )
            
            # Store results
            model_results[model_key]['subfolder_results'][subfolder] = {
                'success_rate': success_rate,
                'bin_results': bin_results,
                'acquisition_time_ms': acq_time
            }
            subfolder_success_rates.append(success_rate)
        
        # Calculate average success rate for this model across all subfolders
        if subfolder_success_rates:
            avg_success_rate = np.mean(subfolder_success_rates)
            model_results[model_key]['average_success_rate'] = avg_success_rate
            print(f"{model_name}: average success rate = {avg_success_rate:.2f}%")
        else:
            print(f"No valid results for {model_name}")
    
    return model_results

def visualize_ks_test_results(model_results, output_dir="ks_test_visualizations"):
    """
    Create visualizations of distributions that passed and failed the KS test.
    
    Parameters:
    - model_results: Dictionary containing KS test results from analyze_energy_spectra
    - output_dir: Directory to save visualizations
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    for model_key, model_data in model_results.items():
        # Create a model-specific directory
        model_dir = os.path.join(output_dir, model_key)
        os.makedirs(model_dir, exist_ok=True)
        
        for subfolder, subfolder_results in model_data['subfolder_results'].items():
            bin_results = subfolder_results.get('bin_results', {})
            
            # Separate bins that passed and failed the test
            passed_bins = []
            failed_bins = []
            excluded_bins = []
            
            for bin_idx, result in bin_results.items():
                if result.get('included_in_analysis', False):
                    if result.get('success', False):
                        passed_bins.append((bin_idx, result))
                    else:
                        failed_bins.append((bin_idx, result))
                else:
                    excluded_bins.append((bin_idx, result))
            
            # Select up to 2 examples from each category
            passed_examples = passed_bins[:2] if len(passed_bins) >= 2 else passed_bins
            failed_examples = failed_bins[:2] if len(failed_bins) >= 2 else failed_bins
            excluded_examples = excluded_bins[:2] if len(excluded_bins) >= 2 else excluded_bins
            
            # Skip if we don't have enough examples
            if not (passed_examples or failed_examples):
                continue
            
            # Create a PDF to save the plots
            pdf_filename = os.path.join(model_dir, f"{subfolder}_examples.pdf")
            with PdfPages(pdf_filename) as pdf:
                # Create plots for passed examples
                for bin_idx, result in passed_examples:
                    if 'val_data' in result and 'gen_data' in result:
                        fig, ax = plt.subplots(figsize=(10, 6))
                        
                        # Get the data
                        val_data = result['val_data']
                        gen_data = result['gen_data']
                        
                        # Create histograms
                        bins = 30
                        ax.hist(val_data, bins=bins, alpha=0.5, label=f'Validation (avg={result["val_avg_charge"]:.2f})', density=True)
                        ax.hist(gen_data, bins=bins, alpha=0.5, label=f'Generated (avg={result["gen_avg_charge"]:.2f})', density=True)
                        
                        ax.set_title(f"PASSED: Bin {bin_idx}, p-value={result['p_value']:.4f}")
                        ax.set_xlabel("Charge Value")
                        ax.set_ylabel("Density")
                        ax.legend()
                        plt.tight_layout()
                        pdf.savefig(fig)
                        plt.close(fig)
                
                # Create plots for failed examples
                for bin_idx, result in failed_examples:
                    if 'val_data' in result and 'gen_data' in result:
                        fig, ax = plt.subplots(figsize=(10, 6))
                        
                        # Get the data
                        val_data = result['val_data']
                        gen_data = result['gen_data']
                        
                        # Create histograms
                        bins = 30
                        ax.hist(val_data, bins=bins, alpha=0.5, label=f'Validation (avg={result["val_avg_charge"]:.2f})', density=True)
                        ax.hist(gen_data, bins=bins, alpha=0.5, label=f'Generated (avg={result["gen_avg_charge"]:.2f})', density=True)
                        
                        ax.set_title(f"FAILED: Bin {bin_idx}, p-value={result['p_value']:.4f}")
                        ax.set_xlabel("Charge Value")
                        ax.set_ylabel("Density")
                        ax.legend()
                        plt.tight_layout()
                        pdf.savefig(fig)
                        plt.close(fig)
                
                # Create plots for excluded examples
                for bin_idx, result in excluded_examples:
                    if 'val_data' in result and 'gen_data' in result:
                        fig, ax = plt.subplots(figsize=(10, 6))
                        
                        # Get the data
                        val_data = result['val_data']
                        gen_data = result['gen_data']
                        
                        # Create histograms
                        bins = 30
                        val_avg = result.get('val_avg_charge', np.mean(val_data) if len(val_data) > 0 else 0)
                        gen_avg = result.get('gen_avg_charge', np.mean(gen_data) if len(gen_data) > 0 else 0)
                        
                        ax.hist(val_data, bins=bins, alpha=0.5, label=f'Validation (avg={val_avg:.2f})', density=True)
                        ax.hist(gen_data, bins=bins, alpha=0.5, label=f'Generated (avg={gen_avg:.2f})', density=True)
                        
                        reason = result.get('reason', 'Not included in analysis')
                        ax.set_title(f"EXCLUDED: Bin {bin_idx}, {reason}")
                        ax.set_xlabel("Charge Value")
                        ax.set_ylabel("Density")
                        ax.legend()
                        plt.tight_layout()
                        pdf.savefig(fig)
                        plt.close(fig)
            
            print(f"Created visualization for {model_key}, subfolder {subfolder}: {pdf_filename}")

def main():
    """
    Main function to execute the KS test workflow.
    """
    # Directory paths
    results_dir = "results_gaindata_batch4_600e"
    validation_dir = "data/with_gain"
    params_file = "data/params.csv"
    
    # Load acquisition times from the CSV file
    acquisition_times = load_acquisition_times(params_file)
    print(f"Loaded acquisition times for {len(acquisition_times)} experiments from {params_file}")
    deflection_MeV, _ = deflection_biexp_calc(1, 512, 62)
    deflection_MeV = deflection_MeV.squeeze()

    # Base parameters for calc_spec function (without acquisition_time_ms)
    base_params = {
        'electron_pointing_pixel': 62,
        'deflection_MeV': deflection_MeV,
        'device': 'cpu'
    }
    
    # Set minimum average charge value for bins to be included in analysis
    min_avg_charge = 5
    
    # Analyze spectra
    results = analyze_energy_spectra(
        results_dir,
        validation_dir,
        calc_spec,
        base_params,
        acquisition_times,
        alpha=0.05,
        min_avg_charge=min_avg_charge
    )
    
    # Create visualizations
    visualize_ks_test_results(results, "ks_test_visualizations")
    
    # Find best performing model (highest average success rate)
    best_model = max(results.items(), key=lambda x: x[1]['average_success_rate'])
    print(f"\nBest performing model: {best_model[0]} with average success rate = {best_model[1]['average_success_rate']:.2f}%")
    
    # Save simplified results to CSV (only model averages, not subfolders)
    output_csv = "ks_test_results.csv"
    csv_data = []
    
    for model_key, model_data in results.items():
        csv_data.append({
            'model': model_key,
            'physics': model_data['physics'],
            'sections': model_data['sections'],
            'cfg': model_data['cfg'],
            'average_success_rate': model_data['average_success_rate']
        })
    
    if csv_data:
        csv_df = pd.DataFrame(csv_data)
        csv_df.to_csv(output_csv, index=False)
        print(f"\nResults saved to {output_csv}")

if __name__ == "__main__":
    main()
