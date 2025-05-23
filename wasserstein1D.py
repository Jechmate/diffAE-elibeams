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

def calculate_wasserstein_distance_bin_by_bin(validation_data, generated_data, min_avg_charge=5):
    """
    Calculate the Wasserstein distance (Earth Mover's Distance) bin-by-bin.
    
    Parameters:
    - validation_data: List where each element is a numpy array of charge values for a specific energy bin
    - generated_data: List where each element is a numpy array of charge values for a specific energy bin
    - min_avg_charge: Minimum average charge value for a bin to be included in analysis (default: 5)
    
    Returns:
    - average_distance: Average Wasserstein distance across all analyzed bins
    - results: Dictionary containing detailed results for each bin
    """
    n_bins = min(len(validation_data), len(generated_data))
    results = {}
    distances = []
    
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
        
        # Skip bins with average charge value <= min_avg_charge
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
        
        # Calculate Wasserstein distance
        try:
            wasserstein_dist = stats.wasserstein_distance(val_bin, gen_bin)
            
            results[i] = {
                'wasserstein_distance': wasserstein_dist,
                'val_data': val_bin,
                'gen_data': gen_bin,
                'val_avg_charge': val_avg_charge,
                'gen_avg_charge': gen_avg_charge,
                'included_in_analysis': True
            }
            
            distances.append(wasserstein_dist)
        except Exception as e:
            results[i] = {
                'error': str(e),
                'included_in_analysis': False
            }
    
    # Calculate average distance
    average_distance = np.mean(distances) if distances else float('nan')
    
    return average_distance, results

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

def process_folder_images(folder_path, calc_spec_function, params, image_extension=".png", min_energy_MeV=None):
    """
    Process all images in a folder and organize charge values by energy bin.
    
    Parameters:
    - folder_path: Path to folder containing images
    - calc_spec_function: Function to calculate spectrum
    - params: Dictionary containing parameters for calc_spec function
    - image_extension: File extension for image files
    - min_energy_MeV: Minimum energy threshold in MeV (bins with energy < min_energy_MeV will be excluded)
    
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
            
            # Find the starting and ending indices based on energy thresholds
            # First find where energy becomes non-zero (start_idx)
            start_idx = next((i for i, e in enumerate(energy) if e > 0), 0)
            
            # Then find where energy drops below min_energy_MeV (end_idx)
            # If min_energy_MeV is specified, find the first index where energy < min_energy_MeV
            if min_energy_MeV is not None:
                end_idx = next((i for i, e in enumerate(energy) if e < min_energy_MeV and e > 0), len(energy))
                # Only keep values between start_idx and end_idx
                bins = bins[start_idx:end_idx]
                values = values[start_idx:end_idx]
            else:
                # If no min_energy_MeV specified, keep all values from start_idx
                bins = bins[start_idx:]
                values = values[start_idx:]
            
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
    min_avg_charge=5,
    image_extension=".png",
    min_energy_MeV=None
):
    """
    Analyze energy spectra data using Wasserstein distance across all subfolders.
    
    Parameters:
    - results_dir: Directory containing generated results (e.g., "results_gaindata_batch4_600e")
    - validation_dir: Directory containing validation data (e.g., "data/with_gain")
    - calc_spec_function: Function to calculate spectrum from images
    - base_params: Dictionary containing base parameters for calc_spec function (without acquisition_time_ms)
    - acquisition_times: Dictionary mapping experiment number (subfolder) to acquisition time in ms
    - min_avg_charge: Minimum average charge value for a bin to be included in analysis (default: 5)
    - image_extension: File extension for image files (default: ".png")
    - min_energy_MeV: Minimum energy threshold in MeV (default: None)
    
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
            'average_wasserstein_distance': 0.0
        }
        
        subfolder_distances = []
        
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
            val_charge_by_bin = process_folder_images(val_folder, calc_spec_function, current_params, 
                                                     image_extension, min_energy_MeV)
            gen_charge_by_bin = process_folder_images(gen_folder, calc_spec_function, current_params, 
                                                     image_extension, min_energy_MeV)
            
            # Prepare data for Wasserstein distance calculation
            num_bins = max(max(val_charge_by_bin.keys(), default=-1), max(gen_charge_by_bin.keys(), default=-1)) + 1
            validation_data = [np.array(val_charge_by_bin.get(i, [])) for i in range(num_bins)]
            generated_data = [np.array(gen_charge_by_bin.get(i, [])) for i in range(num_bins)]
            
            # Calculate Wasserstein distance bin by bin
            avg_distance, bin_results = calculate_wasserstein_distance_bin_by_bin(
                validation_data, 
                generated_data,
                min_avg_charge
            )
            
            # Store results
            model_results[model_key]['subfolder_results'][subfolder] = {
                'average_wasserstein_distance': avg_distance,
                'bin_results': bin_results,
                'acquisition_time_ms': acq_time
            }
            subfolder_distances.append(avg_distance)
        
        # Calculate average Wasserstein distance for this model across all subfolders
        valid_distances = [d for d in subfolder_distances if not np.isnan(d)]
        if valid_distances:
            avg_distance = np.mean(valid_distances)
            model_results[model_key]['average_wasserstein_distance'] = avg_distance
            print(f"{model_name}: average Wasserstein distance = {avg_distance:.4f}")
        else:
            print(f"No valid results for {model_name}")
    
    return model_results

def main():
    """
    Main function to execute the Wasserstein distance workflow.
    """
    # Directory paths
    results_dir = "results_gaindata_batch4_600e_moresections"
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
    min_avg_charge = 0
    
    # Set minimum energy threshold in MeV (can be None to include all energies)
    min_energy_MeV = 0  # Example: only include energies above 30 MeV
    
    # Analyze spectra
    results = analyze_energy_spectra(
        results_dir,
        validation_dir,
        calc_spec,
        base_params,
        acquisition_times,
        min_avg_charge=min_avg_charge,
        min_energy_MeV=min_energy_MeV
    )
    
    # Create visualizations
    # visualize_wasserstein_results(results, "wasserstein_visualizations")
    
    # Find best performing model (lowest average Wasserstein distance)
    valid_models = {k: v for k, v in results.items() 
                   if not np.isnan(v['average_wasserstein_distance'])}
    
    if valid_models:
        best_model = min(valid_models.items(), 
                         key=lambda x: x[1]['average_wasserstein_distance'])
        print(f"\nBest performing model: {best_model[0]} with average Wasserstein distance = {best_model[1]['average_wasserstein_distance']:.4f}")
    
    # Save detailed results to CSV
    output_csv = "wasserstein_results.csv"
    csv_data = []
    
    for model_key, model_data in results.items():
        model_avg_distance = model_data['average_wasserstein_distance']
        
        for subfolder, subfolder_results in model_data['subfolder_results'].items():
            subfolder_avg_distance = subfolder_results.get('average_wasserstein_distance', float('nan'))
            
            # Include bin-level details
            bin_results = subfolder_results.get('bin_results', {})
            for bin_idx, bin_result in bin_results.items():
                if bin_result.get('included_in_analysis', False):
                    csv_data.append({
                        'model': model_key,
                        'physics': model_data['physics'],
                        'sections': model_data['sections'],
                        'cfg': model_data['cfg'],
                        'subfolder': subfolder,
                        'bin': bin_idx,
                        'wasserstein_distance': bin_result.get('wasserstein_distance', float('nan')),
                        'val_avg_charge': bin_result.get('val_avg_charge', float('nan')),
                        'gen_avg_charge': bin_result.get('gen_avg_charge', float('nan')),
                        'subfolder_avg_distance': subfolder_avg_distance,
                        'model_avg_distance': model_avg_distance
                    })
    
    if csv_data:
        csv_df = pd.DataFrame(csv_data)
        csv_df.to_csv(output_csv, index=False)
        print(f"\nDetailed results saved to {output_csv}")
    
    # Also save a summary CSV with just model-level results
    summary_csv = "wasserstein_summary.csv"
    summary_data = []
    
    for model_key, model_data in results.items():
        summary_data.append({
            'model': model_key,
            'physics': model_data['physics'],
            'sections': model_data['sections'],
            'cfg': model_data['cfg'],
            'average_wasserstein_distance': model_data['average_wasserstein_distance']
        })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(summary_csv, index=False)
        print(f"Summary results saved to {summary_csv}")

if __name__ == "__main__":
    main()
