import os
import torch
import cv2
import pandas as pd
from tqdm import tqdm
from src.utils import deflection_biexp_calc, calc_spec

def create_spectrum_dataset(
    root_dir='data/with_gain',
    output_dir='data/spectra',
    beam_point_x=62,
    image_gain=0
):
    """
    Create individual CSV files containing 1D spectra from images.
    Each CSV file contains energy and intensity columns for a single image,
    starting from energies just below 100 MeV.
    
    Args:
        root_dir (str): Directory containing the image folders
        output_dir (str): Directory to save the CSV files
        beam_point_x (int): X coordinate of the electron beam pointing
        image_gain (float): Image gain value
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load parameters file
    params_df = pd.read_csv('data/params.csv')
    
    # Get all subfolders
    subfolders = [f.path for f in os.scandir(root_dir) if f.is_dir()]
    
    for subfolder in tqdm(subfolders, desc="Processing folders"):
        # Get experiment number from folder name
        folder_name = os.path.basename(subfolder)
        experiment_num = int(folder_name)
        
        # Get acquisition time for this experiment
        acquisition_time_ms = params_df.loc[params_df['experiment'] == experiment_num, 'ms'].iloc[0]
        
        # Create corresponding output subfolder
        subfolder_output_dir = os.path.join(output_dir, folder_name)
        os.makedirs(subfolder_output_dir, exist_ok=True)
        
        # Get all PNG files in the subfolder
        image_files = [f for f in os.listdir(subfolder) if f.endswith('.png')]
        
        for image_file in tqdm(image_files, desc=f"Processing images in {folder_name}", leave=False):
            # Read image
            image_path = os.path.join(subfolder, image_file)
            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            
            # Convert to tensor and add batch dimension
            image_tensor = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0)
            
            # Calculate deflection
            deflection_MeV, deflection_MeV_dx = deflection_biexp_calc(
                batch_size=1,
                hor_image_size=image.shape[1],
                electron_pointing_pixel=beam_point_x
            )
            
            # Calculate spectrum
            _, spectrum = calc_spec(
                image=image_tensor/255,
                electron_pointing_pixel=beam_point_x,
                deflection_MeV=deflection_MeV,
                acquisition_time_ms=torch.tensor([acquisition_time_ms]),
                image_gain=image_gain
            )
            
            # Convert to numpy
            energy = deflection_MeV.squeeze().numpy()
            spectrum = spectrum.squeeze().numpy()
            
            # Find index where energy drops below 100 MeV
            start_idx = next((i for i, e in enumerate(energy) if e < 100), 0)
            
            # Create DataFrame for this spectrum, starting from energies below 100 MeV
            df = pd.DataFrame({
                'energy': energy[start_idx:],
                'intensity': spectrum[start_idx:]
            })
            
            # Remove the first beam_point_x lines
            df = df.iloc[beam_point_x:]
            
            # Save to CSV (use the same filename as the image but with .csv extension)
            output_filename = os.path.splitext(image_file)[0] + '.csv'
            output_path = os.path.join(subfolder_output_dir, output_filename)
            df.to_csv(output_path, index=False)

if __name__ == "__main__":
    create_spectrum_dataset()
