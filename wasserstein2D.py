import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from pathlib import Path
import cv2
from tqdm import tqdm
import re
from collections import defaultdict
from scipy import stats
import ot  # Optimal Transport library - install with: pip install POT

def wasserstein_2d(image1, image2, p=1):
    """
    Calculate the Wasserstein distance between two 2D image distributions.
    
    Parameters:
    - image1, image2: 2D arrays representing image distributions
    - p: Power parameter for the Wasserstein distance (default: 1)
    
    Returns:
    - Wasserstein distance between the images
    """
    # Normalize images to sum to 1 (probability distributions)
    image1_norm = image1 / np.sum(image1) if np.sum(image1) > 0 else image1
    image2_norm = image2 / np.sum(image2) if np.sum(image2) > 0 else image2
    
    # Flatten images for the EMD calculation
    image1_flat = image1_norm.flatten()
    image2_flat = image2_norm.flatten()
    
    # Create the cost matrix based on pixel locations
    h, w = image1.shape
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    x = x.flatten()
    y = y.flatten()
    
    # Coordinates for each pixel
    coords = np.vstack((x, y)).T
    
    # Calculate pairwise distances between all pixel locations
    M = ot.dist(coords, coords, 'euclidean')
    
    # Calculate Earth Mover's Distance / Wasserstein distance
    try:
        # If both distributions are non-empty, use POT library
        if np.sum(image1_flat) > 0 and np.sum(image2_flat) > 0:
            distance = ot.emd2(image1_flat, image2_flat, M)
        else:
            # If either distribution is empty, return infinity
            distance = float('inf')
    except Exception as e:
        print(f"Error calculating Wasserstein distance: {e}")
        distance = float('nan')
    
    return distance

def parse_model_folder_name(folder_name):
    """
    Parse model folder name to extract physics, sections, and CFG.
    
    Parameters:
    - folder_name: Name of the model folder (e.g., "cossched_sec10_cfg1" or "cossched_seccos10_cfg1")
    
    Returns:
    - tuple: (physics, sections, cfg)
    """
    pattern = r"(.+)_sec(?:cos)?(\d+)_cfg(\d+)"
    match = re.match(pattern, folder_name)
    
    if match:
        physics = match.group(1)
        sections = match.group(2)
        cfg = int(match.group(3))
        return physics, sections, cfg
    else:
        return None, None, None

def load_images(folder_path, image_extension=".png"):
    """
    Load all images from a folder.
    
    Parameters:
    - folder_path: Path to folder containing images
    - image_extension: File extension for image files
    
    Returns:
    - List of loaded images as numpy arrays
    """
    folder_path = Path(folder_path)
    image_files = list(folder_path.glob(f"*{image_extension}"))
    images = []
    
    for image_file in image_files:
        try:
            if image_extension == ".png":
                image = cv2.imread(str(image_file), cv2.IMREAD_GRAYSCALE)
                if image is None:
                    continue
                images.append(image)
            elif image_extension == ".pt":
                image = torch.load(image_file, map_location='cpu').numpy()
                images.append(image)
        except Exception as e:
            print(f"Error loading {image_file}: {e}")
    
    return images

def calculate_pairwise_wasserstein(real_images, generated_images, max_comparisons=100):
    """
    Calculate pairwise Wasserstein distances between real and generated images.
    
    Parameters:
    - real_images: List of real image arrays
    - generated_images: List of generated image arrays
    - max_comparisons: Maximum number of image pairs to compare (for efficiency)
    
    Returns:
    - List of Wasserstein distances
    """
    distances = []
    
    # Limit the number of comparisons for efficiency
    n_real = len(real_images)
    n_gen = len(generated_images)
    n_comparisons = min(max_comparisons, n_real * n_gen)
    
    # Randomly select image pairs if we have too many
    if n_real * n_gen > max_comparisons:
        real_indices = np.random.choice(n_real, size=min(n_real, int(np.sqrt(max_comparisons))), replace=False)
        gen_indices = np.random.choice(n_gen, size=min(n_gen, int(np.sqrt(max_comparisons))), replace=False)
        
        for i in tqdm(real_indices, desc="Calculating Wasserstein distances"):
            for j in gen_indices:
                distance = wasserstein_2d(real_images[i], generated_images[j])
                distances.append(distance)
    else:
        # Compare all pairs if within the limit
        for i in tqdm(range(n_real), desc="Calculating Wasserstein distances"):
            for j in range(n_gen):
                distance = wasserstein_2d(real_images[i], generated_images[j])
                distances.append(distance)
    
    return distances

def analyze_image_wasserstein(results_dir, validation_dir, image_extension=".png", max_comparisons=100):
    """
    Analyze image quality using Wasserstein distance across all subfolders.
    
    Parameters:
    - results_dir: Directory containing generated results
    - validation_dir: Directory containing validation data
    - image_extension: File extension for image files
    - max_comparisons: Maximum number of image pairs to compare per subfolder
    
    Returns:
    - DataFrame with Wasserstein distance results
    """
    results_path = Path(results_dir)
    validation_base = Path(validation_dir)
    
    # Validation subfolders to process
    validation_subfolders = ["3", "8", "11", "19", "21"]
    
    # Get all model folders
    model_folders = [f for f in results_path.iterdir() if f.is_dir()]
    
    # Data for results DataFrame
    all_results = []
    
    for model_folder in tqdm(model_folders, desc="Processing models"):
        model_name = model_folder.name
        physics, sections, cfg = parse_model_folder_name(model_name)
        
        if physics is None:
            print(f"Skipping folder {model_name}: could not parse name")
            continue
        
        model_key = f"{physics}_sec{sections}_cfg{cfg}"
        
        for subfolder in validation_subfolders:
            val_folder = validation_base / subfolder
            gen_folder = model_folder / subfolder
            
            if not val_folder.exists() or not gen_folder.exists():
                continue
            
            # Load real and generated images
            real_images = load_images(val_folder, image_extension)
            generated_images = load_images(gen_folder, image_extension)
            
            if not real_images or not generated_images:
                print(f"Skipping {model_key}/{subfolder}: no images found")
                continue
            
            # Calculate Wasserstein distances
            distances = calculate_pairwise_wasserstein(real_images, generated_images, max_comparisons)
            
            # Filter out infinities and NaNs
            valid_distances = [d for d in distances if np.isfinite(d)]
            
            if valid_distances:
                avg_distance = np.mean(valid_distances)
                std_distance = np.std(valid_distances)
                min_distance = np.min(valid_distances)
                max_distance = np.max(valid_distances)
                
                # Add to results
                all_results.append({
                    'model': model_key,
                    'physics': physics,
                    'sections': sections,
                    'cfg': cfg,
                    'subfolder': subfolder,
                    'avg_wasserstein': avg_distance,
                    'std_wasserstein': std_distance,
                    'min_wasserstein': min_distance,
                    'max_wasserstein': max_distance,
                    'num_comparisons': len(valid_distances)
                })
                
                print(f"{model_key}, subfolder {subfolder}: avg Wasserstein = {avg_distance:.4f}")
    
    # Create DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Calculate model-level average distances
    model_avg = results_df.groupby(['model', 'physics', 'sections', 'cfg'])['avg_wasserstein'].mean().reset_index()
    model_avg = model_avg.rename(columns={'avg_wasserstein': 'model_avg_wasserstein'})
    
    # Merge back to get model average in the main results
    results_df = pd.merge(results_df, model_avg, on=['model', 'physics', 'sections', 'cfg'])
    
    # Save results
    results_df.to_csv('wasserstein_2d_results.csv', index=False)
    
    # Create a summary DataFrame
    summary_df = model_avg[['model', 'physics', 'sections', 'cfg', 'model_avg_wasserstein']]
    summary_df.to_csv('wasserstein_2d_summary.csv', index=False)
    
    return results_df

def visualize_wasserstein_2d_results(results_df):
    """
    Create visualizations of 2D Wasserstein distance results.
    
    Parameters:
    - results_df: DataFrame containing Wasserstein distance results
    """
    # Group by physics and cfg, then calculate the mean Wasserstein distance
    grouped_data = results_df.groupby(['physics', 'cfg'])['model_avg_wasserstein'].mean().reset_index()
    
    # Set up the plot
    plt.figure(figsize=(12, 8))
    sns.set_style('whitegrid')
    
    # Plot the data
    physics_types = grouped_data['physics'].unique()
    for physics in physics_types:
        # Get data for this physics type
        phys_data = grouped_data[grouped_data['physics'] == physics]
        
        # Sort by cfg
        phys_data = phys_data.sort_values('cfg')
        
        # Plot line with markers
        plt.plot(phys_data['cfg'], phys_data['model_avg_wasserstein'], 
                 marker='o', linestyle='-', linewidth=2, 
                 label=physics)
    
    # Add error bars (standard deviation)
    for physics in physics_types:
        # Calculate standard deviation for each cfg value
        std_data = results_df[results_df['physics'] == physics].groupby('cfg')['model_avg_wasserstein'].std().reset_index()
        mean_data = results_df[results_df['physics'] == physics].groupby('cfg')['model_avg_wasserstein'].mean().reset_index()
        
        # Sort by cfg
        std_data = std_data.sort_values('cfg')
        mean_data = mean_data.sort_values('cfg')
        
        # Plot error bars
        plt.errorbar(std_data['cfg'], mean_data['model_avg_wasserstein'], 
                     yerr=std_data['model_avg_wasserstein'], fmt='none', 
                     alpha=0.3, capsize=5, ecolor='gray')
    
    # Customize the plot
    plt.title('Average 2D Wasserstein Distance by CFG Value for Different Physics Models', fontsize=16)
    plt.xlabel('CFG Value', fontsize=14)
    plt.ylabel('Average Wasserstein Distance (lower is better)', fontsize=14)
    plt.xticks(np.arange(1, max(results_df['cfg'])+1, 1))  # Set x-ticks to whole numbers
    plt.legend(title='Physics Model', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add annotations for the lowest values (best performance)
    for physics in physics_types:
        phys_data = grouped_data[grouped_data['physics'] == physics]
        min_idx = phys_data['model_avg_wasserstein'].idxmin()
        min_cfg = phys_data.loc[min_idx, 'cfg']
        min_distance = phys_data.loc[min_idx, 'model_avg_wasserstein']
        plt.annotate(f'Min: {min_distance:.2f}', 
                     xy=(min_cfg, min_distance),
                     xytext=(5, 10), textcoords='offset points',
                     arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))
    
    # Show the plot
    plt.tight_layout()
    plt.savefig('wasserstein_2d_by_cfg.png', dpi=300)
    plt.show()
    
    # Create a bar plot to compare the physics types
    plt.figure(figsize=(12, 6))
    
    # Plot grouped bar chart
    sns.barplot(x='cfg', y='model_avg_wasserstein', hue='physics', data=grouped_data)
    
    plt.title('Comparison of Average 2D Wasserstein Distances by CFG Value', fontsize=16)
    plt.xlabel('CFG Value', fontsize=14)
    plt.ylabel('Average Wasserstein Distance (lower is better)', fontsize=14)
    plt.legend(title='Physics Model')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('wasserstein_2d_barplot.png', dpi=300)
    plt.show()
    
    # Create a boxplot to show distribution of Wasserstein distances by subfolder
    plt.figure(figsize=(14, 8))
    sns.boxplot(x='physics', y='avg_wasserstein', hue='cfg', data=results_df)
    
    plt.title('Distribution of 2D Wasserstein Distances by Physics Model and CFG', fontsize=16)
    plt.xlabel('Physics Model', fontsize=14)
    plt.ylabel('Wasserstein Distance (lower is better)', fontsize=14)
    plt.legend(title='CFG Value')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('wasserstein_2d_boxplot.png', dpi=300)
    plt.show()
    
    # Print summary statistics
    print("Summary Statistics:")
    summary_stats = results_df.groupby(['physics', 'cfg'])['avg_wasserstein'].agg(['mean', 'std', 'min', 'max']).round(4)
    print(summary_stats)
    
    return summary_stats

# Main execution
def main():
    """
    Main function to execute the 2D Wasserstein distance analysis
    """
    # Directory paths
    results_dir = "results_gaindata_batch4_600e"
    validation_dir = "data/with_gain"
    
    # Set image extension
    image_extension = ".png"
    
    # Set maximum number of image pairs to compare per subfolder for efficiency
    max_comparisons = 500  # You can adjust this based on your computational resources
    
    # Run the analysis
    print("Starting 2D Wasserstein distance analysis...")
    results_df = analyze_image_wasserstein(
        results_dir, 
        validation_dir, 
        image_extension=image_extension,
        max_comparisons=max_comparisons
    )
    
    # Visualize results
    print("\nCreating visualizations...")
    stats = visualize_wasserstein_2d_results(results_df)
    
    # Find best performing model (lowest average Wasserstein distance)
    model_summary = results_df.groupby(['model', 'physics', 'cfg'])['model_avg_wasserstein'].mean().reset_index()
    best_model = model_summary.loc[model_summary['model_avg_wasserstein'].idxmin()]
    print(f"\nBest performing model: {best_model['model']} with average Wasserstein distance = {best_model['model_avg_wasserstein']:.4f}")
    
    return results_df

if __name__ == "__main__":
    main()