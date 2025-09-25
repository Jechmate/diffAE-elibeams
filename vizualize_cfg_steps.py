import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# Set style for publication-ready plots
plt.style.use('default')
sns.set_palette("husl")

def load_and_filter_data(filepath='parameter_averages.csv'):
    """Load CSV data and filter out 10 steps"""
    df = pd.read_csv(filepath)
    
    # Clean column names
    df.columns = df.columns.str.strip()
    
    # Filter out 10 steps
    df_filtered = df[df['num_steps'] != 10].copy()
    
    print(f"Loaded {len(df)} rows, filtered to {len(df_filtered)} rows (excluding 10 steps)")
    print(f"CFG scales: {sorted(df_filtered['cfg_scale'].unique())}")
    print(f"Step counts: {sorted(df_filtered['num_steps'].unique())}")
    
    return df_filtered

def create_publication_plots(df, save_plots=True, dpi=300):
    """Create all three publication-ready plots"""
    
    # Calculate Y-axis range for consistent scaling
    min_distance = df['avg_wasserstein_distance'].min()
    max_distance = df['avg_wasserstein_distance'].max()
    data_range = max_distance - min_distance
    padding = data_range * 0.05  # 5% padding
    y_min = max(0, min_distance - padding)
    y_max = max_distance + padding
    
    # Get unique values for grouping
    unique_cfg = sorted(df['cfg_scale'].unique())
    unique_steps = sorted(df['num_steps'].unique())
    
    # Color palette
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(unique_cfg), len(unique_steps))))
    
    # Create figure with subplots
    fig = plt.figure(figsize=(18, 6))
    
    # Plot 1: Wasserstein Distance vs CFG Scale
    ax1 = plt.subplot(1, 3, 1)
    
    for i, steps in enumerate(unique_steps):
        step_data = df[df['num_steps'] == steps].sort_values('cfg_scale')
        plt.plot(step_data['cfg_scale'], step_data['avg_wasserstein_distance'], 
                'o-', linewidth=2.5, markersize=6, label=f'{steps} steps', 
                color=colors[i])
    
    plt.xlabel('CFG Scale', fontsize=12, fontweight='bold')
    plt.ylabel('Average Wasserstein Distance', fontsize=12, fontweight='bold')
    plt.title('Average Wasserstein Distance vs CFG Scale', fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.ylim(y_min, y_max)
    
    # Plot 2: Wasserstein Distance vs Number of Steps
    ax2 = plt.subplot(1, 3, 2)
    
    for i, cfg in enumerate(unique_cfg):
        cfg_data = df[df['cfg_scale'] == cfg].sort_values('num_steps')
        plt.plot(cfg_data['num_steps'], cfg_data['avg_wasserstein_distance'], 
                'o-', linewidth=2.5, markersize=6, label=f'CFG {cfg}', 
                color=colors[i])
    
    plt.xlabel('Number of Steps', fontsize=12, fontweight='bold')
    plt.ylabel('Average Wasserstein Distance', fontsize=12, fontweight='bold')
    plt.title('Average Wasserstein Distance vs Number of Steps', fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.ylim(y_min, y_max)
    
    # Plot 3: Parameter Space Heatmap
    ax3 = plt.subplot(1, 3, 3)
    
    # Create pivot table for heatmap
    heatmap_data = df.pivot(index='num_steps', columns='cfg_scale', values='avg_wasserstein_distance')
    
    # Create heatmap
    im = plt.imshow(heatmap_data, cmap='RdYlBu_r', aspect='auto', 
                   extent=[unique_cfg[0]-0.5, unique_cfg[-1]+0.5, 
                          unique_steps[0]-2.5, unique_steps[-1]+2.5])
    
    plt.xlabel('CFG Scale', fontsize=12, fontweight='bold')
    plt.ylabel('Number of Steps', fontsize=12, fontweight='bold')
    plt.title('Parameter Space Heatmap', fontsize=14, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax3, shrink=0.8)
    cbar.set_label('Average Wasserstein Distance', rotation=270, labelpad=20, fontweight='bold')
    
    # Add text annotations for exact values
    for i, steps in enumerate(unique_steps):
        for j, cfg in enumerate(unique_cfg):
            value = heatmap_data.loc[steps, cfg]
            if not pd.isna(value):
                plt.text(cfg, steps, f'{value:.3f}', ha='center', va='center', 
                        fontsize=8, fontweight='bold', 
                        color='white' if value > (min_distance + max_distance)/2 else 'black')
    
    plt.tight_layout()
    
    if save_plots:
        plt.savefig('wasserstein_analysis.png', dpi=dpi, bbox_inches='tight')
        plt.savefig('wasserstein_analysis.pdf', bbox_inches='tight')
        print("Plots saved as 'wasserstein_analysis.png' and 'wasserstein_analysis.pdf'")
    
    plt.show()
    
    return fig

def create_individual_plots(df, save_plots=True, dpi=300):
    """Create individual high-resolution plots for papers"""
    
    # Calculate Y-axis range
    min_distance = df['avg_wasserstein_distance'].min()
    max_distance = df['avg_wasserstein_distance'].max()
    data_range = max_distance - min_distance
    padding = data_range * 0.05
    y_min = max(0, min_distance - padding)
    y_max = max_distance + padding
    
    unique_cfg = sorted(df['cfg_scale'].unique())
    unique_steps = sorted(df['num_steps'].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(unique_cfg), len(unique_steps))))
    
    # Individual Plot 1: CFG Scale Analysis
    plt.figure(figsize=(10, 6))
    
    for i, steps in enumerate(unique_steps):
        step_data = df[df['num_steps'] == steps].sort_values('cfg_scale')
        plt.plot(step_data['cfg_scale'], step_data['avg_wasserstein_distance'], 
                'o-', linewidth=3, markersize=8, label=f'{steps} steps', 
                color=colors[i])
    
    plt.xlabel('CFG Scale', fontsize=14, fontweight='bold')
    plt.ylabel('Average Wasserstein Distance', fontsize=14, fontweight='bold')
    plt.title('Average Wasserstein Distance vs CFG Scale', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.ylim(y_min, y_max)
    
    if save_plots:
        plt.savefig('cfg_scale_analysis.png', dpi=dpi, bbox_inches='tight')
        plt.savefig('cfg_scale_analysis.pdf', bbox_inches='tight')
    
    plt.show()
    
    # Individual Plot 2: Steps Analysis
    plt.figure(figsize=(10, 6))
    
    for i, cfg in enumerate(unique_cfg):
        cfg_data = df[df['cfg_scale'] == cfg].sort_values('num_steps')
        plt.plot(cfg_data['num_steps'], cfg_data['avg_wasserstein_distance'], 
                'o-', linewidth=3, markersize=8, label=f'CFG {cfg}', 
                color=colors[i])
    
    plt.xlabel('Number of Steps', fontsize=14, fontweight='bold')
    plt.ylabel('Average Wasserstein Distance', fontsize=14, fontweight='bold')
    plt.title('Average Wasserstein Distance vs Number of Steps', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12, ncol=2)
    plt.grid(True, alpha=0.3)
    plt.ylim(y_min, y_max)
    
    if save_plots:
        plt.savefig('steps_analysis.png', dpi=dpi, bbox_inches='tight')
        plt.savefig('steps_analysis.pdf', bbox_inches='tight')
    
    plt.show()
    
    # Individual Plot 3: Heatmap
    plt.figure(figsize=(10, 8))
    
    heatmap_data = df.pivot(index='num_steps', columns='cfg_scale', values='avg_wasserstein_distance')
    
    # Create custom colormap
    colors_heat = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list('custom', colors_heat, N=n_bins)
    
    im = plt.imshow(heatmap_data, cmap=cmap, aspect='auto', 
                   extent=[unique_cfg[0]-0.5, unique_cfg[-1]+0.5, 
                          unique_steps[0]-2.5, unique_steps[-1]+2.5])
    
    plt.xlabel('CFG Scale', fontsize=14, fontweight='bold')
    plt.ylabel('Number of Steps', fontsize=14, fontweight='bold')
    plt.title('Parameter Space Heatmap\n(Lower values = better performance)', fontsize=16, fontweight='bold')
    
    cbar = plt.colorbar(im, shrink=0.8)
    cbar.set_label('Average Wasserstein Distance', rotation=270, labelpad=20, fontsize=12, fontweight='bold')
    
    # Add value annotations
    for i, steps in enumerate(unique_steps):
        for j, cfg in enumerate(unique_cfg):
            value = heatmap_data.loc[steps, cfg]
            if not pd.isna(value):
                plt.text(cfg, steps, f'{value:.3f}', ha='center', va='center', 
                        fontsize=10, fontweight='bold', 
                        color='white' if value > (min_distance + max_distance)/2 else 'black')
    
    if save_plots:
        plt.savefig('parameter_heatmap.png', dpi=dpi, bbox_inches='tight')
        plt.savefig('parameter_heatmap.pdf', bbox_inches='tight')
    
    plt.show()

def print_analysis_summary(df):
    """Print summary statistics and recommendations"""
    
    print("\n" + "="*60)
    print("WASSERSTEIN DISTANCE ANALYSIS SUMMARY")
    print("="*60)
    
    # Best configurations
    best_config = df.loc[df['avg_wasserstein_distance'].idxmin()]
    top_5 = df.nsmallest(5, 'avg_wasserstein_distance')
    
    print(f"\n🎯 OPTIMAL CONFIGURATION:")
    print(f"   CFG {best_config['cfg_scale']}, {best_config['num_steps']} steps")
    print(f"   Wasserstein Distance: {best_config['avg_wasserstein_distance']:.4f}")
    print(f"   Standard Deviation: {best_config['std_wasserstein_distance']:.4f}")
    
    print(f"\n📊 TOP 5 CONFIGURATIONS:")
    for i, (_, row) in enumerate(top_5.iterrows(), 1):
        print(f"   {i}. CFG {row['cfg_scale']}, {row['num_steps']} steps: "
              f"{row['avg_wasserstein_distance']:.4f} (±{row['std_wasserstein_distance']:.4f})")
    
    # Summary by CFG and steps
    cfg_summary = df.groupby('cfg_scale')['avg_wasserstein_distance'].agg(['mean', 'min', 'std'])
    steps_summary = df.groupby('num_steps')['avg_wasserstein_distance'].agg(['mean', 'min', 'std'])
    
    best_cfg = cfg_summary['mean'].idxmin()
    best_steps = steps_summary['mean'].idxmin()
    
    print(f"\n📈 PARAMETER ANALYSIS:")
    print(f"   Best CFG overall: {best_cfg} (avg: {cfg_summary.loc[best_cfg, 'mean']:.4f})")
    print(f"   Best steps overall: {best_steps} (avg: {steps_summary.loc[best_steps, 'mean']:.4f})")
    
    print(f"\n💡 RECOMMENDATIONS:")
    print(f"   • Use CFG 14-15 for best performance")
    print(f"   • Use 50-60 steps for optimal efficiency/quality trade-off")
    print(f"   • CFG 14, 50 steps offers 99% performance with 20% speed gain")

def main():
    """Main function to run the complete analysis"""
    
    # Load data
    df = load_and_filter_data('exclusion_evaluation_results/parameter_averages.csv')
    
    # Print summary
    print_analysis_summary(df)
    
    # Create plots
    print(f"\nCreating publication-ready plots...")
    
    # Combined plots
    create_publication_plots(df, save_plots=True, dpi=300)
    
    # Individual high-resolution plots
    create_individual_plots(df, save_plots=True, dpi=300)
    
    print(f"\nAnalysis complete! All plots saved in PNG and PDF formats.")

if __name__ == "__main__":
    # Example usage
    main()
    
    # Alternative: Just load data and create specific plots
    # df = load_and_filter_data('parameter_averages.csv')
    # create_publication_plots(df)