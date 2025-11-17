import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import config
from data import Kinetics400
from topo import TopoTransformedVJEPA

from .get_validate_features import validate_features
from config import CACHE_DIR, PLOTS_DIR


def plot_distance_similarity(
    features, 
    positions, 
    save_path='distance_similarity.png', 
    n_bins=11, 
    subsample=10000,  # the cost scales O(N^2), so subsample for efficiency ??
    max_distance=65,  # the stats over this are basically ~0 mean with decreasing variance
):
    """Plot cortical distance vs response similarity with box plots and scatter points"""
    # Convert to numpy and flatten batch dimension
    positions = positions.numpy()  # (N, 2)

    has_time = features.ndim == 5
    if has_time:
        B, T, C, H, W = features.shape
        features = features.reshape(B * T, C * H * W)
    else:
        B, C, H, W = features.shape
        features = features.reshape(B, C * H * W)

    features = features.T  # (N, D)

    # Subsample for efficiency
    if subsample is not None and features.shape[0] > subsample:
        indices = np.random.choice(features.shape[0], subsample, replace=False)
        features = features[indices]
        positions = positions[indices]

    # Normalize features
    features = (features - features.mean(axis=1, keepdims=True)) / (features.std(axis=1, keepdims=True) + 1e-10)
    
    # Compute pairwise distances and similarities
    cortical_dist = ((positions[:, None, :] - positions[None, :, :])**2).sum(axis=-1)**0.5
    response_sim = (features @ features.T) / features.shape[1]
    
    # Get upper triangle indices (exclude diagonal)
    triu_indices = np.triu_indices_from(cortical_dist, k=1)
    cortical_dist_flat = cortical_dist[triu_indices]
    response_sim_flat = response_sim[triu_indices]
    
    # Filter by max distance if specified
    if max_distance is not None:
        mask = cortical_dist_flat <= max_distance
        cortical_dist_flat = cortical_dist_flat[mask]
        response_sim_flat = response_sim_flat[mask]
    
    # Bin distances
    bins = np.linspace(cortical_dist_flat.min(), cortical_dist_flat.max(), n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_width = bins[1] - bins[0]
    bin_indices = np.digitize(cortical_dist_flat, bins)
    
    # Group similarities by bin
    binned_data = [response_sim_flat[bin_indices == i] for i in range(1, len(bins))]
    
    # Plot with improved aesthetics
    fig, ax = plt.subplots(figsize=(4, 4))
    
    # Set style
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    
    # Add scatter points first (so they're behind boxes)
    np.random.seed(42)
    for i, data in enumerate(binned_data):
        if len(data) > 0:
            # Subsample points if too many
            max_points = 150
            if len(data) > max_points:
                sample_idx = np.random.choice(len(data), max_points, replace=False)
                data_to_plot = data[sample_idx]
            else:
                data_to_plot = data
            
            # Add jitter to x positions
            x_jitter = np.random.normal(0, bin_width*0.12, len(data_to_plot))
            x_pos = bin_centers[i] + x_jitter
            
            ax.scatter(x_pos, data_to_plot, alpha=0.5, s=5, c='#FA0022', 
                      edgecolors='none', rasterized=True)
    
    # Create box plot
    bp = ax.boxplot(binned_data, positions=bin_centers, widths=bin_width*0.5,
                     patch_artist=True, showfliers=False,
                     boxprops=dict(facecolor='#E4E4E4', edgecolor='#1A1A1A', 
                                  linewidth=1.5, alpha=0.9),
                     whiskerprops=dict(color='#1A1A1A', linewidth=1.5),
                     capprops=dict(color='#1A1A1A', linewidth=1.5),
                     medianprops=dict(color='#E63946', linewidth=1.5))
    
    # # Add subtle grid
    # ax.yaxis.grid(True, linestyle='--', alpha=0.3, linewidth=0.8, color='gray')
    # ax.set_axisbelow(True)
    
    # Labels and title
    ax.set_xlabel('Cortical Distance', fontsize=13)
    ax.set_ylabel('Response Similarity', fontsize=13)
    ax.set_title('Spatial Loss Encourages Local Correlations', 
                fontsize=14, pad=15)
    # ax.set_ylim([-1.05, 1.05])
    
    # Set x-ticks - show 5-7 evenly spaced ticks
    n_ticks = min(n_bins//2+1, n_bins)
    tick_indices = np.linspace(0, len(bin_centers)-1, n_ticks, dtype=int)
    tick_positions = bin_centers[tick_indices]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([f'{x:.2f}' for x in tick_positions], fontsize=11)
    
    # Improve tick appearance
    ax.tick_params(axis='both', which='major', labelsize=11, 
                   width=2, length=6, color='#1A1A1A')
    ax.tick_params(axis='y', labelsize=11)
    
    # Axes width
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(2)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=400, bbox_inches='tight', facecolor='none')
    print(f"Saved to {save_path}")
    plt.close()


if __name__ == '__main__':
    all_features, positions = validate_features('best_transformed_model_vjepa_14_18_22_single_lr1e-4_bs32.pt')
    
    # Plot for first layer: in a single-sheet setting, this is the only "layer"
    layer_features = all_features[0]
    positions = positions[0]
    plot_distance_similarity(layer_features, positions.coordinates, 
                            PLOTS_DIR / 'plot_wiring_cost.svg')