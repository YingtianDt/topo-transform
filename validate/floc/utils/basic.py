import torch

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


def visualize_tvals(t_vals_dict, layer_positions, store_dir, figsize_per_panel=5, prefix='', suffix='', 
                    topk_percent=100):
    """Visualize t-statistics for each category and layer.
    
    Args:
        t_vals_dict: Dict mapping category names to list of t-value arrays (one per layer)
        layer_positions: List of position arrays for each layer [num_units, 2]
        store_dir: Directory to save visualizations
        figsize_per_panel: Size of each subplot panel
        topk_percent: float, percentage of units to highlight (default: 100)
    """
    os.makedirs(store_dir, exist_ok=True)
    
    categories = list(t_vals_dict.keys())
    n_layers = len(t_vals_dict[categories[0]]) if categories else 0
    
    # Create separate figure for each category
    for cat_name, t_vals_list in t_vals_dict.items():
        if not isinstance(t_vals_list, list):
            t_vals_list = [t_vals_list]
        
        n_cols = len(t_vals_list)
        fig, axes = plt.subplots(1, n_cols, 
                                figsize=(n_cols * figsize_per_panel, figsize_per_panel))
        
        # Ensure axes is always a list
        if n_cols == 1:
            axes = [axes]
        
        # Compute global normalization across all layers for this category
        all_t_vals = np.concatenate([t_vals.flatten() for t_vals in t_vals_list])
        vmax_global = np.abs(all_t_vals).max()
        norm_global = Normalize(vmin=-vmax_global, vmax=vmax_global)
        
        # Compute global threshold for topk_percent across all layers
        if topk_percent < 100:
            k_global = int(len(all_t_vals) * topk_percent / 100)
            threshold_global = np.partition(all_t_vals, -k_global)[-k_global]
        else:
            threshold_global = -np.inf
        
        for layer_idx, t_vals in enumerate(t_vals_list):
            pos = layer_positions[layer_idx]
            if isinstance(pos, torch.Tensor):
                pos = pos.cpu().numpy()
            
            # Apply topk_percent filtering using global threshold
            t_vals_flat = t_vals.flatten()
            if topk_percent < 100:
                mask = t_vals_flat >= threshold_global
                pos_filtered = pos[mask]
                t_vals_filtered = t_vals_flat[mask]
            else:
                pos_filtered = pos
                t_vals_filtered = t_vals_flat
            
            # Create scatter plot with global normalization
            im = axes[layer_idx].scatter(pos_filtered[:, 0], pos_filtered[:, 1], 
                                        c=t_vals_filtered, 
                                        cmap='bwr', norm=norm_global, s=0.1)
            title = f'Layer {layer_idx}'
            if topk_percent < 100:
                title += f' (top {topk_percent}%)'
            axes[layer_idx].set_title(title)
            axes[layer_idx].axis('equal')
            axes[layer_idx].set_aspect('equal', 'box')
            
        # Add colorbar
        fig.colorbar(im, ax=axes, orientation='horizontal', fraction=0.05, label='t-statistic')
        plt.suptitle(f'{cat_name} (one-vs-rest)', fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(f'{store_dir}/{prefix}tvals_{cat_name}{suffix}.png', 
                    dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Saved visualizations to {store_dir}")