from config import PLOTS_DIR

import torch

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from .common import MODEL_CKPT
from .get_localizers import localizers


def plot_all_rois(t_vals_dicts, layer_positions, store_dir, figsize_per_panel=5, 
                  prefix='', suffix='', topk_percent=1, t_threshold=5, dpi=250):

    os.makedirs(store_dir, exist_ok=True)

    all_roi_colors = {
        # pitcher
        "Faces_static_localizer": ("static-face", (0.75, 0.00, 0.00)),        # crimson
        "Bodies_static_localizer": ("static-body", (0.00, 0.45, 0.00)),        # dark green
        "Objects_static_localizer": ("static-object", (0.00, 0.20, 0.65)),    # navy blue
        "Scenes_static_localizer": ("static-place", (0.80, 0.35, 0.00)),      # burnt orange

        "Faces_moving_localizer": ("dynamic-face", (1.00, 0.80, 0.80)),      # blush pink
        "Bodies_moving_localizer": ("dynamic-body", (0.80, 1.00, 0.80)),     # pale mint
        "Objects_moving_localizer": ("dynamic-object", (0.75, 0.85, 1.00)),  # sky blue
        "Scenes_moving_localizer": ("dynamic-place", (1.00, 0.90, 0.70)),    # light peach
        
        # psts, v6
        "V6": ("V6", (0.0, 0.85, 0.95)),                  # bright cyan (cool)
        "MT": ("MT", (0.95, 0.20, 0.70)),                 # vibrant magenta (warm)
        "pSTS": ("pSTS", (0.85, 0.85, 0.0)),              # bright yellow (neutral)
    }

    for group, group_rois in {
        "face": ["Faces_static_localizer", "Faces_moving_localizer"],
        "body": ["Bodies_static_localizer", "Bodies_moving_localizer"],
        "object": ["Objects_static_localizer", "Objects_moving_localizer"],
        "place": ["Scenes_static_localizer", "Scenes_moving_localizer"],
        "motion": ["V6", "MT", "pSTS"],
        "categorical": [
            "Faces_static_localizer", "Bodies_static_localizer", "Objects_static_localizer", "Scenes_static_localizer",
            "Faces_moving_localizer", "Bodies_moving_localizer", "Objects_moving_localizer", "Scenes_moving_localizer",
        ],  
        "all": list(all_roi_colors.keys()),
    }.items():
        roi_colors = {roi: all_roi_colors[roi] for roi in group_rois}

        # Handle both single dict and list of dicts
        if isinstance(t_vals_dicts, dict):
            t_vals_dicts = [t_vals_dicts]
        
        # Combine all dicts
        combined_t_vals_dict = {}
        for d in t_vals_dicts:
            combined_t_vals_dict.update(d)
        
        # Filter to only ROIs in roi_colors
        combined_t_vals_dict = {roi_name: combined_t_vals_dict[roi_name] 
                            for roi_name in roi_colors.keys() 
                            if roi_name in combined_t_vals_dict}
        
        # Determine number of layers
        first_roi = list(combined_t_vals_dict.keys())[0]
        t_vals_list = combined_t_vals_dict[first_roi]
        if not isinstance(t_vals_list, list):
            t_vals_list = [t_vals_list]
        n_layers = len(t_vals_list)
        
        # Calculate number of rows: 1 row for all categorical ROIs
        n_rows = 1
        
        # Create figure
        fig, axes = plt.subplots(n_rows, n_layers,
                                figsize=(n_layers * figsize_per_panel, n_rows * figsize_per_panel))
        
        if n_rows == 1 and n_layers == 1:
            axes = np.array([[axes]])
        if n_layers == 1:
            axes = axes.reshape(n_rows, 1)
        elif n_rows == 1:
            axes = axes.reshape(1, n_layers)
        
        # Track which ROIs we've added to legend
        legend_added = set()

        # Plot all ROIs
        for roi_name in combined_t_vals_dict.keys():
            t_vals_list = combined_t_vals_dict[roi_name]
            if not isinstance(t_vals_list, list):
                t_vals_list = [t_vals_list]
            
            # Compute threshold for this ROI
            all_t_vals = np.concatenate([t_vals.flatten() for t_vals in t_vals_list])
            if topk_percent < 100:
                k = int(len(all_t_vals) * topk_percent / 100)
                threshold = np.partition(all_t_vals, -k)[-k]
            else:
                threshold = -np.inf

            if t_threshold is not None:
                threshold = t_threshold
            
            for layer_idx, t_vals in enumerate(t_vals_list):
                pos = layer_positions[layer_idx]
                if isinstance(pos, torch.Tensor):
                    pos = pos.cpu().numpy()

                pos_x_lim = (np.nanmin(pos[:, 0]), np.nanmax(pos[:, 0]))
                pos_y_lim = (np.nanmin(pos[:, 1]), np.nanmax(pos[:, 1]))
                
                # Filter by threshold
                t_vals_flat = t_vals.flatten()
                mask = t_vals_flat >= threshold
                pos_filtered = pos[mask]
                t_vals_filtered = t_vals_flat[mask]
                
                # Normalize t-values to [0, 1] for alpha intensity
                if len(t_vals_filtered) > 0:
                    t_norm = (t_vals_filtered - t_vals_filtered.min()) / \
                            (t_vals_filtered.max() - t_vals_filtered.min() + 1e-8)

                    # Only add label for first layer and first occurrence of this ROI
                    add_label = (layer_idx == 0 and roi_name not in legend_added)
                    
                    roi_display_name, color = roi_colors[roi_name]

                    axes[0, layer_idx].scatter(
                        pos_filtered[:, 0], 
                        pos_filtered[:, 1],
                        color=color, 
                        s=0.5,
                        # alpha=(0.3 + 0.7 * t_norm),  # Scale alpha by t-value
                        alpha=0.8,
                        label=roi_display_name if add_label else None
                    )
                    
                    if add_label:
                        legend_added.add(roi_display_name)
        
        # Format axes
        for layer_idx in range(n_layers):
            title = f'Layer {layer_idx}'
            axes[0, layer_idx].set_title(title)
            axes[0, layer_idx].axis('equal')
            axes[0, layer_idx].set_aspect('equal', 'box')
            axes[0, layer_idx].set_xlim(pos_x_lim)
            axes[0, layer_idx].set_ylim(pos_y_lim)
        
        # # Add legend
        # if legend_added:
        #     axes[0, 0].legend(
        #         loc='upper right',
        #         bbox_to_anchor=(-0.25, 1.0),  # shift left outside plot
        #         fontsize=12,
        #         borderpad=1
        #     )
        
        plt.suptitle('All ROIs Combined', fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(f'{store_dir}/{prefix}{group}{suffix}.png', 
                    dpi=dpi, bbox_inches='tight')
        plt.close()
        
        print(f"Saved combined ROI {group} visualization to {store_dir}")


if __name__ == "__main__":
    ckpt_name = MODEL_CKPT
    store_dir = PLOTS_DIR / 'localizers'
    store_dir.mkdir(parents=True, exist_ok=True)

    t_vals_dicts, layer_positions = localizers(ckpt_name)
    plot_all_rois(t_vals_dicts, layer_positions, store_dir)