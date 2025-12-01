from config import PLOTS_DIR

import torch

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from .common import *
from .get_localizers import localizers
from validate.floc.utils.cluster import visualize_patches

def plot_all_rois(t_vals_dict, p_vals_dict, layer_positions, store_dir, figsize_per_panel=5, 
                  prefix='', suffix='', p_threshold=LOCALIZER_P_THRESHOLD, t_threshold=0, dpi=250):
    
    os.makedirs(store_dir, exist_ok=True)

    all_roi_colors = {
        "Faces_static_localizer": ("static-face", (0.75, 0.00, 0.00)),
        "Bodies_static_localizer": ("static-body", (0.00, 0.45, 0.00)),
        "Scenes_static_localizer": ("static-place", (0.80, 0.35, 0.00)),
        "Faces_moving_localizer": ("dynamic-face", (1.00, 0.80, 0.80)),
        "Bodies_moving_localizer": ("dynamic-body", (0.80, 1.00, 0.80)),
        "Scenes_moving_localizer": ("dynamic-place", (1.00, 0.90, 0.70)),
        "Faces_static": ("static-face", (0.75, 0.00, 0.00)),
        "Bodies_static": ("static-body", (0.00, 0.45, 0.00)),
        "Scenes_static": ("static-place", (0.80, 0.35, 0.00)),
        "Faces_moving": ("dynamic-face", (1.00, 0.80, 0.80)),
        "Bodies_moving": ("dynamic-body", (0.80, 1.00, 0.80)),
        "Scenes_moving": ("dynamic-place", (1.00, 0.90, 0.70)),
        "V6": ("V6", (0.0, 0.85, 0.95)),
        "MT-Huk": ("MT", (0.95, 0.20, 0.70)),
        "pSTS": ("pSTS", (0.85, 0.85, 0.0)),
    }

    roi_groups = {
        "face-response": ["Faces_static", "Faces_moving"],
        "body-response": ["Bodies_static", "Bodies_moving"],
        "place-response": ["Scenes_static", "Scenes_moving"],
        "face": ["Faces_static_localizer", "Faces_moving_localizer"],
        "body": ["Bodies_static_localizer", "Bodies_moving_localizer"],
        "place": ["Scenes_static_localizer", "Scenes_moving_localizer"],
        "motion": ["V6", "pSTS", "MT-Huk"],
        "categorical": [
            "Faces_static_localizer", "Bodies_static_localizer", "Scenes_static_localizer",
            "Faces_moving_localizer", "Bodies_moving_localizer", "Scenes_moving_localizer",
        ],
        "all": list(all_roi_colors.keys()),
    }

    # Get number of layers from first ROI
    first_roi = next(iter(p_vals_dict.keys()))
    n_layers = len(p_vals_dict[first_roi])
    
    for group, group_rois in roi_groups.items():
        # Filter to available ROIs
        available_rois = [roi for roi in group_rois if roi in p_vals_dict]
        if not available_rois:
            continue
            
        # Create figure
        fig, axes = plt.subplots(1, n_layers,
                                figsize=(n_layers * figsize_per_panel, figsize_per_panel))
        if n_layers == 1:
            axes = [axes]
        
        # Plot each ROI
        for roi_name in available_rois:
            roi_display_name, color = all_roi_colors[roi_name]
            p_vals_list = p_vals_dict[roi_name]
            t_vals_list = t_vals_dict[roi_name]
            
            for layer_idx, (p_vals, t_vals) in enumerate(zip(p_vals_list, t_vals_list)):
                pos = layer_positions[layer_idx]
                if isinstance(pos, torch.Tensor):
                    pos = pos.cpu().numpy()
                
                # Filter by p-value threshold
                if roi_name in ["V6", "MT-Huk"]:
                    # For V6 and MT, p-values were scaled up by 1000x
                    p_vals = p_vals * 1000.0
                mask = (p_vals.flatten() < p_threshold) & (t_vals.flatten() > t_threshold)
                pos_filtered = pos[mask]
                
                if len(pos_filtered) > 0:
                    axes[layer_idx].scatter(
                        pos_filtered[:, 0], 
                        pos_filtered[:, 1],
                        color=color, 
                        s=1.9,
                        alpha=0.8,
                        label=roi_display_name if layer_idx == 0 else None,
                        marker='s',
                        edgecolors='none'
                    )
        
        # Format axes
        for layer_idx, ax in enumerate(axes):
            ax.set_title(f'Layer {layer_idx}')
            ax.axis('equal')
            ax.set_aspect('equal', 'box')
        
        plt.suptitle(f'{group.capitalize()} ROIs', fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(f'{store_dir}/{prefix}{group}{suffix}.png', 
                    dpi=dpi, bbox_inches='tight')
        plt.close()
        
        print(f"Saved {group} ROI visualization to {store_dir}")


if __name__ == "__main__":
    ckpt_name = MODEL_CKPT
    store_dir = PLOTS_DIR / 'localizers'
    patches_dir = store_dir / 'patches'
    store_dir.mkdir(parents=True, exist_ok=True)
    patches_dir.mkdir(parents=True, exist_ok=True)

    t_vals_dict, p_vals_dict, layer_positions = localizers(ckpt_name, ret_merged=True)
    plot_all_rois(t_vals_dict, p_vals_dict, layer_positions, store_dir)


    categories_of_interest = [
        "Faces_moving_localizer",
        "Bodies_moving_localizer",
        "Scenes_moving_localizer",
        "V6",
        "pSTS",
        "MT-Huk"
    ]

    t_vals_dict = {cat: t_vals_dict[cat] for cat in categories_of_interest if cat in t_vals_dict}
    p_vals_dict = {cat: p_vals_dict[cat] for cat in categories_of_interest if cat in p_vals_dict}

    visualize_patches(
        t_vals_dict=t_vals_dict,
        p_vals_dict=p_vals_dict,
        layer_positions=layer_positions,
        viz_dir=str(patches_dir),
        prefix='patches_',
        suffix='',
        t_threshold=0,
        p_threshold=LOCALIZER_P_THRESHOLD,
    )