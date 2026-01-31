from config import PLOTS_DIR

import torch

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from .common import *
from .get_localizers import localizers
from validate.floc.utils.cluster import visualize_patches

def plot_all_rois(
    t_vals_dict, 
    p_vals_dict, 
    layer_positions, 
    store_dir, 
    figsize_per_panel=3, 
    prefix='', 
    suffix='', 
    p_threshold=LOCALIZER_P_THRESHOLD, 
    t_threshold=LOCALIZER_T_THRESHOLD, 
    dpi=400
):
    
    os.makedirs(store_dir, exist_ok=True)

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
                
                if roi_name in ["MT-Huk", "V6", "V6-enhanced"]:
                    t_threshold_used = LOCALIZER_FLOW_T_THRESHOLD
                elif roi_name in ["pSTS"]:
                    t_threshold_used = LOCALIZER_BIOMOTION_T_THRESHOLD
                else:
                    t_threshold_used = t_threshold

                mask = (p_vals.flatten() < p_threshold) & (t_vals.flatten() > t_threshold_used)
                pos_filtered = pos[mask]
                
                if len(pos_filtered) > 0:

                    axes[layer_idx].scatter(
                        pos_filtered[:, 0], 
                        pos_filtered[:, 1],
                        color=color, 
                        s=2.4,
                        alpha=1,
                        label=roi_display_name if layer_idx == 0 else None,
                        marker='s',
                        edgecolors='none'
                    )

                    axes[layer_idx].scatter(
                        pos_filtered[:, 0], 
                        pos_filtered[:, 1],
                        color=color, 
                        s=2.6,
                        alpha=.9,
                        label=roi_display_name if layer_idx == 0 else None,
                        marker='s',
                        edgecolors='none'
                    )

        # Format axes
        for layer_idx, ax in enumerate(axes):
            ax.axis('equal')
            ax.set_aspect('equal', 'box')
            ax.set_xlim([0, 210])
            ax.set_ylim([0, 70])
            # remove ticks
            if group in ['face', 'body']:
                ax.set_xticks([])
            elif group in ['pSTS']:
                ax.set_yticks([])
            elif group in ['place']:
                pass
            else:
                ax.set_xticks([])
                ax.set_yticks([])

            if group == "MT":
                for spine in ax.spines.values():
                    spine.set_color('gray')
                ax.tick_params(axis='both', colors='gray')
        
        plt.tight_layout()
        plt.savefig(f'{store_dir}/{prefix}{group}{suffix}.png', 
                    dpi=dpi, bbox_inches='tight', transparent=True)
        plt.close()
        
        print(f"Saved {group} ROI visualization to {store_dir}")

    # make a standalone legend plot
    legend_rois = [
        'face','body','place','v6','psts','mt'
    ]
    display_names = [
        "Face","Body","Place","V6","pSTS","MT"
    ]
    fig_legend, ax_legend = plt.subplots(figsize=(4, 2))
    for roi_name, display_name in zip(legend_rois, display_names):
        roi_display_name, color = all_roi_colors[roi_name]
        ax_legend.scatter([], [], color=color, s=50, label=display_name, marker='s', edgecolors='none')
    ax_legend.legend(frameon=False, loc='center', ncol=3)
    ax_legend.axis('off')
    plt.savefig(f'{store_dir}/{prefix}legend{suffix}.svg', 
                dpi=dpi, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    ckpt_name = MODEL_CKPT
    store_dir = PLOTS_DIR / 'localizers'
    patches_dir = store_dir / 'patches'
    store_dir.mkdir(parents=True, exist_ok=True)
    patches_dir.mkdir(parents=True, exist_ok=True)

    t_vals_dict, p_vals_dict, layer_positions = localizers(ckpt_name, ret_merged=True)
    plot_all_rois(t_vals_dict, p_vals_dict, layer_positions, store_dir)


    # categories_of_interest = [
    #     "Faces_moving_localizer",
    #     "Bodies_moving_localizer",
    #     "Scenes_moving_localizer",
    #     "V6",
    #     "pSTS",
    #     "MT-Huk"
    # ]

    # t_vals_dict = {cat: t_vals_dict[cat] for cat in categories_of_interest if cat in t_vals_dict}
    # p_vals_dict = {cat: p_vals_dict[cat] for cat in categories_of_interest if cat in p_vals_dict}

    # visualize_patches(
    #     t_vals_dict=t_vals_dict,
    #     p_vals_dict=p_vals_dict,
    #     layer_positions=layer_positions,
    #     viz_dir=str(patches_dir),
    #     prefix='patches_',
    #     suffix='',
    #     t_threshold=0,
    #     p_threshold=LOCALIZER_P_THRESHOLD,
    # )