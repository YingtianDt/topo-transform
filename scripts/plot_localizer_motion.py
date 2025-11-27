from config import PLOTS_DIR

import torch

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from .common import *
from .get_localizers import localizers, get_localizer_human, get_localizer_model

ROBERT_STATS = '/mnt/scratch/ytang/datasets/fsaverage_surfaces_robert'

def load_robert_tvals():
    t_vals = []
    for individual in os.listdir(ROBERT_STATS):
        if not individual.endswith('.npy'):
            continue
        t_val = np.load(os.path.join(ROBERT_STATS, individual))
        t_vals.append(t_val)
    t_vals = np.array(t_vals).mean(axis=0)  # shape: (n_units,)

    # from matplotlib import pyplot as plt
    # from nilearn import datasets, plotting
    # fsaverage = datasets.fetch_surf_fsaverage(mesh='fsaverage5')
    # # Plot Left Hemisphere
    # plotting.plot_surf_stat_map(
    #     surf_mesh=fsaverage.pial_left,
    #     stat_map=-t_vals[:10242],
    #     hemi='left',
    #     bg_map=fsaverage.sulc_left,
    #     title='Robert t-values (Left Hemisphere)',
    #     colorbar=True,
    #     cmap='Spectral',
    # )
    # plt.savefig(PLOTS_DIR / "robert_left.png")
    # plt.close()

    # plotting.plot_surf_stat_map(
    #     surf_mesh=fsaverage.pial_left,
    #     stat_map=-t_vals[:10242],
    #     hemi='left',
    #     bg_map=fsaverage.sulc_left,
    #     title='Robert t-values (Left Hemisphere)',
    #     view='ventral',
    #     colorbar=True,
    #     cmap='Spectral',
    # )
    # plt.savefig(PLOTS_DIR / "robert_ventral.png")
    # plt.close()

    return t_vals

def plot_all_rois(all_t_vals, rois, store_dir, p_threshold=0.05):
    os.makedirs(store_dir, exist_ok=True)

    masks_model = get_localizer_model(rois, MODEL_CKPT, p_thres=p_threshold)
    masks_human = get_localizer_human(rois)
    t_vals_robert = load_robert_tvals()

    all_means_model = []
    means_model = []
    means_human = []
    stds_model = []
    stds_human = []
    for i, roi, mask_model, mask_human in zip(range(len(rois)), rois, masks_model, masks_human):
        
        t_vals_models = []
        for t_vals in all_t_vals:
            t_vals_model = [t_val[mask] for t_val, mask in zip(t_vals, mask_model)]  # layers
            # here just choose the first layer
            t_vals_model = t_vals_model[0]  # shape: (n_units,)
            t_vals_models.append(t_vals_model)

        t_vals_models = np.array(t_vals_models)  # shape: (n_checkpoints, n_units)
        t_vals_human = t_vals_robert[mask_human]  # shape: (n_units)

        # from matplotlib import pyplot as plt
        # from nilearn import datasets, plotting
        # fsaverage = datasets.fetch_surf_fsaverage(mesh='fsaverage5')
        # # Plot Left Hemisphere
        # plotting.plot_surf_stat_map(
        #     surf_mesh=fsaverage.pial_left,
        #     stat_map=mask_human[:10242],
        #     hemi='left',
        #     bg_map=fsaverage.sulc_left,
        #     title='Robert t-values (Left Hemisphere)',
        #     colorbar=True
        # )
        # plt.savefig(f"{roi}.png")
        # plt.close()

        # plotting.plot_surf_stat_map(
        #     surf_mesh=fsaverage.pial_left,
        #     stat_map=mask_human[:10242],
        #     hemi='left',
        #     bg_map=fsaverage.sulc_left,
        #     title='Robert t-values (Left Hemisphere)',
        #     view='ventral',
        #     colorbar=True
        # )
        # plt.savefig(f"{roi}_ventral.png")
        # plt.close()

        portion_models = (t_vals_models > 0)
        portion_human = (t_vals_human > 0)

        mean_models = portion_models.mean(-1)
        mean_model = mean_models.mean(0)
        std_model = mean_models.std(0)
        mean_human = portion_human.mean(-1)

        all_means_model.append(mean_models)
        means_model.append(mean_model)
        stds_model.append(std_model)
        means_human.append(mean_human)

    plt.figure(figsize=(4, 3))
    x = np.arange(len(rois))
    width = 0.35

    plt.bar(x - width/2, means_model, yerr=np.array(stds_model)*1.5, label='Model', capsize=5, color=MODEL_C)
    plt.bar(x + width/2, means_human, width, label='Human', capsize=5, color=HUMAN_C, alpha=0.7)

    for i in range(len(rois)):
        plt.scatter([x[i] - width/2]*len(all_means_model[i]), all_means_model[i], color='k')

    plt.xticks(x, rois)
    plt.ylabel('Mean t-value')
    plt.title('Localizer t-values by ROI')
    plt.legend()
    plt.tight_layout()
    plt.savefig(store_dir / 'localizer_tvals_comparison.svg')
    plt.close()

    print(f"Saved localizer t-values comparison plot to {store_dir / 'localizer_tvals_comparison.svg'}")

if __name__ == "__main__":
    ckpt_names = MODEL_CKPTS
    store_dir = PLOTS_DIR
    store_dir.mkdir(parents=True, exist_ok=True)

    all_t_vals = []
    for ckpt_name in ckpt_names:
        print(f"Processing checkpoint: {ckpt_name}")
        t_vals_dicts, p_vals_dicts, layer_positions = localizers(ckpt_name, ret_merged=True)
        t_vals = t_vals_dicts['robert']
        all_t_vals.append(t_vals)

        # # plot the t values for all
        # from matplotlib import pyplot as plt
        # pos = layer_positions[0]
        # plt.scatter(x=pos[:, 0], y=pos[:, 1], c=t_vals, cmap='Spectral', s=1)
        # plt.colorbar(label='t-value')
        # plt.title('Localizer t-values (Robert Dataset)')
        # plt.xlabel('Layer Position X')
        # plt.ylabel('Layer Position Y')
        # plt.gca().set_aspect('equal', adjustable='box')
        # plt.savefig(store_dir / 'localizer_tvals_robert.png')
        # plt.close()

    rois = [
        'face',
        'body',
        'place',
        'v6',
        'psts',
        # 'mt',
    ]

    plot_all_rois(all_t_vals, rois, store_dir)