from config import PLOTS_DIR

import torch

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from .common import MODEL_CKPT
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

    from matplotlib import pyplot as plt
    from nilearn import datasets, plotting
    fsaverage = datasets.fetch_surf_fsaverage(mesh='fsaverage5')
    # Plot Left Hemisphere
    plotting.plot_surf_stat_map(
        surf_mesh=fsaverage.pial_left,
        stat_map=t_vals[:10242],
        hemi='left',
        bg_map=fsaverage.sulc_left,
        title='Robert t-values (Left Hemisphere)',
        colorbar=True
    )
    plt.savefig("test.png")
    plt.close()

    plotting.plot_surf_stat_map(
        surf_mesh=fsaverage.pial_left,
        stat_map=t_vals[:10242],
        hemi='left',
        bg_map=fsaverage.sulc_left,
        title='Robert t-values (Left Hemisphere)',
        view='ventral',
        colorbar=True
    )
    plt.savefig("test_ventral.png")
    plt.close()

    return t_vals

def plot_all_rois(t_vals, rois, store_dir, t_percentage=1, t_threshold=None):
    os.makedirs(store_dir, exist_ok=True)

    masks_model = get_localizer_model(rois, MODEL_CKPT, t_perc=t_percentage, t_thres=t_threshold)
    masks_human = get_localizer_human(rois)
    t_vals_robert = load_robert_tvals()

    means_model = []
    means_human = []
    stds_model = []
    stds_human = []
    for i, roi, mask_model, mask_human in zip(range(len(rois)), rois, masks_model, masks_human):
        
        t_vals_model = [t_val[mask] for t_val, mask in zip(t_vals, mask_model)]  # layers
        # here just choose the first layer
        t_vals_model = t_vals_model[0]  # shape: (n_units,)

        t_vals_human = t_vals_robert[mask_human]  # shape: (n_units)

        portion_model = (t_vals_model > 0)
        portion_human = (t_vals_human > 0)

        mean_model = portion_model.mean()
        mean_human = portion_human.mean()
        std_model = (mean_model * (1-mean_model))**0.5
        std_human = (mean_human * (1-mean_human))**0.5

        means_model.append(mean_model)
        means_human.append(mean_human)
        stds_model.append(std_model)
        stds_human.append(std_human)

    plt.figure(figsize=(8, 6))
    x = np.arange(len(rois))
    width = 0.35
    plt.bar(x - width/2, means_model, width, yerr=stds_model, label='Model', capsize=5)
    plt.bar(x + width/2, means_human, width, yerr=stds_human, label='Human', capsize=5)
    plt.xticks(x, rois)
    plt.ylabel('Mean t-value')
    plt.title('Localizer t-values by ROI')
    plt.legend()
    plt.tight_layout()
    plt.savefig(store_dir / 'localizer_tvals_comparison.svg')
    plt.close()

    print(f"Saved localizer t-values comparison plot to {store_dir / 'localizer_tvals_comparison.svg'}")

if __name__ == "__main__":
    ckpt_name = MODEL_CKPT
    store_dir = PLOTS_DIR
    store_dir.mkdir(parents=True, exist_ok=True)

    t_vals_dicts, layer_positions = localizers(ckpt_name, ret_merged=True)
    t_vals = t_vals_dicts['robert']

    rois = [
        'face',
        'v6',
        'psts',
        'mt',
    ]

    plot_all_rois(t_vals, rois, store_dir)