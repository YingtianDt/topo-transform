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
    t_vals = np.array(t_vals)
    t_vals_mean = t_vals.mean(0)

    absvmax = np.max(np.abs(t_vals_mean))

    # from matplotlib import pyplot as plt
    # from nilearn import datasets, plotting
    # fsaverage = datasets.fetch_surf_fsaverage(mesh='fsaverage5')
    # # Plot Left Hemisphere
    # plotting.plot_surf_stat_map(
    #     surf_mesh=fsaverage.pial_left,
    #     stat_map=t_vals_mean[:10242],
    #     hemi='left',
    #     bg_map=fsaverage.sulc_left,
    #     title='Robert t-values (Left Hemisphere)',
    #     colorbar=True,
    #     cmap='Spectral',
    #     vmin=-absvmax,
    #     vmax=absvmax,
    # )
    # plt.savefig(PLOTS_DIR / "robert_left.png")
    # plt.close()

    # plotting.plot_surf_stat_map(
    #     surf_mesh=fsaverage.pial_left,
    #     stat_map=t_vals_mean[:10242],
    #     hemi='left',
    #     bg_map=fsaverage.sulc_left,
    #     title='Robert t-values (Left Hemisphere)',
    #     view='ventral',
    #     colorbar=True,
    #     cmap='Spectral',
    #     vmin=-absvmax,
    #     vmax=absvmax,
    # )
    # plt.savefig(PLOTS_DIR / "robert_ventral.png")
    # plt.close()

    return t_vals

def plot_all_rois(all_t_vals, ckpts, rois, store_dir=None, p_threshold=LOCALIZER_P_THRESHOLD, t_threshold=LOCALIZER_T_THRESHOLD):
    if store_dir is not None:
        os.makedirs(store_dir, exist_ok=True)

    masks_models = [[] for _ in rois]
    for ckpt in ckpts:
        masks_model = get_localizer_model(rois, ckpt, p_thres=p_threshold, t_thres=t_threshold)
        for r, _ in enumerate(rois):
            masks_models[r].append(masks_model[r])
            
    masks_human = get_localizer_human(rois)
    t_vals_robert = load_robert_tvals()

    all_means_model = []
    all_means_human = []
    means_model = []
    means_human = []
    stds_model = []
    stds_human = []
    for i, roi, mask_model, mask_human in zip(range(len(rois)), rois, masks_models, masks_human):
        
        t_vals_models = []
        for t_vals, mask in zip(all_t_vals, mask_model):
            t_vals_model = [t_val[mask] for t_val, mask in zip(t_vals, mask)]  # layers
            # here just choose the first layer
            t_vals_model = t_vals_model[0]  # shape: (n_units,)
            t_vals_models.append(t_vals_model)

        mean_models = np.array([(t_vals>0).mean() for t_vals in t_vals_models])  # shape: (n_checkpoints)
        mean_humans = np.array([(t_vals[mask_human]>0).mean() for t_vals in t_vals_robert]) # shape: (n_individuals)
        
        # sometimes model have no units in the ROI
        mean_models = np.nan_to_num(mean_models, nan=0.0)

        mean_model = mean_models.mean(0)
        std_model = mean_models.std(0)
        mean_human = mean_humans.mean(0)
        std_human = mean_humans.std(0)

        all_means_model.append(mean_models)
        means_model.append(mean_model)
        stds_model.append(std_model)
        all_means_human.append(mean_humans)
        means_human.append(mean_human)
        stds_human.append(std_human)

    # compute mae between model and human
    mae = np.mean(np.abs(np.array(means_model) - np.array(means_human)))
    print(f"Mean Absolute Error (mae) between model and human: {mae:.4f}")

    if store_dir is None:
        return mae

    plt.figure(figsize=(4, 3))
    x = np.arange(len(rois))
    width = 0.35

    plt.bar(x - width/2, means_model, width, yerr=np.array(stds_model)*1.5, label='Model', capsize=5, color=MODEL_C)
    plt.bar(x + width/2, means_human, width, yerr=np.array(stds_human)*1.5, label='Human', capsize=5, color=HUMAN_C)

    for i in range(len(rois)):
        plt.scatter([x[i] - width/2]*len(all_means_model[i]), all_means_model[i], color='k')
        plt.scatter([x[i] + width/2]*len(all_means_human[i]), all_means_human[i], color='k')

    plt.xticks(x, rois)
    plt.ylabel('Motion Index')
    plt.legend()
    plt.tight_layout()
    plt.savefig(store_dir / 'localizer_tvals_comparison.svg')
    plt.close()

    print(f"Saved localizer t-values comparison plot to {store_dir / 'localizer_tvals_comparison.svg'}")

    return mae

if __name__ == "__main__":
    store_dir = PLOTS_DIR
    store_dir.mkdir(parents=True, exist_ok=True)

    rois = [
        'face',
        'body',
        'place',
        'v6',
        'psts',
    ]

    all_t_vals = []
    for ckpt_name in MODEL_CKPTS:
        print(f"Processing checkpoint: {ckpt_name}")
        t_vals_dicts, p_vals_dicts, layer_positions = localizers(ckpt_name, ret_merged=True)
        t_vals = t_vals_dicts['robert']
        all_t_vals.append(t_vals)

        if ckpt_name == MODEL_CKPT:
            # plot the t values for all
            from matplotlib import pyplot as plt
            pos = layer_positions[0]
            plt.scatter(x=pos[:, 0], y=pos[:, 1], c=t_vals, cmap='Spectral', s=1)
            plt.colorbar(label='t-value')
            plt.title('Localizer t-values (Robert Dataset)')
            plt.xlabel('Layer Position X')
            plt.ylabel('Layer Position Y')
            plt.gca().set_aspect('equal', adjustable='box')
            plt.savefig(store_dir / 'localizer_tvals_robert.png')
            plt.close()
            print("Model t vals saved.")

    model_mae = plot_all_rois(all_t_vals, MODEL_CKPTS, rois, store_dir)

    all_t_vals = []
    for ckpt_name in TDANN_CKPTS:
        print(f"Processing checkpoint: {ckpt_name}")
        t_vals_dicts, p_vals_dicts, layer_positions = localizers(ckpt_name, ret_merged=True)
        t_vals = t_vals_dicts['robert']
        all_t_vals.append(t_vals)

    tdann_mae = plot_all_rois(all_t_vals, TDANN_CKPTS, rois, store_dir=None)

    all_t_vals = []
    for ckpt_name in UNOPTIMIZED_CKPTS:
        print(f"Processing checkpoint: {ckpt_name}")
        t_vals_dicts, p_vals_dicts, layer_positions = localizers(ckpt_name, ret_merged=True)
        t_vals = t_vals_dicts['robert']
        all_t_vals.append(t_vals)

    unoptimized_mae = plot_all_rois(all_t_vals, UNOPTIMIZED_CKPTS, rois, store_dir=None)

    # plot bar comparison
    plt.figure(figsize=(3, 3))
    plt.bar(['Model', 'TDANN', 'Unoptimized'], [model_mae, tdann_mae, unoptimized_mae], color=[MODEL_C, DEFAULT_C, DEFAULT_C])
    plt.ylabel('Mean Absolute Error (mae)')
    plt.title('Localizer Motion Index mae Comparison')
    plt.tight_layout()
    plt.savefig(store_dir / 'localizer_motion_mae_comparison.svg')
    plt.close()

    print(f"Saved localizer motion mae comparison plot to {store_dir / 'localizer_motion_mae_comparison.svg'}")