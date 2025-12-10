from config import PLOTS_DIR

import torch

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from config import ROBERT_STATS
from .common import *
from .get_localizers import localizers, get_localizer_human, get_localizer_model


def load_robert_tvals():
    t_vals = []
    for individual in os.listdir(ROBERT_STATS):
        if not individual.endswith('.npy'):
            continue
        t_val = np.load(os.path.join(ROBERT_STATS, individual))
        t_vals.append(t_val)
    t_vals = np.array(t_vals)
    t_vals_mean = t_vals.mean(0)
    return t_vals_mean


if __name__ == "__main__":
    store_dir = PLOTS_DIR
    store_dir.mkdir(parents=True, exist_ok=True)

    t_vals_model = []
    for ckpt_name in MODEL_CKPTS:
        t_vals_dicts, p_vals_dicts, layer_positions = localizers(ckpt_name, ret_merged=True)
        t_vals_model.append(t_vals_dicts['robert'][0].flatten())

    t_vals_model = np.array(t_vals_model).mean(axis=0)
    t_vals_human = load_robert_tvals()

    from validate.rois.nsd import get_region_voxels

    high_level = get_region_voxels([
        'high-dorsal', 
        'high-ventral', 
        'high-lateral',
    ])
    t_vals_human = t_vals_human[high_level].flatten()

    # t_vals_human = t_vals_human/(t_vals_human**2).mean()**0.5
    # t_vals_model = t_vals_model/(t_vals_model**2).mean()**0.5

    # make distribution plots
    plt.figure(figsize=(5, 2.7))
    # plt.hist(t_vals_human, bins=100, alpha=0.5, label='Human', density=True, color=HUMAN_C)
    plt.hist(t_vals_model, bins=100, alpha=1, label='Model', density=True, color=MODEL_C)

    # despine
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    plt.xlabel('t-value')
    plt.ylabel('Density')
    plt.ylim(0, 0.05)
    plt.title('Distribution of model t-values')
    plt.savefig(store_dir / "robert_tval_distribution.svg", bbox_inches='tight')
    plt.close()
    print(f"Saved robert t-value distribution plot to {store_dir / 'robert_tval_distribution.svg'}")