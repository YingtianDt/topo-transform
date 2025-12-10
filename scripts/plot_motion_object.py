from config import PLOTS_DIR

import torch

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from .common import *
from config import ROBERT_STATS
from .get_localizers import localizers, get_localizer_human, get_localizer_model


if __name__ == "__main__":
    store_dir = PLOTS_DIR
    store_dir.mkdir(parents=True, exist_ok=True)

    rois = [
        'face',
        'body',
        'place',
        # 'mt',
        'v6',
        'psts',
    ]

    colors = [
        (0.75, 0.00, 0.00),
        (0.00, 0.45, 0.00),
        (0.80, 0.35, 0.00),
        # (0.95, 0.20, 0.70),
        (0.0, 0.85, 0.95),
        (0.85, 0.85, 0.0),
    ]

    t_vals_dicts, p_vals_dicts, layer_positions = localizers(MODEL_CKPT, ret_merged=True)
    t_val_motion = t_vals_dicts['robert_motion'][0].flatten()
    t_val_object = t_vals_dicts['robert_static'][0].flatten()
    masks = get_localizer_model(rois, MODEL_CKPT)
    masks = [m[0].flatten() for m in masks]

    # plot a coordinate motion vs object, where the dots are colored by their roi

    plt.scatter(
        t_val_object,
        t_val_motion,
        color='gray',
        alpha=0.3,
        s=2,
    )

    for mask, roi, color in zip(masks, rois, colors):
        plt.scatter(
            t_val_object[mask],
            t_val_motion[mask],
            label=roi,
            color=color,
            alpha=1,
            s=2,
        )

    plt.xlabel('Object t-value')
    plt.ylabel('Motion t-value')
    plt.title('Model Localizer: Motion vs Object')
    plt.legend(markerscale=5)
    plt.savefig(store_dir / 'model_localizer_motion_vs_object.png')
    plt.close()

    print('Saved plot to', store_dir / 'model_localizer_motion_vs_object.png')