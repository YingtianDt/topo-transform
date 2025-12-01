from config import PLOTS_DIR

import torch

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from .common import *
from .get_localizer_decode import localizer_decode
from .get_localizer_decode_ceiling import localizer_decode_ceiling
    

def plot_localizer_decode(model_ckpts, store_dir=PLOTS_DIR):

    rois = [
        'face',
        'place',
        'body',
        'v6',
        'psts',
    ]
    num_splits = 1

    all_scores = []
    for model_ckpt in model_ckpts:
        scores = localizer_decode(model_ckpt, rois, num_splits=num_splits, fwhm_mm=2.0, resolution_mm=1.0)
        all_scores.append(scores)

    ceilings = localizer_decode_ceiling(rois, folds=10)

    model_scores = []
    human_ceilings = []
    for r, roi in enumerate(rois):
        mean_score = np.mean([scores[:, r, r].mean() for scores in all_scores])
        mean_ceiling = ceilings[rois.index(roi)].mean()

        model_scores.append(mean_score)
        human_ceilings.append(mean_ceiling)

    if store_dir is None:
        return model_scores, human_ceilings
        
    plt.figure(figsize=(4, 3))
    x = np.arange(len(rois))
    width = 0.35
    plt.bar(x - width/2, [s.mean() for s in model_scores], width, label='Model', capsize=5, color=MODEL_C)
    plt.bar(x + width/2, human_ceilings, width, label='Human', capsize=5, color=HUMAN_C, alpha=0.7)

    # scatter individual model scores
    for m, scores in enumerate(all_scores):
        ind_scores = [scores[:, r, r].mean() for r in range(len(rois))]
        plt.scatter([x - width/2], ind_scores, color='black', alpha=0.3)

    plt.xticks(x, rois)
    plt.ylabel('Decoding Score')
    plt.title('Localizer Decoding Scores by ROI')
    plt.legend()
    plt.tight_layout()
    plt.savefig(store_dir / 'localizer_decoding_scores_comparison.svg')
    plt.close()

    return model_scores, human_ceilings

if __name__ == '__main__':
    model_scores, human_scores = plot_localizer_decode(MODEL_CKPTS, store_dir=PLOTS_DIR)
    tdann_scores, human_scores = plot_localizer_decode(TDANN_CKPTS, store_dir=None)
    unoptimized_scores, human_scores = plot_localizer_decode(UNOPTIMIZED_CKPTS, store_dir=None)

    plt.bar(['Model', 'TDANN', 'Unoptimized'], [np.mean(model_scores), np.mean(tdann_scores), np.mean(unoptimized_scores)], color=[MODEL_C, MODEL_C, MODEL_C])
    plt.ylabel('Mean Decoding Score')
    plt.title('Localizer Decoding Score Comparison')

    # add ceiling as horizontal zone
    mean_human_score = np.mean(human_scores)
    std_human_score = np.std(human_scores)
    plt.fill_between([-0.5, 2.5], 
                     mean_human_score - std_human_score, 
                     mean_human_score + std_human_score, 
                     color=HUMAN_C, alpha=0.3, label='Human Ceiling ±1 std')
    plt.axhline(mean_human_score, color=HUMAN_C, linestyle='--', linewidth=2, alpha=0.7)

    plt.savefig(PLOTS_DIR / 'localizer_decoding_score_comparison.svg', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved localizer decoding score comparison plot to {PLOTS_DIR / 'localizer_decoding_score_comparison.svg'}")