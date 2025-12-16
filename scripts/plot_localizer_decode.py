from config import PLOTS_DIR

import torch

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import seaborn as sns

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
    human_ceiling_stds = []
    for r, roi in enumerate(rois):
        mean_score = np.array([scores[:, r, r].mean() for scores in all_scores])
        mean_ceiling = ceilings[rois.index(roi)].mean(-1)

        model_scores.append(mean_score)
        human_ceilings.append(mean_ceiling)
        human_ceiling_stds.append(ceilings[rois.index(roi)].std(-1))

    model_scores = np.array(model_scores)
    human_ceilings = np.array(human_ceilings)
    human_ceiling_stds = np.array(human_ceiling_stds)

    if store_dir is None:
        return model_scores, human_ceilings
        
    plt.figure(figsize=(2.5, 2.6))
    x = np.arange(len(rois))
    width = 0.7
    for r, (roi, model_score, human_ceiling, human_ceiling_std) in enumerate(zip(rois, model_scores, human_ceilings, human_ceiling_stds)):
        model_score = model_score.mean()
        human_ceiling = human_ceiling.mean()
        _, roi_color = all_roi_colors[roi]
        plt.bar(x[r], model_score, width, color=roi_color, label='Model' if r == 0 else "", alpha=1)
        plt.plot([x[r] - width/2, x[r] + width/2], 
                 [human_ceiling, human_ceiling], 
                 color=HUMAN_C, linestyle='-', linewidth=2, alpha=1, label='Human Ceiling' if r == 0 else "")

    # scatter individual model scores
    for m, scores in enumerate(all_scores):
        ind_scores = [scores[:, r, r].mean() for r in range(len(rois))]
        colors = [all_roi_colors[roi][1] for roi in rois]
        plt.scatter([x], ind_scores, color='k', alpha=1, s=5)

    sns.despine()

    roi_display_names = ['Face', 'Place', 'Body', 'V6', 'pSTS']
    plt.xticks(x, roi_display_names, rotation=45, ha='right')
    plt.ylim(0, 0.75)
    plt.xlim(-.8, len(rois)-0.2)
    plt.ylabel('Prediction score (R)')
    plt.tight_layout()
    plt.savefig(store_dir / 'localizer_decoding_scores_comparison.svg')
    plt.close()

    return model_scores, human_ceilings

if __name__ == '__main__':
    model_scores, human_scores = plot_localizer_decode(MODEL_CKPTS, store_dir=PLOTS_DIR)
    tdann_scores, human_scores = plot_localizer_decode(TDANN_CKPTS, store_dir=None)
    unoptimized_scores, human_scores = plot_localizer_decode(UNOPTIMIZED_CKPTS, store_dir=None)
    swapopt_scores, human_scores = plot_localizer_decode(SWAPOPT_CKPTS, store_dir=None)
    onelayer_scores, human_scores = plot_localizer_decode(ONELAYER_CKPTS, store_dir=None)

    human_scores = human_scores.mean(axis=0)
    model_scores = model_scores.mean(axis=0)
    tdann_scores = tdann_scores.mean(axis=0)
    unoptimized_scores = unoptimized_scores.mean(axis=0)
    swapopt_scores = swapopt_scores.mean(axis=0)
    onelayer_scores = onelayer_scores.mean(axis=0)

    all_scores = [
        model_scores,
        tdann_scores,
        unoptimized_scores,
        swapopt_scores,
        onelayer_scores,
    ]

    plt.figure(figsize=(2.2, 2.0))

    width = 0.71
    colors = [MODEL_C, DEFAULT_C, DEFAULT_C, DEFAULT_C, DEFAULT_C]
    labels = ['Ours', 'TDANN', 'SwapOpt', 'VJEPA', 'OneLayer']
    means = [np.mean(scores) for scores in all_scores]

    bars = plt.barh(labels, means, color=colors, height=width, alpha=1)

    # Add white text labels inside bars
    for bar, label in zip(bars, labels):
        plt.text(0.03, bar.get_y() + bar.get_height() / 2, 
                label, ha='left', va='center', color='white', fontsize=10)

    for i, (scores, color) in enumerate(zip(all_scores, colors)):
        x = np.ones(len(scores)) * i
        plt.scatter(scores, x, color='k', alpha=1, s=5)

    # plt.axvline(human_scores.mean(), color=HUMAN_C, linestyle='-', linewidth=2, label='Human Ceiling')
    ci = 1.96 * human_scores.std()
    plt.fill_betweenx([-1, len(labels)], human_scores.mean() - ci, human_scores.mean() + ci, color=HUMAN_C, alpha=0.3, edgecolor='none')

    # test each model against human ceiling
    for i, scores in enumerate(all_scores):
        from scipy import stats
        t_stat, p_value = stats.ttest_ind(scores, human_scores)
        print(f"{labels[i]} vs Human Ceiling: t={t_stat:.3f}, p={p_value:.5f}")
        print(f"Mean {labels[i]} score: {np.mean(scores):.3f}, Human ceiling: {human_scores.mean():.3f}")

    # test ours' improvement percentage and significance against others
    our_scores = all_scores[0]
    for i, scores in enumerate(all_scores[1:]):
        improvement = (our_scores - scores) / np.abs(scores) * 100
        t_stat, p_value = stats.ttest_rel(our_scores, scores)
        print(f"Ours vs {labels[i+1]}: improvement={improvement.mean():.2f}%, t={t_stat:.3f}, p={p_value:.5f}")

    plt.xlabel('Mean prediction score (R)')

    # add ceiling as horizontal zone
    plt.yticks([])  # Remove y-axis labels since they're now inside the bars
    plt.ylim(-.8, len(labels)-0.2)
    plt.xlim(0, 0.6)
    sns.despine()
    plt.savefig(PLOTS_DIR / 'localizer_decoding_score_comparison.svg', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved localizer decoding score comparison plot to {PLOTS_DIR / 'localizer_decoding_score_comparison.svg'}")