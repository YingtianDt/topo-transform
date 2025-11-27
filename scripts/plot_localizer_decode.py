from config import PLOTS_DIR

import torch

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from .common import *
from .get_localizer_decode import localizer_decode
from .get_localizer_decode_ceiling import localizer_decode_ceiling
    

num_splits = 1

rois = [
    'face',
    'place',
    'body',
    'v6',
    'psts',
    'mt',
]

scores = localizer_decode(MODEL_CKPT, rois, num_splits=num_splits, fwhm_mm=2.0, resolution_mm=1.0)
ceilings = localizer_decode_ceiling(rois, folds=10)

model_scores = []
human_ceilings = []
for roi in rois:
    mean_score = scores[roi].mean()
    mean_ceiling = ceilings[rois.index(roi)].mean()

    model_scores.append(mean_score)
    human_ceilings.append(mean_ceiling)

plt.figure(figsize=(4, 3))
x = np.arange(len(rois))
width = 0.35
plt.bar(x - width/2, model_scores, width, label='Model', capsize=5, color=MODEL_C)
plt.bar(x + width/2, human_ceilings, width, label='Human', capsize=5, color=HUMAN_C, alpha=0.7)
plt.xticks(x, rois)
plt.ylabel('Decoding Score')
plt.title('Localizer Decoding Scores by ROI')
plt.legend()
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'localizer_decoding_scores_comparison.svg')
plt.close()
