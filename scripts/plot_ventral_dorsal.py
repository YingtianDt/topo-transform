from validate.floc.utils.cluster import visualize_patches
from .get_localizers import localizers, get_localizer_human, get_localizer_model

from .common import *
from config import PLOTS_DIR

import os
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    ventral_rois = ['face', 'body', 'place']
    dorsal_rois = ['v6', 'psts']
    dorsal_rois = ['mt']

    masks = get_localizer_model(
        rois=ventral_rois + dorsal_rois,
        ckpt_name=MODEL_CKPT,
    )
    masks = {roi: np.array(mask, dtype=bool) for roi, mask in zip(ventral_rois + dorsal_rois, masks)}

    mask_ventral = np.zeros_like(masks[ventral_rois[0]], dtype=bool)
    for roi in ventral_rois:
        mask_ventral = mask_ventral | masks[roi]

    mask_dorsal = np.zeros_like(masks[dorsal_rois[0]], dtype=bool)
    for roi in dorsal_rois:
        mask_dorsal = mask_dorsal | masks[roi]

    mask_ventral = mask_ventral[0,0]
    mask_dorsal = mask_dorsal[0,0]

    plt.imshow(mask_ventral)
    plt.savefig(os.path.join(PLOTS_DIR, 'ventral_mask.png'), dpi=300)
    plt.close()

    plt.imshow(mask_dorsal)
    plt.savefig(os.path.join(PLOTS_DIR, 'dorsal_mask.png'), dpi=300)
    plt.close()

    breakpoint()