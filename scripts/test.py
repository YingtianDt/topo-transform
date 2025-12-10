import os
import numpy as np
import cortex
from config import PLOTS_DIR
from .common import *
from .get_localizers import get_localizer_human
from neuroparc.atlas import Atlas

# ------------------------------------------------
# Define ROIs and load vertex masks
# ------------------------------------------------
rois = ['face', 'body', 'place', 'mt', 'v6', 'psts']
masks = get_localizer_human(rois)

for i, mask in enumerate(masks):
    surface = Atlas("fsaverage", mask.astype(float))
    masks[i] = surface.label_surface("fsaverage7")  # boolean mask

# Number of vertices for fsaverage5
n_vertices = 163842 * 2  # left + right
vertex_colors = np.ones((n_vertices, 3)) * np.nan

# Assign colors to ROIs
for mask, roi in zip(masks, rois):
    _, color = all_roi_colors[roi]  # RGB tuple
    vertex_colors[mask>0.5] = color

# ------------------------------------------------
# Split left/right hemispheres
# ------------------------------------------------
data = {
    'roi_colors': {
        'left': vertex_colors[:163842, :],
        'right': vertex_colors[163842:, :]
    }
}

import pickle

pickle.dump(data, open("temp_roi_colors_fsaverage7.pkl", "wb"))
exit()
data = pickle.load(open("temp_roi_colors_fsaverage7.pkl", "rb"))

# ------------------------------------------------
# Create pycortex VertexRGB objects
# ------------------------------------------------
subject = 'fsaverage'
if subject not in cortex.db.subjects:
    cortex.db.add_subject_from_fsaverage(subject)

cortex_data = {}
for name, hemi_dict in data.items():
    cortex_data[name] = {}
    for hemi in ['left', 'right']:
        rgb = hemi_dict[hemi]
        cortex_data[name][hemi] = cortex.VertexRGB(
            red=rgb[:, 0],
            green=rgb[:, 1],
            blue=rgb[:, 2],
            subject=subject,
            xfmname='identity',
            hemi=hemi
        )

# ------------------------------------------------
# Launch web visualization and save snapshot
# ------------------------------------------------
for name, hemi_dict in cortex_data.items():
    view = cortex.web.show_hemi(hemi_dict['left'])
    os.makedirs(PLOTS_DIR, exist_ok=True)
    cortex.webgl.screenshot(view, os.path.join(PLOTS_DIR, f"{name}_cortex.png"))
