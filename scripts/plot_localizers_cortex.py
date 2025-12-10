import os
import numpy as np
from nilearn import datasets, surface
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from config import PLOTS_DIR
from .common import *
from .get_localizers import get_localizer_human

rois = [
    'face',
    'body',
    'place',
    'mt',
    'v6',
    'psts',
]


# ------------------------------------------------
# Load fsaverage (choose fsaverage5 or fsaverage)
# ------------------------------------------------
fsavg = datasets.fetch_surf_fsaverage('fsaverage5')

# Example: your RGBA colors (n_vertices x 4)
n_vertices = 20484

# Initialize with gray color (0.5, 0.5, 0.5, 1.0)
colors = np.ones((n_vertices, 4)) * [0.5, 0.5, 0.5, 1.0]

masks = get_localizer_human(rois)
for mask, roi in zip(masks, rois):
    _, color = all_roi_colors[roi]
    colors[mask] = color + (1.0,)

# ------------------------------------------------
# Use Poly3DCollection directly
# ------------------------------------------------
# Load both hemisphere meshes
mesh_left = surface.load_surf_mesh(fsavg['flat_left'])
mesh_right = surface.load_surf_mesh(fsavg['flat_right'])

coords_left, faces_left = mesh_left[0], mesh_left[1]
coords_right, faces_right = mesh_right[0], mesh_right[1]
coords_right += np.array([0, 300, 0])  # Shift right hemisphere for better visualization

# Get colors for each hemisphere
left_colors = colors[:10242]
right_colors = colors[10242:]

# Create the triangles for each hemisphere
triangles_left = coords_left[faces_left]
triangles_right = coords_right[faces_right]

# Use the color of the first vertex of each face (no averaging/interpolation)
face_colors_left = left_colors[faces_left[:, 0]]
face_colors_right = right_colors[faces_right[:, 0]]

# Create figure
fig = plt.figure(figsize=(24, 8))
ax = fig.add_subplot(111, projection='3d')

# Create the mesh for left hemisphere
mesh_plot_left = Poly3DCollection(triangles_left, facecolors=face_colors_left, edgecolors='none', linewidth=0, antialiased=False)
mesh_plot_left.set_facecolor(face_colors_left)
ax.add_collection3d(mesh_plot_left)

# # Create the mesh for right hemisphere
# mesh_plot_right = Poly3DCollection(triangles_right, facecolors=face_colors_right, edgecolors='none', linewidth=0, antialiased=False)
# mesh_plot_right.set_facecolor(face_colors_right)
# ax.add_collection3d(mesh_plot_right)

# # Set the axis limits to include both hemispheres
# all_coords = np.vstack([coords_left, coords_right])
# ax.set_xlim(all_coords[:, 0].min(), all_coords[:, 0].max())
# ax.set_ylim(all_coords[:, 1].min(), all_coords[:, 1].max())
# ax.set_zlim(all_coords[:, 2].min(), all_coords[:, 2].max())

# aspect ratio
ax.set_box_aspect([1, 1, 1])

# Set viewing angle for upside-down view (rotate 180 degrees)
ax.view_init(elev=90, azim=0)
ax.set_axis_off()

# Equal aspect ratio
ax.set_box_aspect([1, 1, 1])

plt.savefig(os.path.join(PLOTS_DIR, 'localizers_cortex.png'), dpi=400, bbox_inches='tight')
plt.close()

print("Saved localizers cortex visualization to", os.path.join(PLOTS_DIR, 'localizers_cortex.png'))