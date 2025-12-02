"""
3D visualization of topological layer positions.

Visualizes neural network layer positions in 3D space similar to cortical tissue organization.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import torch
from pathlib import Path

from config import CACHE_DIR, PLOTS_DIR
from spacetorch.models.positions import LayerPositions

from .common import MODEL_CKPT
from validate import load_transformed_model


POSITION_DIR = CACHE_DIR / "positions"


def load_layer_positions(layer_config_dir):
    """Load layer positions for a given model configuration."""
    
    layer_positions = []
    layer_config_dir = Path(layer_config_dir)
    position_files = sorted(layer_config_dir.glob("*.pkl"))
    
    for file_path in position_files:
        layer_position = LayerPositions.load(file_path)
        layer_positions.append(layer_position)
    
    return layer_positions


def visualize_3d_layers(layer_positions, layer_indices=None, 
                        figsize=(20, 5), elevation=8, azimuth=-90,
                        show_neurons=True, neuron_size=3, alpha=0.8,
                        colormap='viridis', depth_spacing=10,
                        subsample_factor=0.01,
                        layer_names=None, save_path=None,
                        show_shadows=True, shadow_alpha=0.9):
    
    if layer_indices is None:
        layer_indices = range(len(layer_positions))
    
    n_layers = len(layer_indices)
    n_layers = 1
    
    # Create subplots
    fig = plt.figure(figsize=figsize)
    
    for plot_idx, layer_idx in enumerate(layer_indices[:n_layers]):
        ax = fig.add_subplot(1, n_layers, plot_idx + 1, projection='3d')
        
        layer_pos = layer_positions[layer_idx]
        coords = layer_pos.coordinates
        dims = layer_pos.dims
        assert dims == (1024*3, 14, 14)

        # assume single layer
        # color the bottom-left channel of the first layer
        mask = np.zeros(dims, dtype=bool)
        mask[:1024, :, :] = True
        mask = mask.flatten()

        # Subsample coordinates for visualization
        if subsample_factor < 1.0:
            num_neurons = coords.shape[0]
            subsample_size = max(1, int(num_neurons * subsample_factor))
            selected_indices = np.random.choice(num_neurons, subsample_size, replace=False)
            coords = coords[selected_indices]
            mask = mask[selected_indices]
        
        # Convert to numpy if tensor
        if torch.is_tensor(coords):
            coords = coords.cpu().numpy()
        
        # Add depth dimension (z-axis) with spacing between layers
        z_offset = plot_idx * depth_spacing
        z_coords = np.full(coords.shape[0], z_offset)
        
        # Get z limits for shadow projection
        z_min = 0  # Project shadows to the bottom plane
        
        if show_neurons:
            # Plot shadows first (on the bottom plane)
            if show_shadows:
                ax.scatter(coords[:, 0], coords[:, 1], 
                          z_coords - 0.06,
                          c='black', s=neuron_size * 1.6, alpha=shadow_alpha,
                          edgecolors='none')
            
            # Plot all neurons in grey with edge colors for 3D appearance
            color = np.array(['#808080']*coords.shape[0])
            for i in range(coords.shape[0]):
                if mask[i]:
                    color[i] = "#66CCFF"  # Highlighted color
            ax.scatter(coords[:, 0], coords[:, 1], z_coords,
                      color=color,  # Grey color
                      s=neuron_size, 
                      alpha=alpha,
                      edgecolors='white',  # White edges for 3D effect
                      linewidths=0.3,  # Thin edge line
                      depthshade=True)  # Enable depth shading
        
        # Set viewing angle: azimuth=0 makes x-axis parallel to screen
        ax.view_init(elev=elevation, azim=azimuth)
        
        # Remove all axes, labels, ticks, and grid
        ax.set_axis_off()
        
        # Set aspect ratio
        ax.set_box_aspect([1, 1, 0.3])
        
        # Set limits to zoom in closer (reduce margin around data)
        x_range = coords[:, 0].max() - coords[:, 0].min()
        y_range = coords[:, 1].max() - coords[:, 1].min()
        x_center = coords[:, 0].mean()
        y_center = coords[:, 1].mean()
        
        # Zoom in by using tighter limits (0.55 gives ~10% margin on each side)
        zoom_factor = 0.55
        ax.set_xlim(x_center - x_range * zoom_factor, x_center + x_range * zoom_factor)
        ax.set_ylim(y_center - y_range * zoom_factor, y_center + y_range * zoom_factor)
        ax.set_zlim(z_min - depth_spacing * 0.1, z_offset + depth_spacing * 0.1)
    
    plt.tight_layout(pad=0)
    
    if save_path:
        plt.savefig(save_path, dpi=600, bbox_inches='tight', pad_inches=0)
        print(f"Figure saved to: {save_path}")
    
    plt.close()

    return fig


if __name__ == "__main__":
    
    layer_position_dir = "/mnt/scratch/ytang/tdann/cache/positions/vjepa_14_18_22_single_neighbInf"

    # Create save directory if specified
    save_dir = PLOTS_DIR / "plot_single_sheet"
    save_dir.mkdir(exist_ok=True, parents=True)
    
    # Load layer positions
    print(f"Loading layer positions from directory: {layer_position_dir}")
    layer_positions = load_layer_positions(layer_position_dir)
    print(f"Loaded {len(layer_positions)} layers")
    
    visualize_3d_layers(
        layer_positions,
        layer_indices=None,
        save_path=save_dir / "single_sheet.png" if save_dir else None
    )