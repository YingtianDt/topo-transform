import numpy as np
import skimage.measure
import shapely.geometry
import matplotlib.pyplot as plt
import matplotlib.patches
from typing import Tuple, List, Optional
from dataclasses import dataclass
import torch
import os

from spacetorch.utils.spatial_utils import concave_hull


def gaussian_2d(
    positions: np.ndarray, center: Tuple[float, float], sigma: float
) -> np.ndarray:
    """Compute 2D Gaussian weights for positions relative to center."""
    sigma_sq = sigma**2
    return (
        1.0 / (2.0 * np.pi * sigma_sq)
        * np.exp(
            -(
                (positions[:, 0] - center[0]) ** 2.0 / (2.0 * sigma_sq)
                + (positions[:, 1] - center[1]) ** 2.0 / (2.0 * sigma_sq)
            )
        )
    )


def labels_to_unit_indices(
    labels: np.ndarray, positions: np.ndarray, extent: np.ndarray
) -> List[np.ndarray]:
    """Convert labeled image regions to unit indices."""
    clusters = []
    n_anchors = labels.shape[0]
    grid_to_units = {}
    
    for unit_idx, pos in enumerate(positions):
        grid_x = int((pos[0] - positions[:, 0].min()) / extent[0] * (n_anchors - 1))
        grid_y = int((pos[1] - positions[:, 1].min()) / extent[1] * (n_anchors - 1))
        grid_y = n_anchors - 1 - grid_y
        
        grid_x = np.clip(grid_x, 0, n_anchors - 1)
        grid_y = np.clip(grid_y, 0, n_anchors - 1)
        
        label = labels[grid_y, grid_x]
        if label > 0:
            if label not in grid_to_units:
                grid_to_units[label] = []
            grid_to_units[label].append(unit_idx)
    
    for label, unit_list in grid_to_units.items():
        clusters.append(np.array(unit_list))
    
    return clusters


@dataclass
class Contrast:
    """Simple contrast definition."""
    name: str
    on_categories: List[str]
    color: str = 'blue'


class Patch:
    """Represents a category-selective patch."""
    
    def __init__(
        self,
        positions: np.ndarray,
        unit_indices: np.ndarray,
        selectivities: np.ndarray,
        contrast: Contrast,
        hull_alpha: float = 0.1,
    ):
        self.positions = positions
        self.unit_indices = unit_indices
        self.contrast = contrast
        self.selectivities = selectivities
        
        self._points = [shapely.geometry.Point(p) for p in self.positions]
        self.concave_hull = concave_hull(self._points, alpha=hull_alpha)
    
    @property
    def center(self) -> np.ndarray:
        return np.array(self.concave_hull.centroid.coords)[0]
    
    @property
    def area(self) -> float:
        return self.concave_hull.area
    
    @property
    def hull_vertices(self) -> np.ndarray:
        return np.array(list(zip(*self.concave_hull.exterior.coords.xy)))
    
    def __repr__(self) -> str:
        return (
            f"Patch of {self.contrast.name} with {len(self.unit_indices)} units, "
            f"centered at {self.center}, total area: {self.area:.1f}"
        )
    
    def to_mpl_poly(self, alpha: float = 0.6, lw: float = 2, hollow: bool = False):
        edgecolor = self.contrast.color if hollow else "white"
        fill = False if hollow else True
        return matplotlib.patches.Polygon(
            self.hull_vertices,
            facecolor=self.contrast.color,
            alpha=alpha,
            edgecolor=edgecolor,
            lw=lw,
            fill=fill,
        )


def find_patches_smoothing(
    positions: np.ndarray,
    selectivities: np.ndarray,
    contrast: Contrast,
    threshold: float = 2.0,
    minimum_size: float = 100,
    maximum_size: float = 4500,
    min_count: int = 10,
    sigma: float = 2.4,
    n_anchors: int = 100,
    hull_alpha: float = 0.1,
    verbose: bool = False,
) -> List[Patch]:
    """Find category-selective patches using Gaussian smoothing."""
    if isinstance(positions, torch.Tensor):
        positions = positions.cpu().numpy()
    if isinstance(selectivities, torch.Tensor):
        selectivities = selectivities.cpu().numpy()
    
    selectivities = selectivities.flatten()
    
    if verbose:
        print(f"Finding patches for {contrast.name}")
    
    # Create smoothing grid
    anchors = np.linspace(np.min(positions), np.max(positions), n_anchors)
    cx, cy = np.meshgrid(anchors, anchors)
    
    # Smooth selectivity values onto grid
    smoothed = np.zeros((n_anchors, n_anchors))
    for row in range(n_anchors):
        for col in range(n_anchors):
            center = (cx[row, col], cy[row, col])
            dist_from_center = gaussian_2d(positions, center, sigma=sigma)
            weighted = np.average(selectivities, weights=dist_from_center)
            smoothed[-row, col] = weighted
    
    # Threshold and label connected regions
    smoothed[smoothed < threshold] = 0
    labels = skimage.measure.label(smoothed > 0)
    
    # Convert labeled regions to unit indices
    clusters = labels_to_unit_indices(labels, positions, np.ptp(positions, axis=0))
    
    # Create patches and filter by size/count
    patches = []
    for cluster in clusters:
        patch = Patch(
            positions=positions[cluster],
            unit_indices=cluster,
            selectivities=selectivities[cluster],
            contrast=contrast,
            hull_alpha=hull_alpha,
        )
        
        if (
            patch.area >= minimum_size
            and patch.area <= maximum_size
            and len(patch.unit_indices) >= min_count
        ):
            patches.append(patch)
    
    if verbose:
        print(f"  Found {len(patches)} patches")
    
    return patches


def find_patches_for_categories(
    t_vals_dict: dict,
    layer_positions: List[np.ndarray],
    threshold: float = 2.0,
    minimum_size: float = 100,
    maximum_size: float = 4500,
    min_count: int = 10,
    sigma: float = 2.4,
    n_anchors: int = 100,
    hull_alpha: float = 0.1,
    verbose: bool = False,
) -> dict:
    """Find patches for all categories and layers."""
    results = {}
    
    for cat_name, t_vals_list in t_vals_dict.items():
        if not isinstance(t_vals_list, list):
            t_vals_list = [t_vals_list]
        
        category_patches = []
        contrast = Contrast(name=cat_name, on_categories=[cat_name])
        
        for layer_idx, t_vals in enumerate(t_vals_list):
            if verbose:
                print(f"\nProcessing {cat_name}, Layer {layer_idx}")
            
            patches = find_patches_smoothing(
                positions=layer_positions[layer_idx],
                selectivities=t_vals,
                contrast=contrast,
                threshold=threshold,
                minimum_size=minimum_size,
                maximum_size=maximum_size,
                min_count=min_count,
                sigma=sigma,
                n_anchors=n_anchors,
                hull_alpha=hull_alpha,
                verbose=verbose,
            )
            category_patches.append(patches)
        
        results[cat_name] = category_patches
    
    return results


def visualize_all_patches(
    patch_results: dict,
    layer_positions: List[np.ndarray],
    store_dir: str,
    figsize_per_panel: int = 6,
    prefix: str = '',
    suffix: str = '',
    show_stats: bool = True,
    color_map: Optional[dict] = None,
):
    """Visualize patches from find_patches_for_categories results."""
    os.makedirs(store_dir, exist_ok=True)
    
    categories = list(patch_results.keys())
    n_layers = len(layer_positions)
    
    if color_map is None:
        colors = plt.cm.tab10(np.linspace(0, 1, len(categories)))
        color_map = {cat: colors[i] for i, cat in enumerate(categories)}
    
    # Individual plots for each category and layer
    for cat_name, patches_by_layer in patch_results.items():
        for layer_idx, patches in enumerate(patches_by_layer):
            if show_stats:
                print(f"\n{cat_name} - Layer {layer_idx}: {len(patches)} patches")
            
            _plot_single(patches, layer_positions[layer_idx], cat_name, layer_idx,
                        store_dir, color_map.get(cat_name, 'blue'), 
                        (figsize_per_panel, figsize_per_panel), prefix, suffix)
    
    # Multi-layer comparison per category
    for cat_name, patches_by_layer in patch_results.items():
        _plot_category_layers(patches_by_layer, layer_positions, cat_name,
                             store_dir, color_map.get(cat_name, 'blue'),
                             figsize_per_panel, prefix, suffix)
    
    # Summary grid
    _plot_summary(patch_results, layer_positions, store_dir,
                 color_map, figsize_per_panel // 2, prefix, suffix)
    
    # Statistics
    _plot_stats(patch_results, store_dir, prefix, suffix)
    
    print(f"\nAll visualizations saved to {store_dir}")


def _plot_single(patches, all_positions, cat_name, layer_idx, store_dir, 
                color, figsize, prefix, suffix):
    """Plot single category-layer combination."""
    if isinstance(all_positions, torch.Tensor):
        all_positions = all_positions.cpu().numpy()
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.scatter(all_positions[:, 0], all_positions[:, 1],
              c='lightgray', s=1, alpha=0.3)
    
    for i, patch in enumerate(patches):
        ax.scatter(patch.positions[:, 0], patch.positions[:, 1], s=5, alpha=0.6, c=[color])
        poly = patch.to_mpl_poly(alpha=0.2, hollow=False)
        poly.set_facecolor(color)
        ax.add_patch(poly)
        ax.text(patch.center[0], patch.center[1], f"{i+1}", fontsize=10, 
               fontweight='bold', ha='center', va='center',
               bbox=dict(boxstyle='circle', facecolor='white', edgecolor=color, linewidth=2))
    
    ax.set_title(f'{cat_name} - Layer {layer_idx} ({len(patches)} patches)', 
                fontsize=12, fontweight='bold')
    ax.set_aspect('equal', 'box')
    ax.axis('equal')
    
    plt.tight_layout()
    plt.savefig(f'{store_dir}/{prefix}{cat_name}_layer{layer_idx}{suffix}.png', 
                dpi=300, bbox_inches='tight')
    plt.close()


def _plot_category_layers(patches_by_layer, layer_positions, cat_name, store_dir,
                          color, figsize_per_panel, prefix, suffix):
    """Plot all layers for a category."""
    n_layers = len(patches_by_layer)
    fig, axes = plt.subplots(1, n_layers, 
                            figsize=(n_layers * figsize_per_panel, figsize_per_panel))
    if n_layers == 1:
        axes = [axes]
    
    for layer_idx, (patches, ax) in enumerate(zip(patches_by_layer, axes)):
        all_positions = layer_positions[layer_idx]
        if isinstance(all_positions, torch.Tensor):
            all_positions = all_positions.cpu().numpy()
        
        ax.scatter(all_positions[:, 0], all_positions[:, 1], c='lightgray', s=1, alpha=0.3)
        
        for i, patch in enumerate(patches):
            ax.scatter(patch.positions[:, 0], patch.positions[:, 1], s=3, alpha=0.6, c=[color])
            poly = patch.to_mpl_poly(alpha=0.2, hollow=False)
            poly.set_facecolor(color)
            ax.add_patch(poly)
            ax.text(patch.center[0], patch.center[1], f"{i+1}", fontsize=8,
                   fontweight='bold', ha='center', va='center',
                   bbox=dict(boxstyle='circle', facecolor='white', edgecolor=color, linewidth=1.5))
        
        ax.set_title(f'Layer {layer_idx}\n({len(patches)} patches)', fontsize=10)
        ax.set_aspect('equal', 'box')
        ax.axis('equal')
    
    plt.suptitle(f'{cat_name} - All Layers', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{store_dir}/{prefix}{cat_name}_all_layers{suffix}.png', 
                dpi=300, bbox_inches='tight')
    plt.close()


def _plot_summary(patch_results, layer_positions, store_dir, color_map, 
                 figsize_per_panel, prefix, suffix):
    """Create summary grid."""
    categories = list(patch_results.keys())
    n_cats, n_layers = len(categories), len(layer_positions)
    
    fig, axes = plt.subplots(n_cats, n_layers,
                            figsize=(n_layers * figsize_per_panel, n_cats * figsize_per_panel))
    
    if n_cats == 1 and n_layers == 1:
        axes = np.array([[axes]])
    elif n_cats == 1:
        axes = axes.reshape(1, -1)
    elif n_layers == 1:
        axes = axes.reshape(-1, 1)
    
    for cat_idx, cat_name in enumerate(categories):
        for layer_idx in range(n_layers):
            ax = axes[cat_idx, layer_idx]
            all_positions = layer_positions[layer_idx]
            if isinstance(all_positions, torch.Tensor):
                all_positions = all_positions.cpu().numpy()
            
            ax.scatter(all_positions[:, 0], all_positions[:, 1], c='lightgray', s=0.5, alpha=0.2)
            
            patches = patch_results[cat_name][layer_idx]
            color = color_map.get(cat_name, 'blue')
            
            for patch in patches:
                ax.scatter(patch.positions[:, 0], patch.positions[:, 1], s=2, alpha=0.6, c=[color])
                poly = patch.to_mpl_poly(alpha=0.2, hollow=False)
                poly.set_facecolor(color)
                ax.add_patch(poly)
            
            if cat_idx == 0:
                ax.set_title(f'Layer {layer_idx}', fontsize=9)
            if layer_idx == 0:
                ax.set_ylabel(cat_name, fontsize=9, fontweight='bold')
            
            ax.set_aspect('equal', 'box')
            ax.axis('equal')
            ax.set_xticks([])
            ax.set_yticks([])
    
    plt.suptitle('Patch Summary - All Categories and Layers', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{store_dir}/{prefix}summary_grid{suffix}.png', dpi=300, bbox_inches='tight')
    plt.close()


def _plot_stats(patch_results, store_dir, prefix, suffix):
    """Create statistical summary plots."""
    categories = list(patch_results.keys())
    n_layers = len(patch_results[categories[0]])
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Prepare data
    patch_counts = {cat: [] for cat in categories}
    total_units = {cat: [] for cat in categories}
    mean_areas = {cat: [] for cat in categories}
    mean_sels = {cat: [] for cat in categories}
    
    for cat_name, patches_by_layer in patch_results.items():
        for patches in patches_by_layer:
            patch_counts[cat_name].append(len(patches))
            total_units[cat_name].append(sum(len(p.unit_indices) for p in patches))
            if patches:
                mean_areas[cat_name].append(np.mean([p.area for p in patches]))
                mean_sels[cat_name].append(np.mean([np.mean(p.selectivities) for p in patches]))
            else:
                mean_areas[cat_name].append(0)
                mean_sels[cat_name].append(0)
    
    # Plot 1: Patch count
    ax = axes[0, 0]
    x = np.arange(n_layers)
    width = 0.8 / len(categories)
    for i, cat in enumerate(categories):
        ax.bar(x + i * width, patch_counts[cat], width, label=cat)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Number of Patches')
    ax.set_title('Patch Count by Layer')
    ax.set_xticks(x + width * (len(categories) - 1) / 2)
    ax.set_xticklabels([f'L{i}' for i in range(n_layers)])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Plot 2: Total units
    ax = axes[0, 1]
    for i, cat in enumerate(categories):
        ax.bar(x + i * width, total_units[cat], width, label=cat)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Total Units in Patches')
    ax.set_title('Total Selective Units by Layer')
    ax.set_xticks(x + width * (len(categories) - 1) / 2)
    ax.set_xticklabels([f'L{i}' for i in range(n_layers)])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Plot 3: Mean area
    ax = axes[1, 0]
    for cat in categories:
        ax.plot(range(n_layers), mean_areas[cat], marker='o', label=cat, linewidth=2)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Mean Patch Area')
    ax.set_title('Mean Patch Area by Layer')
    ax.set_xticks(range(n_layers))
    ax.set_xticklabels([f'L{i}' for i in range(n_layers)])
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Plot 4: Mean selectivity
    ax = axes[1, 1]
    for cat in categories:
        ax.plot(range(n_layers), mean_sels[cat], marker='o', label=cat, linewidth=2)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Mean Selectivity')
    ax.set_title('Mean Selectivity by Layer')
    ax.set_xticks(range(n_layers))
    ax.set_xticklabels([f'L{i}' for i in range(n_layers)])
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{store_dir}/{prefix}statistics{suffix}.png', dpi=300, bbox_inches='tight')
    plt.close()