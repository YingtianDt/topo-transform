import numpy as np
import skimage.measure
import shapely.geometry
import matplotlib.pyplot as plt
import matplotlib.patches
from typing import Tuple, List, Optional
from dataclasses import dataclass
import torch
import os
from tqdm import tqdm
from typing import Tuple
from scipy.spatial.distance import cdist
from spacetorch.utils.spatial_utils import concave_hull


class Patch:
    """Represents a category-selective patch."""
    
    def __init__(
        self,
        positions: np.ndarray,
        unit_indices: np.ndarray,
        selectivities: np.ndarray,
        hull_alpha: float = 0.1,
    ):
        self.positions = positions
        self.unit_indices = unit_indices
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
    
    def to_mpl_poly(self, color="red", alpha: float = 0.6, lw: float = 2, hollow: bool = False):
        edgecolor = color if hollow else "white"
        fill = not hollow
        return matplotlib.patches.Polygon(
            self.hull_vertices,
            facecolor=color,
            alpha=alpha,
            edgecolor=edgecolor,
            lw=lw,
            fill=fill,
        )

_BACKGROUND_CLUSTER = 0

def labels_to_unit_indices(labels, positions, extent: Tuple[float, float] = None):
    # positions: [N, 2] array of real physical positions
    # labels: length N vector of cluster labels

    unique_labels = np.unique(labels)
    index_sets = []
    
    for lab in unique_labels:
        if lab == _BACKGROUND_CLUSTER:
            continue

        # units that belong to this label
        unit_ids = np.where(labels == lab)[0]
        if len(unit_ids) == 0:
            continue

        # these units’ physical positions
        matching_pos = positions[unit_ids]  # (k, 2)

        # compute minimal distance of every unit to this cluster
        distances = cdist(positions, matching_pos)  # (N, k)
        min_distance_for_each_unit = distances.min(axis=1)

        # dilation threshold, allow edge units to be included
        min_cutoff = 2**0.5 + 1e-3
        units_in_cluster = np.where(min_distance_for_each_unit <= min_cutoff)[0]

        if len(units_in_cluster) < 3:
            continue

        index_sets.append(units_in_cluster)

    return index_sets

def _is_degenerate(points: np.ndarray) -> bool:
    return np.any(np.ptp(points, axis=0) == 0)

def find_patches(
    positions: np.ndarray,
    selectivities: np.ndarray,
    threshold: float = 2.0,
    minimum_size: float = 100,
    maximum_size: float = 4500,
    min_count: int = 10,
    hull_alpha: float = 0.1,
    verbose: bool = False,
) -> List[Patch]:
    """Find category-selective patches using Gaussian smoothing."""
    if isinstance(positions, torch.Tensor):
        positions = positions.cpu().numpy()
    if isinstance(selectivities, torch.Tensor):
        selectivities = selectivities.cpu().numpy()

    selectivities = selectivities.copy()
    positions = positions.copy()

    assert selectivities.shape[0] == 1
    selectivities = selectivities[0]  # Remove channel dimension if present

    selectivities, positions = _to_regular_grid(positions, selectivities)
    positions = positions.reshape(-1, 2)

    if verbose:
        print(f"Finding patches for {contrast.name}")
    
    # Threshold and label connected regions in the grid
    selectivities[(selectivities < threshold)|(np.isnan(selectivities))] = 0
    labels = skimage.measure.label(selectivities > 0)
    
    # Convert labeled regions to unit indices
    clusters = labels_to_unit_indices(labels.flatten(), positions, np.ptp(positions, axis=0))
    
    # Create patches and filter by size/count
    patches = []
    for cluster in clusters:
        point_positions = positions[cluster]

        if _is_degenerate(point_positions):
            continue

        patch = Patch(
            positions=point_positions,
            unit_indices=cluster,
            selectivities=selectivities.flatten()[cluster],
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
    
    # Progress bar for categories
    pbar_cats = tqdm(t_vals_dict.items(), desc="Processing categories", disable=not verbose)
    
    for cat_name, t_vals_list in pbar_cats:
        if not isinstance(t_vals_list, list):
            t_vals_list = [t_vals_list]
        
        category_patches = []
        pbar_cats.set_postfix_str(f"Category: {cat_name}")
        
        # Progress bar for layers within each category
        for layer_idx in tqdm(range(len(t_vals_list)), 
                              desc=f"  Layers for {cat_name}", 
                              leave=False,
                              disable=not verbose):
            t_vals = t_vals_list[layer_idx]
            
            if verbose:
                print(f"\nProcessing {cat_name}, Layer {layer_idx}")
            
            patches = find_patches(
                positions=layer_positions[layer_idx],
                selectivities=t_vals,
                threshold=threshold,
                minimum_size=minimum_size,
                maximum_size=maximum_size,
                min_count=min_count,
                hull_alpha=hull_alpha,
                verbose=verbose,
            )
            category_patches.append(patches)
        
        results[cat_name] = category_patches
    
    return results

def _to_regular_grid(
    positions: np.ndarray,
    selectivities: np.ndarray,
    tolerance: float = 1e-6
) -> Tuple[np.ndarray, np.ndarray]:
    
    # Flatten selectivities if needed
    selectivities = selectivities.flatten()
    
    assert positions.shape[0] == selectivities.shape[0], \
        f"Positions ({positions.shape[0]}) and selectivities ({selectivities.shape[0]}) must have same length"
    
    # Find unique x and y coordinates
    unique_x = np.unique(np.round(positions[:, 0] / tolerance) * tolerance)
    unique_y = np.unique(np.round(positions[:, 1] / tolerance) * tolerance)
    
    height = len(unique_y)
    width = len(unique_x)
    
    # Verify we have a complete grid
    assert height * width == positions.shape[0], \
        f"Positions don't form a complete grid: {height}x{width}={height*width} != {positions.shape[0]}"
    
    # Create coordinate to index mapping
    x_to_col = {x: i for i, x in enumerate(unique_x)}
    y_to_row = {y: i for i, y in enumerate(unique_y)}
    
    # Initialize output grids
    grid_selectivities = np.full((height, width), np.nan)
    grid_positions = np.full((height, width, 2), np.nan)
    
    # Place each point in the grid
    for idx, (pos, sel) in enumerate(zip(positions, selectivities)):
        # Find nearest grid coordinates
        x_rounded = unique_x[np.argmin(np.abs(unique_x - pos[0]))]
        y_rounded = unique_y[np.argmin(np.abs(unique_y - pos[1]))]
        
        row = y_to_row[y_rounded]
        col = x_to_col[x_rounded]
        
        grid_selectivities[row, col] = sel
        grid_positions[row, col] = pos
    
    # Verify no NaN values remain (complete grid)
    assert not np.any(np.isnan(grid_selectivities)), "Grid has missing values"
    
    return grid_selectivities, grid_positions


def visualize_all_patches(
    patch_results: dict,
    layer_positions: List[np.ndarray],
    store_dir: str,
    figsize_per_panel: int = 6,
    prefix: str = '',
    suffix: str = '',
    show_stats: bool = True,
    color_map: Optional[dict] = None,
    plot_individual: bool = False,
):
    """Visualize patches from find_patches_for_categories results."""
    os.makedirs(store_dir, exist_ok=True)
    
    categories = list(patch_results.keys())
    n_layers = len(layer_positions)
    
    if color_map is None:
        colors = plt.cm.tab10(np.linspace(0, 1, len(categories)))
        color_map = {cat: colors[i] for i, cat in enumerate(categories)}
    
    # Individual plots for each category and layer
    if plot_individual:
        total_plots = sum(len(patches_by_layer) for patches_by_layer in patch_results.values())
        pbar = tqdm(total=total_plots, desc="Creating individual plots")
        
        for cat_name, patches_by_layer in patch_results.items():
            for layer_idx, patches in enumerate(patches_by_layer):
                if show_stats:
                    print(f"\n{cat_name} - Layer {layer_idx}: {len(patches)} patches")
                
                _plot_single(patches, layer_positions[layer_idx], cat_name, layer_idx,
                            store_dir, color_map.get(cat_name, 'blue'), 
                            (figsize_per_panel, figsize_per_panel), prefix, suffix)
                pbar.update(1)
        pbar.close()
    
    # Multi-layer comparison per category
    print("\nCreating multi-layer comparisons...")
    _plot_category_layers(patch_results, layer_positions,
                          store_dir, color_map, figsize_per_panel, prefix, suffix)
    
    # Summary grid
    print("\nCreating summary grid...")
    _plot_summary(patch_results, layer_positions, store_dir,
                 color_map, figsize_per_panel // 2, prefix, suffix)
    
    # Statistics
    print("\nCreating statistics plots...")
    _plot_stats(patch_results, store_dir, prefix, suffix)
    
    print(f"\nAll visualizations saved to {store_dir}")


def _plot_single(patches, all_positions, cat_name, layer_idx, store_dir, 
                color, figsize, prefix, suffix, scatter=False):
    """Plot single category-layer combination."""
    if isinstance(all_positions, torch.Tensor):
        all_positions = all_positions.cpu().numpy()
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    if scatter:
        ax.scatter(all_positions[:, 0], all_positions[:, 1],
                c='lightgray', s=1, alpha=0.3)
    
    for i, patch in enumerate(patches):
        if scatter:
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


def _plot_category_layers(patch_results, layer_positions, store_dir,
                          color_map, figsize_per_panel, prefix, suffix, scatter=False):
    """Plot all layers for a category."""
    n_layers = len(layer_positions)
    fig, axes = plt.subplots(1, n_layers, 
                            figsize=(n_layers * figsize_per_panel, figsize_per_panel))
    if n_layers == 1:
        axes = [axes]
    
    for cat_name, patches_by_layer in tqdm(patch_results.items(), desc="Multi-layer plots"):
        color = color_map.get(cat_name, 'blue')
        for layer_idx, (patches, ax) in enumerate(zip(patches_by_layer, axes)):
            all_positions = layer_positions[layer_idx]
            if isinstance(all_positions, torch.Tensor):
                all_positions = all_positions.cpu().numpy()
            
            if scatter:
                ax.scatter(all_positions[:, 0], all_positions[:, 1], c='lightgray', s=1, alpha=0.3)
            
            for i, patch in enumerate(patches):
                if scatter:
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

    # legend
    handles = [matplotlib.patches.Patch(color=color_map.get(cat, 'blue'), label=cat) for cat in patch_results.keys()]
    fig.legend(handles=handles, loc='upper right', fontsize=10)
    
    plt.suptitle(f'{cat_name} - All Layers', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{store_dir}/{prefix}{cat_name}_all_layers{suffix}.png', 
                dpi=300, bbox_inches='tight')
    plt.close()


def _plot_summary(patch_results, layer_positions, store_dir, color_map, 
                 figsize_per_panel, prefix, suffix, scatter=False):
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
            
            if scatter:
                ax.scatter(all_positions[:, 0], all_positions[:, 1], c='lightgray', s=0.5, alpha=0.2)
            
            patches = patch_results[cat_name][layer_idx]
            color = color_map.get(cat_name, 'blue')
            
            for patch in patches:
                if scatter:
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


def visualize_patches(
    t_vals_dict: dict,
    layer_positions: List[np.ndarray],
    viz_dir: str,
    prefix: str = '',
    suffix: str = '',
    threshold: float = 2.0,
    minimum_size: float = 100,
    maximum_size: float = 4500,
    min_count: int = 10,
    hull_alpha: float = 0.1,
    verbose: bool = False,
    figsize_per_panel: int = 6,
    show_stats: bool = True,
    color_map: Optional[dict] = None,
):
    """
    Find and visualize patches from t-values dictionary.
    
    Args:
        t_vals_dict: Dictionary mapping category names to t-values (or list of t-values per layer)
        layer_positions: List of position arrays for each layer
        viz_dir: Directory to save visualizations
        prefix: Prefix for output filenames
        suffix: Suffix for output filenames
        threshold: Selectivity threshold for patch detection
        minimum_size: Minimum patch area
        maximum_size: Maximum patch area
        min_count: Minimum number of units in a patch
        sigma: Gaussian smoothing sigma
        n_anchors: Number of grid points for smoothing
        hull_alpha: Alpha parameter for concave hull
        verbose: Print detailed information
        figsize_per_panel: Figure size per panel in inches
        show_stats: Print patch statistics
        color_map: Optional dictionary mapping category names to colors
    
    Returns:
        patch_results: Dictionary mapping category names to lists of patches per layer
    """
    # Find patches for all categories
    patch_results = find_patches_for_categories(
        t_vals_dict=t_vals_dict,
        layer_positions=layer_positions,
        threshold=threshold,
        minimum_size=minimum_size,
        maximum_size=maximum_size,
        min_count=min_count,
        hull_alpha=hull_alpha,
        verbose=verbose,
    )
    
    # Visualize all patches
    visualize_all_patches(
        patch_results=patch_results,
        layer_positions=layer_positions,
        store_dir=viz_dir,
        figsize_per_panel=figsize_per_panel,
        prefix=prefix,
        suffix=suffix,
        show_stats=show_stats,
        color_map=color_map,
    )
    
    return patch_results