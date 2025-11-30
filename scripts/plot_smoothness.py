import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import seaborn as sns
from tqdm import tqdm
from scipy import stats
import pandas as pd

from config import CACHE_DIR, PLOTS_DIR
from .get_smoothness import smoothness
from .common import *

def plot_smoothness_comparison(model_paths, category, fwhm_mm=2.0, resolution_mm=1.0, save_dir=None):
    """
    Generate bar plots comparing model and human smoothness across categories,
    and a correlation plot between model and human smoothness values.
    
    Parameters:
    -----------
    model_paths : str or list of str
        Path(s) to the model checkpoint(s). If multiple paths provided, will compute
        mean and confidence intervals across models.
    category : str
        Category name for analysis
    fwhm_mm : float
        FWHM for smoothing in mm
    resolution_mm : float
        Resolution in mm
    save_dir : str, optional
        Directory to save plots. If None, uses PLOTS_DIR from config
    """
    
    # Convert single model path to list
    if isinstance(model_paths, str):
        model_paths = [model_paths]
    
    # Get smoothness data for all models
    print(f"Computing smoothness for {len(model_paths)} model(s), category: {category}")
    
    all_model_results = []
    for i, model_path in enumerate(model_paths):
        print(f"  Model {i+1}/{len(model_paths)}: {model_path}")
        ret = smoothness(model_path, category, fwhm_mm=fwhm_mm, resolution_mm=resolution_mm)
        all_model_results.append(ret)
    
    if save_dir is None:
        save_dir = PLOTS_DIR
    
    # Prepare data for plotting - aggregate across models
    categories = list(all_model_results[0].keys())
    
    # Collect model smoothness values across all models
    model_smoothness_all = []  # List of lists: [model][category]
    human_smoothness = []  # Human data (same across models)
    
    for cat in categories:
        model_vals = [result[cat]['model_smoothness'] for result in all_model_results]
        model_smoothness_all.append(model_vals)
        # Human smoothness is the same for all models, just take from first
        human_smoothness.append(all_model_results[0][cat]['human_smoothness'])
    
    # Compute statistics across models
    model_smoothness_mean = [np.mean(vals) for vals in model_smoothness_all]
    model_smoothness_std = [np.std(vals) for vals in model_smoothness_all]
    model_smoothness_sem = [stats.sem(vals) if len(vals) > 1 else 0 for vals in model_smoothness_all]
    
    # 95% confidence interval
    confidence_level = 0.95
    model_smoothness_ci = []
    for vals in model_smoothness_all:
        if len(vals) > 1:
            ci = stats.t.interval(confidence_level, len(vals)-1, 
                                 loc=np.mean(vals), 
                                 scale=stats.sem(vals))
            model_smoothness_ci.append((ci[1] - np.mean(vals)))  # Half-width of CI
        else:
            model_smoothness_ci.append(0)
    
    # Create figure with two subplots - improved aesthetics (vertical layout with horizontal bars)
    fig, axes = plt.subplots(2, 1, figsize=(5, 12))
    
    # Set clean style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # ==========================================
    # Subplot 1: Grouped Bar Plot with Category Groups
    # ==========================================
    ax1 = axes[0]
    
    # Group categories by base type (bodies, faces, etc.) and variant (static/dynamic)
    grouped_data = {}
    print("\nCategory grouping:")
    for i, cat in enumerate(categories):
        model_vals = model_smoothness_all[i]
        model_mean = model_smoothness_mean[i]
        model_ci = model_smoothness_ci[i]
        human_val = human_smoothness[i]
        
        # Extract base category and variant
        cat_lower = cat.lower()
        
        # Determine if static or dynamic first
        if 'dynamic' in cat_lower or 'moving' in cat_lower or 'motion' in cat_lower:
            variant = 'dynamic'
        else:
            variant = 'static'
        
        # Extract base category (remove dynamic/static keywords)
        cat_clean = cat_lower.replace('dynamic', '').replace('static', '').replace('moving', '').strip('_- ')
        
        if 'body' in cat_clean or 'bodies' in cat_clean:
            base = 'Bodies'
        elif 'face' in cat_clean:
            base = 'Faces'
        elif 'object' in cat_clean:
            base = 'Objects'
        elif 'place' in cat_clean or 'scene' in cat_clean:
            base = 'Places'
        elif 'character' in cat_clean:
            base = 'Characters'
        else:
            base = 'Other'
        
        print(f"  {cat:20s} -> {base:15s} ({variant})")
        
        if base not in grouped_data:
            grouped_data[base] = {'static': None, 'dynamic': None}
        
        # Store data directly (not in nested dict) - should be one per variant
        grouped_data[base][variant] = {
            'name': cat,
            'model_mean': model_mean,
            'model_ci': model_ci,
            'model_vals': model_vals,
            'human': human_val
        }
    
    # Prepare plotting data
    plot_positions = []
    plot_model_means = []
    plot_model_cis = []
    plot_model_vals_list = []
    plot_human_vals = []
    plot_labels = []
    group_boundaries = []
    
    current_pos = 0
    width = 0.55
    group_gap = .5
    within_group_gap = 0.75
    
    for group_name in sorted(grouped_data.keys()):
        group_start = current_pos
        
        # Plot static first, then dynamic
        for variant in ['static', 'dynamic']:
            data = grouped_data[group_name][variant]
            if data is not None:  # Check if this variant exists
                plot_positions.append(current_pos)
                plot_model_means.append(data['model_mean'])
                plot_model_cis.append(data['model_ci'])
                plot_model_vals_list.append(data['model_vals'])
                plot_human_vals.append(data['human'])
                plot_labels.append(f"{data['name']}")
                current_pos += within_group_gap
        
        group_boundaries.append((group_start, current_pos - within_group_gap, group_name))
        current_pos += group_gap
    
    # Create horizontal bars for model data (mean) - no edge color
    bars = ax1.barh(plot_positions, plot_model_means, width, 
                    color=MODEL_C, alpha=0.7, edgecolor='none',
                    label='Model (mean)')
    
    # Add error bars for confidence intervals
    if len(model_paths) > 1:
        ax1.errorbar(plot_model_means, plot_positions, xerr=plot_model_cis,
                    fmt='none', ecolor='#5A6B89', elinewidth=2, capsize=5, capthick=2,
                    alpha=0.6, label='95% CI')
    
    # Overlay scatter points for individual model values
    for pos, model_vals in zip(plot_positions, plot_model_vals_list):
        # Add small jitter to y-position for visibility when multiple models
        if len(model_vals) > 1:
            jitter = np.random.uniform(-0.08, 0.08, len(model_vals))
        else:
            jitter = [0]
        ax1.scatter(model_vals, [pos + j for j in jitter], 
                   color='#3E4F6B', s=60, alpha=0.9, edgecolor='white', 
                   linewidth=1, zorder=10)
    
    # Add vertical lines for human data
    for pos, human_val in zip(plot_positions, plot_human_vals):
        ax1.plot([human_val, human_val], [pos - width/2, pos + width/2],
                color=HUMAN_C, linestyle='--', linewidth=2.5, alpha=0.8)
    
    # Add a single legend entry for human data
    human_line = Line2D([0], [0], color=HUMAN_C, linestyle='-', linewidth=2.5, 
                        label='Human', alpha=0.8)
    
    # Create custom legend entry for scatter points
    scatter_point = Line2D([0], [0], marker='o', color='w', markerfacecolor='#3E4F6B',
                          markersize=8, markeredgecolor='white', markeredgewidth=1,
                          label='Individual models', linestyle='None')
    
    # Customize the bar plot
    ax1.set_ylabel('Category', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Smoothness', fontsize=13, fontweight='bold')
    ax1.set_title(f'Model vs Human Smoothness by Category (n={len(model_paths)} models)', 
                 fontsize=14, fontweight='bold', pad=15)
    ax1.set_yticks(plot_positions)
    ax1.set_yticklabels(plot_labels, fontsize=10)
    
    # Add group labels
    for start, end, name in group_boundaries:
        mid = (start + end) / 2
        x_offset = ax1.get_xlim()[0] - 0.08 * (ax1.get_xlim()[1] - ax1.get_xlim()[0])
        ax1.text(x_offset, mid, name, ha='right', va='center', fontsize=11, fontweight='bold',
                color='#333333')
    
    # Build legend
    handles = [bars, scatter_point]
    # if len(model_paths) > 1:
    #     handles.append(ax1.errorbar([], [], yerr=[], fmt='none', ecolor='#5A6B89', 
    #                                 elinewidth=2, capsize=5, capthick=2, alpha=0.6, 
    #                                 label='95% CI')[0])
    handles.append(human_line)
    ax1.legend(handles=handles, loc='upper left', fontsize=10, frameon=True, 
              fancybox=False, shadow=False, framealpha=0.9)
    
    # Remove top and right spines
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_linewidth(1.5)
    ax1.spines['bottom'].set_linewidth(1.5)
    
    # # Improve grid
    # ax1.grid(axis='x', alpha=0.25, linestyle='--', linewidth=1)
    # ax1.set_axisbelow(True)
    
    # Adjust y-axis limits for better spacing
    ax1.set_ylim(-0.6, max(plot_positions) + 0.6)
    
    # Invert y-axis so first category is at top
    ax1.invert_yaxis()
    
    # ==========================================
    # Subplot 2: Model vs Human Scores
    # ==========================================
    ax2 = axes[1]
    
    # Create regression plot using seaborn (using mean values)
    sns.regplot(x=model_smoothness_mean, y=human_smoothness, ax=ax2,
                scatter_kws={'s': 100, 'alpha': 0.7, 'edgecolor': 'none'},
                line_kws={'color': '#D32F2F', 'linewidth': 2.5, 'alpha': 0.8},
                color='#1976D2')
    
    # Calculate correlation statistics
    corr_coef, p_value = stats.pearsonr(model_smoothness_mean, human_smoothness)
    
    # Customize plot
    ax2.set_xlabel('Model Smoothness (mean)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Human Smoothness', fontsize=13, fontweight='bold')
    ax2.set_title(f'Model vs Human Smoothness\n(r={corr_coef:.3f}, p={p_value:.4f})', 
                 fontsize=14, fontweight='bold', pad=15)
    
    # Remove top and right spines
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_linewidth(1.5)
    ax2.spines['bottom'].set_linewidth(1.5)
    
    # Improve grid
    ax2.grid(True, alpha=0.25, linestyle='--', linewidth=1)
    ax2.set_axisbelow(True)
    
    # Adjust layout with better spacing
    plt.tight_layout(pad=2.0)
    
    # Save figure
    save_path = f"{save_dir}/smoothness_comparison_{category}.svg"
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Plot saved to: {save_path}")

    # Print statistics
    print("\n" + "="*50)
    print("SMOOTHNESS STATISTICS")
    print("="*50)
    print(f"Number of models: {len(model_paths)}")
    print(f"Pearson Correlation (mean model vs human): r = {corr_coef:.4f}, p = {p_value:.6f}")
    print(f"Mean Model Smoothness: {np.mean(model_smoothness_mean):.4f} ± {np.std(model_smoothness_mean):.4f}")
    print(f"Mean Human Smoothness: {np.mean(human_smoothness):.4f} ± {np.std(human_smoothness):.4f}")
    
    # Category-wise differences
    print("\nCategory-wise Differences (Model Mean - Human):")
    for i, cat in enumerate(categories):
        model_mean = model_smoothness_mean[i]
        model_ci = model_smoothness_ci[i]
        human_val = human_smoothness[i]
        diff = model_mean - human_val
        print(f"  {cat:15s}: {diff:+.4f} (Model: {model_mean:.4f}±{model_ci:.4f}, Human: {human_val:.4f})")
    
    # If multiple models, show variance across models
    if len(model_paths) > 1:
        print("\nVariance across models:")
        for i, cat in enumerate(categories):
            model_std = model_smoothness_std[i]
            print(f"  {cat:15s}: SD = {model_std:.4f}")
    
    return all_model_results


if __name__ == '__main__':
    # Example usage
    
    # Single model
    model_paths = MODEL_CKPTS
    category = 'pitcher'
    ret = plot_smoothness_comparison(model_paths, category, fwhm_mm=2.0, resolution_mm=1.0)
    
    # Multiple models
    # model_paths = [MODEL_CKPT_1, MODEL_CKPT_2, MODEL_CKPT_3]
    # ret = plot_smoothness_comparison(model_paths, category, fwhm_mm=2.0, resolution_mm=1.0)