import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
from tqdm import tqdm
from scipy import stats
import pandas as pd

from config import CACHE_DIR, PLOTS_DIR
from .get_smoothness import smoothness
from .common import *

def plot_smoothness_comparison(model_path, category, fwhm_mm=2.0, resolution_mm=1.0, save_dir=None):
    """
    Generate bar plots comparing model and human smoothness across categories,
    and a correlation plot between model and human smoothness values.
    
    Parameters:
    -----------
    model_path : str
        Path to the model checkpoint
    category : str
        Category name for analysis
    save_dir : str, optional
        Directory to save plots. If None, uses PLOTS_DIR from config
    """
    
    # Get smoothness data
    print(f"Computing smoothness for model: {model_path}, category: {category}")
    ret = smoothness(model_path, category, fwhm_mm=fwhm_mm, resolution_mm=resolution_mm)
    
    if save_dir is None:
        save_dir = PLOTS_DIR
    
    # Prepare data for plotting
    categories = list(ret.keys())
    model_smoothness = [ret[cat]['model_smoothness'] for cat in categories]
    human_smoothness = [ret[cat]['human_smoothness'] for cat in categories]
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # ==========================================
    # Subplot 1: Grouped Bar Plot
    # ==========================================
    ax1 = axes[0]
    
    x = np.arange(len(categories))
    width = 0.35
    
    # Create bars
    bars1 = ax1.bar(x - width/2, model_smoothness, width, label='Model', 
                    color='steelblue', alpha=0.8, edgecolor='black', linewidth=1)
    bars2 = ax1.bar(x + width/2, human_smoothness, width, label='Human', 
                    color='coral', alpha=0.8, edgecolor='black', linewidth=1)
    
    # Customize the bar plot
    ax1.set_xlabel('Category', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Smoothness', fontsize=12, fontweight='bold')
    ax1.set_title('Model vs Human Smoothness by Category', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, rotation=45, ha='right')
    ax1.legend(loc='upper left', fontsize=11)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_ylim(0.5)
    
    # Add value labels on bars
    def add_bar_labels(bars, ax):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9)
    
    add_bar_labels(bars1, ax1)
    add_bar_labels(bars2, ax1)
    
    # ==========================================
    # Subplot 2: Model vs Human Scores
    # ==========================================
    ax2 = axes[1]
    
    # Create regression plot using seaborn
    sns.regplot(x=model_smoothness, y=human_smoothness, ax=ax2,
                scatter_kws={'s': 80, 'alpha': 0.7, 'edgecolor': 'black'},
                line_kws={'color': 'red', 'linewidth': 2, 'alpha': 0.8},
                color='steelblue')
    
    # Calculate correlation statistics
    corr_coef, p_value = stats.pearsonr(model_smoothness, human_smoothness)
    
    # Customize plot
    ax2.set_xlabel('Model Smoothness', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Human Smoothness', fontsize=12, fontweight='bold')
    ax2.set_title(f'Model vs Human Smoothness\n(r={corr_coef:.3f}, p={p_value:.4f})', 
                 fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    save_path = f"{save_dir}/smoothness_comparison_{category}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {save_path}")

    # Print statistics
    print("\n" + "="*50)
    print("SMOOTHNESS STATISTICS")
    print("="*50)
    print(f"Pearson Correlation: r = {corr_coef:.4f}, p = {p_value:.6f}")
    print(f"Mean Model Smoothness: {np.mean(model_smoothness):.4f} ± {np.std(model_smoothness):.4f}")
    print(f"Mean Human Smoothness: {np.mean(human_smoothness):.4f} ± {np.std(human_smoothness):.4f}")
    
    # Category-wise differences
    print("\nCategory-wise Differences (Model - Human):")
    for cat, model_val, human_val in zip(categories, model_smoothness, human_smoothness):
        diff = model_val - human_val
        print(f"  {cat:15s}: {diff:+.4f} (Model: {model_val:.4f}, Human: {human_val:.4f})")
    
    return ret


if __name__ == '__main__':
    # Example usage
    model_path = MODEL_CKPT
    category = 'pitcher'
    
    # Generate main comparison plots
    ret = plot_smoothness_comparison(model_path, category, fwhm_mm=2.0, resolution_mm=1.0)