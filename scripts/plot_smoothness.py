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

def plot_smoothness_comparison(model_paths, category, fwhm_mm=2.0, resolution_mm=1.0, save_dir=PLOTS_DIR):
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
    fig, ax1 = plt.subplots(1, 1, figsize=(5, 6))
    
    # Set clean style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # ==========================================
    # Subplot 1: Anaylsis
    # ==========================================
    
    diff = np.array(model_smoothness_mean) - np.array(human_smoothness)
    mae = np.mean(np.abs(diff))
    
    # Adjust layout with better spacing
    plt.tight_layout()
    
    # Save figure
    if save_dir is not None:
        save_path = f"{save_dir}/smoothness_comparison_{category}.svg"
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Plot saved to: {save_path}")

    # Print statistics
    print("\n" + "="*50)
    print("SMOOTHNESS STATISTICS")
    print("="*50)
    print(f"Number of models: {len(model_paths)}")
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

    plt.close()
    return mae


if __name__ == '__main__':
    category = 'pitcher'

    # model_mae = plot_smoothness_comparison(MODEL_CKPTS, category, fwhm_mm=2.0, resolution_mm=1.0)
    tdann_mae = plot_smoothness_comparison(TDANN_CKPTS, category, fwhm_mm=2.0, resolution_mm=1.0, save_dir=None)
    
    plt.bar(['Model', 'TDANN'], [model_mae, tdann_mae], color=[MODEL_C, MODEL_C])
    plt.ylabel('Mean Absolute Error (MAE)')
    plt.title(f'Smoothness MAE Comparison - {category}')
    plt.savefig(PLOTS_DIR / f'smoothness_mae_comparison_{category}.svg', dpi=300, bbox_inches='tight')
    plt.close()