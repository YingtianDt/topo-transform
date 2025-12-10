import config 
from config import PLOTS_DIR

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from .get_neural_alignment import neural_alignment
from .get_task_performance import task_performance

from .common import *

def plot_task_performance_comparison(ckpt_name):
    tasks = [
        "imagenet",
        "ssv2",
    ]
    for task in tasks:
        _ = task_performance(ckpt_name, task)

def plot_neural_alignment_comparison(ckpt_name, num_splits=1, figsize=(4, 4)):
    """
    Plot horizontal bar plot comparing pre- and post-transform neural alignment.
    
    Args:
        ckpt_name: Name of the checkpoint file
        num_splits: Number of cross-validation splits
        figsize: Figure size (width, height)
    """
    # Get the data
    scores_pre, scores_post, mask, ceiling = neural_alignment(ckpt_name, num_splits=num_splits)
    return

    ceiling = ceiling[:, mask]
    
    # Compute mean scores across voxels for each split
    mean_pre = scores_pre.mean(axis=1)  # [num_splits]
    mean_post = scores_post.mean(axis=1)  # [num_splits]
    mean_ceiling = ceiling.mean(axis=1)  # [num_splits]
    
    # Perform paired t-test
    if num_splits > 1:
        t_stat, p_value = stats.ttest_rel(mean_post, mean_pre)
        # Also compute effect size (Cohen's d for paired samples)
        diff = mean_post - mean_pre
        cohens_d = diff.mean() / diff.std(ddof=1)
    else:
        t_stat, p_value, cohens_d = None, None, None
    
    # Compute overall means and ceiling
    overall_mean_pre = scores_pre.mean()
    overall_mean_post = scores_post.mean()
    overall_ceiling = ceiling.mean()
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Define positions and colors
    positions = [0, 1]
    colors = ['#FA0022', '#959595']  # Gray for original, red for transformed
    labels = ['Transformed', 'Original']
    means = [overall_mean_post, overall_mean_pre]
    split_data = [mean_pre, mean_post]
    
    # Plot horizontal bars
    bars = ax.barh(positions, means, height=0.45, color=colors, alpha=1, edgecolor='none')
    # TODO: std
    
    # Add text labels inside bars
    for i, (bar, label, mean_val) in enumerate(zip(bars, labels, means)):
        # Position text in the middle of the bar
        ax.text(mean_val / 2, bar.get_y() + bar.get_height() / 2, 
                label, ha='center', va='center', 
                fontsize=12, color='white')
    
    # Plot individual split points
    for i, (pos, splits) in enumerate(zip(positions, split_data)):
        # Add jitter to y-position for visibility
        y_positions = np.random.normal(pos, 0.08, size=len(splits))
        ax.scatter(splits, y_positions, color='black', s=50, alpha=0.6, 
                   zorder=3, edgecolors='white', linewidth=2)
    
    # Add ceiling as a vertical shaded region
    mean_ceiling_err = mean_ceiling.std()
    ax.axvspan(overall_ceiling - mean_ceiling_err, overall_ceiling + mean_ceiling_err, 
               color='gray', alpha=0.3, label='Ceiling ±1 std')
    ax.axvline(overall_ceiling, color='gray', linestyle='--', linewidth=2, alpha=0.7)
    
    # Formatting
    ax.set_yticks(positions)
    ax.set_yticklabels([])  # Remove y-tick labels since text is embedded
    ax.set_xlabel('Neural Alignment Score (normalized R)', fontsize=12)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.7, 1.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # # Add legend for ceiling
    # ax.legend(loc='upper right', frameon=False, fontsize=10)
    
    # Add title with stats
    improvement = ((overall_mean_post - overall_mean_pre) / overall_mean_pre) * 100
    title = f'Neural Alignment Comparison ({num_splits} split{"s" if num_splits > 1 else ""})\n'
    title += f'Improvement: {improvement:+.1f}% | Ceiling: {overall_ceiling:.3f}'
    
    # Add significance info if we have multiple splits
    if num_splits > 1 and p_value is not None:
        if p_value < 0.001:
            sig_str = '***'
        elif p_value < 0.01:
            sig_str = '**'
        elif p_value < 0.05:
            sig_str = '*'
        else:
            sig_str = 'n.s.'
        title += f' | p={p_value:.4f} {sig_str}'
    
    ax.set_title(title, fontsize=13, pad=15)
    
    plt.tight_layout()
    
    # Prepare statistics dictionary
    result_stats = {
        'pre': overall_mean_pre,
        'post': overall_mean_post,
        'ceiling': overall_ceiling,
        'improvement_pct': improvement,
        't_stat': t_stat,
        'p_value': p_value,
        'cohens_d': cohens_d,
        'mean_pre_splits': mean_pre,
        'mean_post_splits': mean_post
    }
    
    return fig, ax, result_stats


if __name__ == "__main__":

    tasks = [
        "imagenet",
        "ssv2",
    ]

    for ckpt_name in MODEL_CKPTS:
        for task in tasks:
            _ = task_performance(ckpt_name, task)
        neural_alignment(ckpt_name, num_splits=1)

    exit()

    # Example usage
    ckpt_name = MODEL_CKPT
    num_splits = 1  # Adjust as needed

    plot_task_performance_comparison(ckpt_name)
    fig, ax, stats = plot_neural_alignment_comparison(ckpt_name, num_splits=num_splits)
    
    # Print statistics
    print("\n" + "="*50)
    print("NEURAL ALIGNMENT COMPARISON RESULTS")
    print("="*50)
    print(f"Original (mean ± std):     {stats['pre']:.4f} ± {stats['mean_pre_splits'].std():.4f}")
    print(f"Transformed (mean ± std):  {stats['post']:.4f} ± {stats['mean_post_splits'].std():.4f}")
    print(f"Ceiling:                   {stats['ceiling']:.4f}")
    print(f"Improvement:               {stats['improvement_pct']:+.2f}%")
    
    if stats['p_value'] is not None:
        print("\n" + "-"*50)
        print("PAIRED T-TEST RESULTS")
        print("-"*50)
        print(f"t-statistic:               {stats['t_stat']:.4f}")
        print(f"p-value:                   {stats['p_value']:.6f}")
        print(f"Cohen's d (effect size):   {stats['cohens_d']:.4f}")
        
        if stats['p_value'] < 0.001:
            sig_level = "highly significant (p < 0.001) ***"
        elif stats['p_value'] < 0.01:
            sig_level = "very significant (p < 0.01) **"
        elif stats['p_value'] < 0.05:
            sig_level = "significant (p < 0.05) *"
        else:
            sig_level = "not significant (p >= 0.05) n.s."
        
        print(f"Significance:              {sig_level}")
        
        # Interpret effect size
        if abs(stats['cohens_d']) < 0.2:
            effect_interp = "negligible"
        elif abs(stats['cohens_d']) < 0.5:
            effect_interp = "small"
        elif abs(stats['cohens_d']) < 0.8:
            effect_interp = "medium"
        else:
            effect_interp = "large"
        print(f"Effect size interpretation: {effect_interp}")
    else:
        print("\nNote: Need at least 2 splits for statistical testing")
    
    print("="*50 + "\n")
    
    # Save figure
    plt.savefig(PLOTS_DIR / f'plot_neural_alignment.svg', dpi=300, bbox_inches='tight')
    plt.show()
