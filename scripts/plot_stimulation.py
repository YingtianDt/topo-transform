import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import scipy.optimize
import pandas as pd
import os

from config import CACHE_DIR, PLOTS_DIR
from scripts.get_localizers import localizers
from .microstimulation import test_stimulation
from .common import *


def logistic(x, a, b):
    return 1 / (1 + np.exp(-(a + b * x)))

def plot_response_curve(response, label_signal_levels, color='blue'):
    # group responses by signal level
    signal_levels = np.unique(label_signal_levels)
    signal_levels = np.sort(signal_levels)
    mean_responses = []
    for level in signal_levels:
        inds = np.where(label_signal_levels == level)[0]
        mean_responses.append(np.mean(response[inds]))
    mean_responses = np.array(mean_responses)

    # scatter and plot a logistic curve fit
    plt.scatter(signal_levels, mean_responses, alpha=0.3, label='Data', color=color)

    # Fit logistic curve
    popt, _ = scipy.optimize.curve_fit(logistic, signal_levels, mean_responses, p0=[0, 1], maxfev=10000)
    x_fit = np.linspace(signal_levels.min(), signal_levels.max(), 100)
    y_fit = logistic(x_fit, *popt)
    plt.plot(x_fit, y_fit, color=color, label='Logistic Fit', linestyle='--')
    midpoint = logistic(0.0, *popt)

    return midpoint


def plot_stimulation_results(ckpt_name, dataset_name, save_dir=PLOTS_DIR):
    """
    Generate plots to visualize the effects of microstimulation on model behavior.
    
    Parameters:
    -----------
    ckpt_name : str
        Name of the model checkpoint used for stimulation testing.
    dataset_name : str
        Name of the dataset used for testing.
    save_dir : str, optional
        Directory to save plots. If None, uses PLOTS_DIR from config
    """

    results = test_stimulation(ckpt_name, dataset_name)
    
    pre_stim = np.array(results['pre_stimulation'])  # (N_samples, )
    post_stim = np.array(results['post_stimulation'])  # list of (N_samples, ) arrays
    stimulation_locations = results['stimulation_locations']  # list of locations
    # selecitivities = results['selecitivities']  # list of selectivities
    label_signal_levels = results['label_signal_levels']
    sampled_indices = results['sampled_indices']
    n_locations = len(stimulation_locations)

    # Plot pre vs post stimulation predictions for each location
    plt.figure(figsize=(6, 4))

    mid_post = []
    for i in range(len(post_stim)):
        mid_p = plot_response_curve(post_stim[i], label_signal_levels, color='blue')
        mid_post.append(mid_p)

    mid_pre = plot_response_curve(pre_stim, label_signal_levels, color='green')
    path = os.path.join(save_dir, f'stimulation_response_curve.svg')
    plt.savefig(path)
    plt.close()
    print(f"Saved stimulation response curve plot to {path}")

    mid_post = np.array(mid_post)
    mid_shifts = mid_post - mid_pre  # (n_locations, )

    t_vals_dicts, p_vals_dicts, layer_positions = localizers(ckpt_name, ['afraz'], ret_merged=True)
    t_vals = t_vals_dicts['face_vs_nonface'][0].flatten()
    t_vals = t_vals[sampled_indices]

    plt.scatter(t_vals, mid_shifts, alpha=0.7)
    plt.xlabel('Selectivity (t-values for face)')
    plt.ylabel('Midpoint Shift (Post - Pre)')
    plt.title('Stimulation Effect vs Selectivity')
    path = os.path.join(save_dir, f'stimulation_selectivity_vs_shift.svg')
    plt.savefig(path)
    plt.close()
    print(f"Saved selectivity vs shift plot to {path}")


if __name__ == "__main__":
    plot_stimulation_results(
        ckpt_name=MODEL_CKPT,
        dataset_name="afraz2006"
    )
    

