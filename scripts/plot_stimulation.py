import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import scipy.optimize
import pandas as pd
import os

from config import CACHE_DIR, PLOTS_DIR
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
    popt, _ = scipy.optimize.curve_fit(logistic, signal_levels, mean_responses, p0=[0, 1])
    x_fit = np.linspace(signal_levels.min(), signal_levels.max(), 100)
    y_fit = logistic(x_fit, *popt)
    plt.plot(x_fit, y_fit, color=color, label='Logistic Fit', linestyle='--')


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
    selecitivities = results['selecitivities']  # list of selectivities
    label_signal_levels = results['label_signal_levels']
    n_locations = len(stimulation_locations)

    # Plot pre vs post stimulation predictions for each location
    plt.figure(figsize=(6, 4))

    plot_response_curve(pre_stim, label_signal_levels, color='green')
    
    for i in range(n_locations):
        plot_response_curve(post_stim[i], label_signal_levels, color='blue')
        break

    path = os.path.join(save_dir, f'stimulation_response_curve.svg')
    plt.savefig(path)
    plt.close()

    print(f"Saved stimulation response curve plot to {path}")


if __name__ == "__main__":
    plot_stimulation_results(
        ckpt_name=MODEL_CKPT,
        dataset_name="afraz2006"
    )
    

