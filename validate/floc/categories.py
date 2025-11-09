import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

import os
import numpy as np
from scipy import stats

from utils import cached
from .utils import t_test


VPNL = '/mnt/scratch/ytang/datasets/fLoc_stimuli'
KANWISHER = '/mnt/scratch/ytang/datasets/lahner/stimulus_set/stimuli/localizer'


def VPNL_category_dataset(data_dir=VPNL, transform=None, frames_per_video=24, video_fps=12):
    """Create a category dataset for the VPNL dataset."""
    datasets = defaultdict(list)
    for fname in os.listdir(data_dir):
        if fname.endswith(('.jpg', '.png', '.jpeg')):
            category = fname.split('-')[0]
            if category in {'adult', 'child'}:
                category = 'face'
            elif category in {'word', 'number'}:
                category = 'character'
            elif category in {'corridor', 'house'}:
                category = 'place'
            elif category in {'car', 'instrument'}:
                category = 'object'
            elif category in {'body', 'limb'}:
                category = 'body'
            else:
                assert category == 'scrambled'
                continue # skip scrambled images
            datasets[category].append((os.path.join(data_dir, fname), category))
    return datasets


def KANWISHER_category_dataset(data_dir=KANWISHER, transform=None, frames_per_video=24, video_fps=12):
    """Create a category dataset for the Kanwisher dataset."""
    datasets = defaultdict(list)
    for category in os.listdir(data_dir):
        if category == "Scrambled15G": # skip scrambled images
            continue
        category_dir = os.path.join(data_dir, category)
        if os.path.isdir(category_dir):
            for fname in os.listdir(category_dir):
                if fname.endswith(('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv')):
                    datasets[category].append((os.path.join(category_dir, fname), category))
    return datasets


def functional_localization_one_vs_rest(model, transform, datasets, 
                                        batch_size=32, device='cuda', downsampler=None,
                                        video_fps=12, frames_per_video=24):

    # Group files by category
    n_categories = len(datasets)
    t_vals_dict = t_test(
        model, transform, datasets=datasets,
        contrasts=[[1 if i == j else -1 for i in range(n_categories)] for j in range(n_categories)],
        batch_size=batch_size, device=device, downsampler=downsampler,
        video_fps=video_fps, frames_per_video=frames_per_video
    )[0]

    ret = {cat.split("_vs_")[0]: t_vals for cat, t_vals in t_vals_dict.items()}

    return ret


def localize_categories(model, transform, dataset_name, frames_per_video=24, video_fps=12, 
                        batch_size=32, device='cuda', downsampler=None):
    if dataset_name == "vpnl":
        datasets = VPNL_category_dataset(transform=transform, frames_per_video=frames_per_video, video_fps=video_fps)
    elif dataset_name == "kanwisher":
        datasets = KANWISHER_category_dataset(transform=transform, frames_per_video=frames_per_video, video_fps=video_fps)
    else:
        raise ValueError(f"Unknown dataset_name: {dataset_name}")

    t_vals_dict = functional_localization_one_vs_rest(
        model, transform=transform, datasets=datasets,
        batch_size=batch_size, device=device, downsampler=downsampler,
        video_fps=video_fps, frames_per_video=frames_per_video
    )
    return t_vals_dict