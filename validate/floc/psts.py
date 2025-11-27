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


BIOLOGICAL_MOTION = '/mnt/scratch/ytang/datasets/biological-motion'


def biomotion_category_dataset(data_dir=BIOLOGICAL_MOTION, transform=None, frames_per_video=24, video_fps=12):
    """Create a category dataset for the Biological Motion dataset (Vanrie 2004)."""
    file_infos = defaultdict(list)
    for category in os.listdir(data_dir):
        if category == "normal_static":  # skip neutral category
            continue
        category_dir = os.path.join(data_dir, category)
        if os.path.isdir(category_dir):
            for fname in os.listdir(category_dir):
                if fname.endswith(('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv')):
                    file_infos[category].append((os.path.join(category_dir, fname), category))

    return file_infos

def localize_psts(model, transform, 
                    batch_size=32, device='cuda', downsampler=None,
                    video_fps=12, frames_per_video=24, ret_pvals=False):

    categories = ["normal_dynamic", "scrambled_dynamic", "scrambled_static"]

    # Group files by category
    datasets = biomotion_category_dataset(transform=transform, frames_per_video=frames_per_video, video_fps=video_fps)
    datasets = {cat: datasets[cat] for cat in categories}
    n_categories = len(categories)
    t_vals_dict, p_vals_dict = t_test(
        model, transform, 
        datasets=datasets, contrasts=[(1, -1, 0), (0, 1, -1)],
        batch_size=batch_size, device=device, downsampler=downsampler,
        video_fps=video_fps, frames_per_video=frames_per_video
    )

    t_vals_ret = {
        "pSTS": t_vals_dict["normal_dynamic_vs_scrambled_dynamic"],
        "MT": t_vals_dict["scrambled_dynamic_vs_scrambled_static"],
    }
    p_vals_ret = {
        "pSTS": p_vals_dict["normal_dynamic_vs_scrambled_dynamic"],
        "MT": p_vals_dict["scrambled_dynamic_vs_scrambled_static"],
    }

    if ret_pvals:
        return t_vals_ret, p_vals_ret
    return t_vals_ret
    