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


ROBERT = '/mnt/scratch/ytang/datasets/robert2023'

def Robert_category_dataset(data_dir=ROBERT, transform=None, frames_per_video=36, video_fps=12):
    """Create a category dataset for the Robert dataset."""
    file_infos = defaultdict(list)
    for fname in os.listdir(data_dir):
        if fname.endswith('.mat'): continue
        category = 'dynamic' if fname.endswith('.mp4') else 'static'
        file_infos[category].append((os.path.join(data_dir, fname), category))

    return file_infos

def localize_robert(model, transform, 
                    batch_size=32, device='cuda', downsampler=None,
                    video_fps=12, frames_per_video=36, ret_pvals=False):

    categories = ["dynamic", "static"]

    # Group files by category
    datasets = Robert_category_dataset(transform=transform, frames_per_video=frames_per_video, video_fps=video_fps)
    datasets = {cat: datasets[cat] for cat in categories}
    n_categories = len(categories)
    t_vals_dict, p_vals_dict = t_test(
        model, transform, 
        datasets=datasets, contrasts=[(1, -1)],
        batch_size=batch_size, device=device, downsampler=downsampler,
        video_fps=video_fps, frames_per_video=frames_per_video
    )

    t_vals_ret = {"robert": t_vals_dict["dynamic_vs_static"]}
    p_vals_ret = {"robert": p_vals_dict["dynamic_vs_static"]}

    if ret_pvals:
        return t_vals_ret, p_vals_ret
    return t_vals_ret

    
