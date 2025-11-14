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

FLOW = '/mnt/scratch/ytang/datasets/flow_fields'


def Pitzalis_category_dataset(data_dir=FLOW, transform=None, frames_per_video=24, video_fps=12):
    """Create a category dataset for the Pitzalis dataset."""
    file_infos = defaultdict(list)
    for category in os.listdir(data_dir):
        category_dir = os.path.join(data_dir, category)
        if os.path.isdir(category_dir):
            for fname in os.listdir(category_dir):
                if fname.endswith(('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv')):
                    file_infos[category].append((os.path.join(category_dir, fname), category))

    return file_infos

def localize_v6(model, transform, 
                batch_size=32, device='cuda', downsampler=None,
                video_fps=12, frames_per_video=24):

    categories = ["coherent", "scrambled"]

    # Group files by category
    datasets = Pitzalis_category_dataset(transform=transform, frames_per_video=frames_per_video, video_fps=video_fps)
    datasets = {cat: datasets[cat] for cat in categories}
    n_categories = len(categories)
    t_vals_dict = t_test(
        model, transform, 
        datasets=datasets, contrasts=[(1, -1)],
        batch_size=batch_size, device=device, downsampler=downsampler,
        video_fps=video_fps, frames_per_video=frames_per_video
    )[0]

    ret = {"V6": t_vals_dict["coherent_vs_scrambled"]}

    return ret
    

    