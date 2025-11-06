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
from .utils import video_transform, safe_decoder, run_features, visualize_tvals


VPNL = '/mnt/scratch/ytang/datasets/fLoc_stimuli'
KANWISHER = '/mnt/scratch/ytang/datasets/lahner/stimulus_set/stimuli/localizer'


class CategoryDataset(Dataset):
    def __init__(self, file_infos, transform=None, frames_per_video=24, video_fps=12):
        self.file_paths = [info[0] for info in file_infos]
        self.labels = [info[1] for info in file_infos]
        self.transform = transform
        self.frames_per_video = frames_per_video
        self.video_fps = video_fps
        self.video_exts = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'}
        
        # Extract unique categories from labels
        self.categories = sorted(list(set(self.labels)))
        self.category_to_idx = {cat: idx for idx, cat in enumerate(self.categories)}
    
    def __len__(self):
        return len(self.file_paths)
    
    def _is_video(self, path):
        return Path(path).suffix.lower() in self.video_exts
    
    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.category_to_idx[self.labels[idx]]
        
        if self._is_video(file_path):
            decoder = safe_decoder(file_path)
            duration = decoder.metadata.duration_seconds
            time_duration = self.frames_per_video / self.video_fps
            
            if duration < time_duration:
                time_start, time_end = 0, duration
            else:
                time_start = (duration - time_duration) / 2
                time_end = time_start + time_duration
            
            data = video_transform(
                file_path, time_start, time_end,
                torch_transforms=self.transform or transforms.Lambda(lambda x: x),
                fps=self.video_fps
            )
        else:
            img = Image.open(file_path).convert('RGB')
            img = transforms.ToTensor()(img)
            if self.transform:
                img = self.transform(img)
            data = img.unsqueeze(0).repeat(self.frames_per_video, 1, 1, 1)
        
        return (data, label)


def VPNL_category_dataset(data_dir=VPNL, transform=None, frames_per_video=24, video_fps=12):
    """Create a category dataset for the VPNL dataset."""
    file_infos = []
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
            file_infos.append((os.path.join(data_dir, fname), category))

    return CategoryDataset(file_infos, transform=transform, frames_per_video=frames_per_video, video_fps=video_fps)


def KANWISHER_category_dataset(data_dir=KANWISHER, transform=None, frames_per_video=24, video_fps=12):
    """Create a category dataset for the Kanwisher dataset."""
    file_infos = []
    for category in os.listdir(data_dir):
        if category == "Scrambled15G": # skip scrambled images
            continue
        category_dir = os.path.join(data_dir, category)
        if os.path.isdir(category_dir):
            for fname in os.listdir(category_dir):
                if fname.endswith(('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv')):
                    file_infos.append((os.path.join(category_dir, fname), category))
    return CategoryDataset(file_infos, transform=transform, frames_per_video=frames_per_video, video_fps=video_fps)


def functional_localization_one_vs_rest(model, transform, dataset, 
                                        batch_size=32, device='cuda', downsampler=None):

    # Group files by category
    category_files = defaultdict(list)
    for file_path, label in zip(dataset.file_paths, dataset.labels):
        category_files[label].append((file_path, label))
    
    print(f"Found categories: {list(category_files.keys())}")
    print(f"Activations over time are averaged for each stimulus.")
    
    # Extract features
    category_feats = {}
    for cat_name, file_infos in category_files.items():
        print(f"Extracting features for {cat_name}...")
        cat_dataset = CategoryDataset(file_infos, transform=transform, 
                                      frames_per_video=dataset.frames_per_video,
                                      video_fps=dataset.video_fps)
        loader = DataLoader(cat_dataset, batch_size=batch_size, num_workers=int(batch_size/1.5), shuffle=False)
        feats, _ = run_features(model, loader, device, downsampler)
        feats = [np.mean(f, axis=1) for f in feats]  # NOTE: average over time
        category_feats[cat_name] = feats
    
    # Compute one-vs-rest t-statistics
    t_vals_dict = {}
    for target_cat in category_feats.keys():
        print(f"Computing t-stats for {target_cat} vs rest...")
        target_feats = category_feats[target_cat]
        other_cats = [cat for cat in category_feats.keys() if cat != target_cat]
        num_layers = len(target_feats)
        
        other_feats = [
            np.concatenate([category_feats[cat][i] for cat in other_cats], axis=0)
            for i in range(num_layers)
        ]
        t_vals = [
            stats.ttest_ind(target_feats[i], other_feats[i], axis=0)[0]
            for i in tqdm(range(num_layers), desc=f"t-stats: {target_cat}")
        ]
        t_vals_dict[target_cat] = t_vals

        # Print summary statistics
        for i in range(num_layers):
            print(f"  Layer {i}: mean |t| = {np.mean(np.abs(t_vals[i])):.3f}, "
                  f"max |t| = {np.max(np.abs(t_vals[i])):.3f}")
    
    return t_vals_dict


def validate_floc(model, transform, dataset_name, viz_dir, epoch, viz_params={}):
    if dataset_name == "vpnl":
        dataset = VPNL_category_dataset(transform=transform)
    elif dataset_name == "kanwisher":
        dataset = KANWISHER_category_dataset(transform=transform)
    else:
        raise ValueError(f"Unknown dataset_name: {dataset_name}")

    layer_positions = [lp.coordinates.cpu() for lp in model.layer_positions]
    functional_localization_one_vs_rest_wrapped = cached(
        f'floc_one_vs_rest_{dataset_name}_epoch_{epoch}',
    )(functional_localization_one_vs_rest)
    t_vals_dict = functional_localization_one_vs_rest_wrapped(model, transform=transform, dataset=dataset)
    visualize_tvals(t_vals_dict, layer_positions, viz_dir, prefix=f'{dataset_name}_', suffix=f'_{epoch+1}', **viz_params)