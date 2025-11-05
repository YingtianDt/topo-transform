import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from pathlib import Path
from torchcodec.decoders import VideoDecoder
from collections import defaultdict
from tqdm import tqdm

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy import stats


VPNL = '/mnt/scratch/ytang/datasets/fLoc_stimuli'
KANWISHER = '/mnt/scratch/ytang/datasets/lahner/stimulus_set/stimuli/localizer'



def run_features(model, data_loader, device, downsampler=None):
    """Run feature extraction on a data loader.
    
    Returns:
        offline_feats: np.ndarray or list of np.ndarray [num_samples, ...]
                       assuming features [B, T, ...]
        offline_targets: np.ndarray [num_samples, ...]
    """
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    all_feats = []
    all_targets = []

    for data, target in tqdm(data_loader, desc="Extracting features"):
        data = data.to(device, non_blocking=True)

        with torch.no_grad():
            outputs = model(data)[0]
            
            # Apply downsampler if provided
            if downsampler is not None:
                outputs = [downsampler(out) for out in outputs] if isinstance(outputs, (list, tuple)) else downsampler(outputs)
            
            # Handle list or single output
            if isinstance(outputs, (list, tuple)):
                outputs = [out.cpu().numpy() for out in outputs]
                if len(all_feats) == 0:
                    all_feats = [[] for _ in range(len(outputs))]
                for j, out in enumerate(outputs):
                    all_feats[j].append(out)
            else:
                all_feats.append(outputs.cpu().numpy())
        
        all_targets.append(target.cpu().numpy())

    # Concatenate results
    if isinstance(all_feats[0], list):
        all_feats = [np.concatenate(feats, 0) for feats in all_feats]
    else:
        all_feats = np.concatenate(all_feats, 0)
    all_targets = np.concatenate(all_targets, 0)

    return all_feats, all_targets


def safe_decoder(path):
    """Create a VideoDecoder safely"""
    return VideoDecoder(path)


def safe_get_frames(decoder, frame_indices, max_attempt=10):
    """
    Safely get frames from a torchcodec decoder.

    - If frame 0 fails, searches for the first valid frame up to max_attempt.
    - Shifts all indices so decoding starts from the first valid frame.
    """
    # Try to decode requested frames
    try:
        return decoder.get_frames_at(frame_indices).data
    except RuntimeError as e:
        print(f"[WARN] Failed to get frames {frame_indices}: {e}")
        # search for first valid frame
        first_valid = None
        for i in range(max_attempt):
            try:
                _ = decoder.get_frames_at([i]).data
                first_valid = i
                print(f"[INFO] Found first valid frame at index {i}")
                break
            except RuntimeError:
                continue
        if first_valid is None:
            raise RuntimeError(f"No valid frames found in first {max_attempt} frames for {decoder.source}")

        # shift indices to start from first_valid
        safe_indices = [max(idx, first_valid) for idx in frame_indices]
        return decoder.get_frames_at(safe_indices).data


def video_transform(path, time_start, time_end, torch_transforms, fps=12):
    """
    Load video, clip by time, pad out-of-bound frames with gray [255/2] frames,
    resample fps, and apply torchvision transforms.

    Args:
        path (str): path to video file
        time_start (float): start time in seconds
        time_end (float): end time in seconds
        torch_transforms (callable): torchvision transforms to apply
        fps (int): target frames per second

    Returns:
        Tensor: [num_frames, channels, height, width]
    """
    decoder = safe_decoder(path)
    metadata = decoder.metadata
    num_video_frames = metadata.num_frames
    video_fps = metadata.average_fps
    video_duration = metadata.duration_seconds

    # Convert requested times to frame indices (float)
    start_frame_f = time_start * video_fps
    end_frame_f = time_end * video_fps

    # Compute actual frame indices in bounds
    start_idx = max(0, int(start_frame_f))
    end_idx = min(num_video_frames, int(end_frame_f))

    # Number of frames before/after for padding
    pad_start = max(0, int(-start_frame_f))
    pad_end = max(0, int(end_frame_f - num_video_frames))

    # Extract frames in bounds
    if start_idx < end_idx:
        frame_indices = list(range(start_idx, end_idx))
        frames = safe_get_frames(decoder, frame_indices)  # [num_frames, H, W, C]
    else:
        frames = torch.empty((0, 0, 0, 0))

    # Determine target number of frames
    num_target_frames = int(round((time_end - time_start) * fps))

    # Case 1: Entire range is outside video → return pure gray frames
    if start_idx >= end_idx:
        frames = torch.full(
            (num_target_frames, 3, 224, 224), 255 / 2, dtype=torch.float32
        )
    else:
        # Pad start
        if pad_start > 0:
            pad_frames = torch.full(
                (pad_start, frames.shape[1], frames.shape[2], frames.shape[3]),
                255 / 2,
                dtype=frames.dtype,
            )
            frames = torch.cat([pad_frames, frames], dim=0)

        # Pad end
        if pad_end > 0:
            pad_frames = torch.full(
                (pad_end, frames.shape[1], frames.shape[2], frames.shape[3]),
                255 / 2,
                dtype=frames.dtype,
            )
            frames = torch.cat([frames, pad_frames], dim=0)

        # Resample to target FPS
        if frames.shape[0] != num_target_frames and frames.shape[0] > 0:
            indices = torch.linspace(0, frames.shape[0] - 1, steps=num_target_frames).long()
            frames = frames[indices]

    if frames.numel() > 0:
        # Normalize and apply transforms
        frames = torch_transforms(frames / 255.0)

    return frames


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
            file_infos.append((os.path.join(data_dir, fname), category))

    return CategoryDataset(file_infos, transform=transform, frames_per_video=frames_per_video, video_fps=video_fps)


def KANWISHER_category_dataset(data_dir=KANWISHER, transform=None, frames_per_video=24, video_fps=12):
    """Create a category dataset for the Kanwisher dataset."""
    file_infos = []
    for category in os.listdir(data_dir):
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
        feats = [np.mean(f, axis=1) for f in feats]  # average over time
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
    
    return t_vals_dict


def visualize_tvals(t_vals_dict, layer_positions, store_dir, figsize_per_panel=5, prefix='', suffix=''):
    """Visualize t-statistics for each category and layer.
    
    Args:
        t_vals_dict: Dict mapping category names to list of t-value arrays (one per layer)
        layer_positions: List of position arrays for each layer [num_units, 2]
        store_dir: Directory to save visualizations
        figsize_per_panel: Size of each subplot panel
    """
    os.makedirs(store_dir, exist_ok=True)
    
    categories = list(t_vals_dict.keys())
    n_layers = len(t_vals_dict[categories[0]]) if categories else 0
    
    # Create separate figure for each category
    for cat_name, t_vals_list in t_vals_dict.items():
        if not isinstance(t_vals_list, list):
            t_vals_list = [t_vals_list]
        
        n_cols = len(t_vals_list)
        fig, axes = plt.subplots(1, n_cols, 
                                figsize=(n_cols * figsize_per_panel, figsize_per_panel))
        
        # Ensure axes is always a list
        if n_cols == 1:
            axes = [axes]
        
        for layer_idx, t_vals in enumerate(t_vals_list):
            pos = layer_positions[layer_idx]
            if isinstance(pos, torch.Tensor):
                pos = pos.cpu().numpy()
            
            # Compute normalization based on absolute max
            vmax = np.abs(t_vals).max()
            norm = Normalize(vmin=-vmax, vmax=vmax)
            
            # Create scatter plot
            im = axes[layer_idx].scatter(pos[:, 0], pos[:, 1], c=t_vals.flatten(), 
                                        cmap='bwr', norm=norm, s=0.1)
            axes[layer_idx].set_title(f'Layer {layer_idx}')
            axes[layer_idx].axis('equal')
            axes[layer_idx].set_aspect('equal', 'box')
            
        # Add colorbar
        fig.colorbar(im, ax=axes, orientation='horizontal', 
                    pad=0.02, fraction=0.046, label='t-statistic')
        plt.suptitle(f'{cat_name} (one-vs-rest)', fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(f'{store_dir}/{prefix}tvals_{cat_name}{suffix}.png', 
                    dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Saved visualizations to {store_dir}")


def validate_floc(model, transform, data_path, viz_dir, epoch):
    if data_path == VPNL:
        dataset = VPNL_category_dataset(data_dir=data_path, transform=transform)
        dataset_name = "VPNL"
    elif data_path == KANWISHER:
        dataset = KANWISHER_category_dataset(data_dir=data_path, transform=transform)
        dataset_name = "KANWISHER"
    else:
        raise ValueError(f"Unknown data_path: {data_path}")
    
    layer_positions = [lp.coordinates.cpu() for lp in model.layer_positions]
    t_vals_dict = functional_localization_one_vs_rest(model, transform=transform, dataset=dataset)
    visualize_tvals(t_vals_dict, layer_positions, viz_dir, prefix=f'{dataset_name}_', suffix=f'_{epoch+1}')