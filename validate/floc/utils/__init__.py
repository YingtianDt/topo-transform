import torch
from torchvision import transforms
from pathlib import Path
from torchcodec.decoders import VideoDecoder
from tqdm import tqdm

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from .basic import visualize_tvals
from .cluster import visualize_patches


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

    for tmp in tqdm(data_loader, desc="Extracting features"):
        data, target = tmp[0], tmp[1]
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

