import torch
import numpy as np
from scipy import stats
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm
import copy

from utils import cached
from data import AFD101, ImageNetVid
from .utils import run_features, visualize_tvals


# Produce motion index by reporting t-values contrasting AFD vs imagenet

def motion_index(model, transform, dataset_motion, dataset_static, 
                batch_size=32, device='cuda', downsampler=None, seed=42):

    print(f"Processing {len(dataset_motion)+len(dataset_static)} videos with seed {seed}")
    print("Extracting features for each ordering...")
    
    # Extract features for each ordering
    dataset_feats = []
    
    print("Averaging over time dimension for each video.")
    for dataset in [dataset_motion, dataset_static]:
        loader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            num_workers=min(8, int(batch_size/1.5)),
            shuffle=False,  # Important: don't shuffle to maintain correspondence
            pin_memory=True
        )
        
        feats, _ = run_features(model, loader, device, downsampler)
        # NOTE: Average over time dimension for each video
        feats = [np.mean(f, axis=1) for f in feats]
        dataset_feats.append(feats)
    
    # Compute t-tests for each comparison
    print("\nComputing t-tests as motion index...")
    
    feats1, feats2 = dataset_feats
    num_layers = len(feats1)
    
    # Paired t-test (same videos in different orderings)
    t_vals = [
        stats.ttest_rel(feats1[i], feats2[i], axis=0)[0]
        for i in tqdm(range(num_layers), desc="t-stats")  # Fixed: removed undefined comparison_name
    ]
    
    # Print summary statistics
    for i in range(num_layers):
        print(f"  Layer {i}: mean t = {np.mean(t_vals[i]):.3f}, "
                f"max |t| = {np.max(np.abs(t_vals[i])):.3f}")

    return t_vals


def validate_motion(model, transform, viz_dir, epoch, duration=2000, fps=12,
                    batch_size=32, device='cuda', num_samples=256, seed=42, 
                    viz_params=None):
    """
    Validate motion selectivity by computing motion index.
    
    Args:
        model: Model to extract features from
        transform: Transforms to apply
        viz_dir: Directory to save visualizations
        epoch: Current epoch number
        duration: Video duration in milliseconds
        fps: Frames per second
        batch_size: Batch size for feature extraction
        device: Device to run on
        num_samples: Number of samples to use from each dataset
        seed: Random seed
        viz_params: Visualization parameters dict
    """
    if viz_params is None:
        viz_params = {}
    
    # Get layer positions for visualization
    layer_positions = [lp.coordinates.cpu() for lp in model.layer_positions]
    
    motion_dataset = AFD101(
        test_transforms=transform,
        duration=duration,
        fps=fps
    )
    static_dataset = ImageNetVid(
        test_transforms=transform,
        train_transforms=transform,
        duration=duration,
        fps=fps
    )

    # Create subsets
    motion_dataset = Subset(
        motion_dataset.valset, 
        list(range(min(num_samples, len(motion_dataset.valset))))
    )
    static_dataset = Subset(
        static_dataset.valset, 
        list(range(min(num_samples, len(static_dataset.valset))))
    )

    motion_index_wrapped = cached("motion_index")(motion_index)

    # Compute motion index
    t_vals = motion_index_wrapped(
        model, 
        transform, 
        motion_dataset, 
        static_dataset, 
        batch_size=batch_size, 
        device=device,
        downsampler=None,  # Fixed: added missing parameter
        seed=seed
    )

    t_vals_dict = {"motion_index": t_vals}
    
    # Visualize results
    visualize_tvals(
        t_vals_dict, 
        layer_positions, 
        viz_dir, 
        prefix='motion_index_', 
        suffix=f'_epoch{epoch+1}',
        **viz_params  # Fixed: pass as viz_params parameter
    )
    
    return t_vals_dict