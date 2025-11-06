import torch
import numpy as np
from scipy import stats
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm
import copy

from utils import cached
from data.smthsmthv2 import SmthSmthV2
from .utils import run_features, visualize_tvals


class OrderingWrapper(Dataset):
    """
    Wrapper dataset that modifies the ordering of frames from an underlying dataset.
    """
    def __init__(self, base_dataset, transform, ordering='normal', seed=42):
        """
        Args:
            base_dataset: The underlying dataset (e.g., SmthSmthV2 trainset/valset)
            ordering: 'normal', 'shuffled', or 'static'
            seed: Random seed for reproducibility
        """
        self.base_dataset = base_dataset
        self.ordering = ordering
        self.seed = seed
        
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        # Get item from base dataset (always in normal ordering)
        tmp = self.base_dataset[idx]
        data, label = tmp[0], tmp[1]
        
        # data shape: (T, C, H, W)
        num_frames = data.shape[0]
        
        # Apply ordering transformation
        if self.ordering == 'shuffled':
            # Use deterministic shuffling based on seed and idx
            rng = np.random.RandomState(self.seed + idx)
            indices = torch.from_numpy(rng.permutation(num_frames))
            data = data[indices]
        elif self.ordering == 'static':
            # Use only the middle frame, repeated
            middle_idx = num_frames // 2
            data = data[middle_idx:middle_idx+1].repeat(num_frames, 1, 1, 1)
        # else: 'normal' - keep original ordering

        return data, label


def compare_video_orderings(model, transform, dataset, 
                            batch_size=32, device='cuda', downsampler=None, seed=42):
    """
    Compare features extracted from videos in different orderings.
    
    Args:
        model: The model to extract features from
        transform: Transform to apply (not used if dataset already has transforms)
        dataset: Base dataset object (e.g., SmthSmthV2().trainset or .valset)
        batch_size: Batch size for processing
        device: Device to run on
        downsampler: Optional downsampler
        seed: Random seed for reproducibility
        
    Returns:
        t_vals_dict: Dictionary with keys 'normal_vs_shuffled', 'normal_vs_static', 
                     'shuffled_vs_static', each containing list of t-values per layer
    """
    
    print(f"Processing {len(dataset)} videos with seed {seed}")
    print("Extracting features for each ordering...")
    
    # Extract features for each ordering
    orderings = ['normal', 'shuffled', 'static']
    ordering_feats = {}
    
    print("Averaging over time dimension for each video.")
    for ordering in orderings:
        print(f"\nExtracting features for {ordering} ordering...")
        
        # Create wrapped dataset with specific ordering
        wrapped_dataset = OrderingWrapper(dataset, transform, ordering=ordering, seed=seed)
        
        loader = DataLoader(
            wrapped_dataset, 
            batch_size=batch_size, 
            num_workers=min(8, int(batch_size/1.5)),
            shuffle=False,  # Important: don't shuffle to maintain correspondence
            pin_memory=True
        )
        
        feats, _ = run_features(model, loader, device, downsampler)
        # NOTE: Average over time dimension for each video
        feats = [np.mean(f, axis=1) for f in feats]
        ordering_feats[ordering] = feats
    
    # Compute paired t-tests for each comparison
    print("\nComputing paired t-tests...")
    comparisons = [
        ('normal', 'shuffled'),
        ('normal', 'static'),
        ('shuffled', 'static')
    ]
    
    t_vals_dict = {}
    for order1, order2 in comparisons:
        comparison_name = f'{order1}_vs_{order2}'
        print(f"\nComputing t-stats for {comparison_name}...")
        
        feats1 = ordering_feats[order1]
        feats2 = ordering_feats[order2]
        num_layers = len(feats1)
        
        # Paired t-test (same videos in different orderings)
        t_vals = [
            stats.ttest_rel(feats1[i], feats2[i], axis=0)[0]
            for i in tqdm(range(num_layers), desc=f"t-stats: {comparison_name}")
        ]
        t_vals_dict[comparison_name] = t_vals
        
        # Print summary statistics
        for i in range(num_layers):
            print(f"  Layer {i}: mean |t| = {np.mean(np.abs(t_vals[i])):.3f}, "
                  f"max |t| = {np.max(np.abs(t_vals[i])):.3f}")
    
    return t_vals_dict


def validate_temporal(model, transform, dataset_name, viz_dir, epoch, 
                      batch_size=32, device='cuda', num_samples=256, seed=42, viz_params={}):
    """
    Validate model by comparing responses to different video orderings.
    
    Args:
        model: The model to validate
        dataset: Dataset object (e.g., SmthSmthV2().trainset or .valset)
        viz_dir: Directory to save visualizations
        epoch: Current epoch number
        batch_size: Batch size for processing
        device: Device to run on
        downsampler: Optional downsampler
        seed: Random seed for reproducibility
    """
    
    # Get layer positions for visualization
    layer_positions = [lp.coordinates.cpu() for lp in model.layer_positions]
    
    # Get dataset
    if dataset_name == 'smthsmthv2':
        smthsmth = SmthSmthV2(train_transforms=transform, test_transforms=transform)
        dataset = smthsmth.valset
        dataset = Subset(dataset, list(range(min(num_samples, len(dataset)))))

    # Compute t-values for different orderings
    compare_video_orderings_wrapped = cached(
        f'floc_temporal_{dataset_name}_epoch_{epoch}_seed_{seed}',
    )(compare_video_orderings)
    t_vals_dict = compare_video_orderings_wrapped(
        model, transform=transform, dataset=dataset,
        batch_size=batch_size, device=device, seed=seed
    )
    
    # Visualize results
    visualize_tvals(
        t_vals_dict, 
        layer_positions, 
        viz_dir, 
        prefix=f'{dataset_name}_temporal_', 
        suffix=f'_epoch{epoch+1}',
        **viz_params
    )
    
    return t_vals_dict