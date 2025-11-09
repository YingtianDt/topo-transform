from utils import cached
from .utils import video_transform, safe_decoder, run_features, visualize_tvals, visualize_patches

from .categories import localize_categories
from .temporal import *
from .motion import localize_motion
from .v6 import localize_v6
from .psts import localize_psts

FLOC_DATASETS = ['vpnl', 'kanwisher', 'motion', 'pitzalis', 'biomotion']

def validate_floc(
        model, 
        transform, 
        dataset_name, 
        epoch=None, 
        viz_dir=None, 
        viz_patches=False, 
        viz_params=None,
        batch_size=32, 
        device='cuda',
        frames_per_video=24,
        video_fps=12,
    ):

    if viz_params is None:
        viz_params = {}

    layer_positions = [lp.coordinates.cpu() for lp in model.layer_positions]

    if dataset_name in ["vpnl", "kanwisher"]:
        t_vals_dict = localize_categories(
            model, 
            transform, 
            dataset_name=dataset_name,
            batch_size=batch_size, 
            device=device,
            frames_per_video=frames_per_video,
            video_fps=video_fps,
        )
    elif dataset_name == "motion":
        t_vals_dict = localize_motion(
            model, 
            transform, 
            batch_size=batch_size, 
            device=device,
            frames_per_video=frames_per_video,
            video_fps=video_fps,
        )
    elif dataset_name == "pitzalis":
        t_vals_dict = localize_v6(
            model,
            transform,
            batch_size=batch_size,
            device=device,
            frames_per_video=frames_per_video,
            video_fps=video_fps,
        )
    elif dataset_name == "biomotion":
        t_vals_dict = localize_psts(
            model,
            transform,
            batch_size=batch_size,
            device=device,
            frames_per_video=frames_per_video,
            video_fps=video_fps,
        )
    else:
        raise ValueError(f"Unknown dataset_name: {dataset_name}")

    if viz_dir is not None:
        suffix = f'_{epoch+1}' if epoch is not None else ''
        visualize_tvals(t_vals_dict, layer_positions, viz_dir, prefix=f'{dataset_name}_', suffix=suffix, **viz_params)
        if viz_patches:
            visualize_patches(t_vals_dict, layer_positions, viz_dir, prefix=f'{dataset_name}_', suffix=f'_patches{suffix}')