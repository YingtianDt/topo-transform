from utils import cached
from .utils import *

from .categories import localize_categories
from .temporal import localize_temporal
from .motion import localize_motion
from .v6 import localize_v6
from .psts import localize_psts
from .pitcher import localize_pitcher, localize_pitcher_human
from .robert import localize_robert, load_robert_tvals, localize_robert_human
from .afraz import localize_afraz

FLOC_DATASETS = ['vpnl', 'kanwisher', 'pitzalis', 'biomotion', 'pitcher', 'robert']

def validate_floc(
        model, 
        transform, 
        dataset_names, 
        epoch=None, 
        viz_dir=None, 
        viz_patches=False, 
        viz_params=None,
        batch_size=32, 
        device='cuda',
        frames_per_video=24,
        video_fps=12,
        plot_individual=False,
        plot_aggregate=False,
    ):

    if viz_params is None:
        viz_params = {}

    if viz_dir is not None and epoch is None:
        viz_dir = viz_dir / "eval"
        viz_dir.mkdir(parents=True, exist_ok=True)

    if model.smoothing:
        layer_positions = [lp.coordinates.cpu() for lp in model.smoothed_layer_positions]
        layer_dims = [lp.dims for lp in model.smoothed_layer_positions]
    else:
        layer_positions = [lp.coordinates.cpu() for lp in model.layer_positions]
        layer_dims = [lp.dims for lp in model.layer_positions]

    t_vals_dicts = []
    for dataset_name in dataset_names:
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
        elif dataset_name == "temporal":
            t_vals_dict = localize_temporal(
                model,
                transform,
                batch_size=batch_size,
                device=device,
                frames_per_video=frames_per_video,
                video_fps=video_fps,
            )
        elif dataset_name == "pitcher":
            print("For Pitcher dataset, using duration = 3 seconds")
            t_vals_dict = localize_pitcher(
                model,
                transform,
                batch_size=batch_size,
                device=device,
                frames_per_video=video_fps*3,
                video_fps=video_fps,
            )
        elif dataset_name == "robert":
            print("For Robert dataset, using duration = 3 seconds")
            t_vals_dict = localize_robert(
                model,
                transform,
                batch_size=batch_size,
                device=device,
                frames_per_video=video_fps*3,
                video_fps=video_fps,
            )
        else:
            raise ValueError(f"Unknown dataset_name: {dataset_name}")

        t_vals_dicts.append(t_vals_dict)

        if viz_dir is not None and plot_individual:
            suffix = f'_{epoch+1}' if epoch is not None else ''
            visualize_tvals(t_vals_dict, layer_positions, viz_dir, prefix=f'{dataset_name}_', suffix=suffix, **viz_params)
            if viz_patches:
                visualize_patches(t_vals_dict, layer_positions, viz_dir, prefix=f'{dataset_name}_', suffix=f'_patches{suffix}')

    if viz_dir is not None and plot_aggregate:
        suffix = f'_{epoch+1}' if epoch is not None else ''
        visualize_all_rois_v2(t_vals_dicts, layer_positions, viz_dir, prefix=f'rois_', suffix=suffix)

    return t_vals_dicts


def validate_floc_human(
    dataset_names,
):
    t_vals_dicts = []
    for dataset_name in dataset_names:
        if dataset_name == "pitcher":
            t_vals_dict = localize_pitcher_human()
        elif dataset_name == "robert":
            t_vals_dict = localize_robert_human()
        else:
            raise ValueError(f"Unknown dataset_name for human: {dataset_name}")
        t_vals_dicts.append(t_vals_dict)
    return t_vals_dicts
