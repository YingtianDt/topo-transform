from utils import cached
from models import vit_transform
from validate.floc import *
from validate import loadload_transformed_model


FLOC_DATASETS = ['vpnl', 'kanwisher', 'motion', 'pitzalis', 'biomotion', 'pitcher', 'temporal']

def _localizers(
        checkpoint_name, 
        dataset_names, 
        device='cuda',
        frames_per_video=24,
        video_fps=12,
    ):

    if viz_params is None:
        viz_params = {}

    model = loadload_transformed_model(checkpoint_name=checkpoint_name, device=device)
    model.eval()
    transform = vit_transform
    layer_positions = [lp.coordinates.cpu() for lp in model.layer_positions]

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
        else:
            raise ValueError(f"Unknown dataset_name: {dataset_name}")

        t_vals_dicts.append(t_vals_dict)

    return t_vals_dicts, layer_positions


def localizers(checkpoint_name, dataset_names=FLOC_DATASETS, device='cuda', frames_per_video=24, video_fps=12):
    import hashlib
    dataset_str = '_'.join(sorted(dataset_names))
    hash_suffix = hashlib.md5(dataset_str.encode()).hexdigest()[:8]
    return cached(f"localizers_{checkpoint_name}_{hash_suffix}")(
        _localizers
    )(checkpoint_name, dataset_names, device=device, frames_per_video=frames_per_video, video_fps=video_fps)