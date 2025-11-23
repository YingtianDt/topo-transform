from utils import cached
from models import vit_transform
from validate.floc import *
from validate import load_transformed_model


FLOC_DATASETS = ['vpnl', 'kanwisher', 'motion', 'pitzalis', 'biomotion', 'pitcher', 'temporal', 'robert']

def _localizers(
        checkpoint_name, 
        dataset_names, 
        device='cuda',
        batch_size=16,
        video_fps=12,
        frames_per_video=24,
        fwhm_mm=2.0,
        resolution_mm=1.0,
    ):

    model, epoch = load_transformed_model(checkpoint_name=checkpoint_name, device=device)
    model.eval()
    transform = vit_transform

    with model.smoothing_enabled(
            fwhm_mm=fwhm_mm, 
            resolution_mm=resolution_mm, 
        ):

        if model.smoothing:
            layer_positions = [lp.coordinates.cpu() for lp in model.smoothed_layer_positions]
        else:
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

        return t_vals_dicts, layer_positions


def localizers(
        checkpoint_name, 
        dataset_names=FLOC_DATASETS, 
        device='cuda', 
        frames_per_video=24, 
        video_fps=12,
        fwhm_mm=2.0,
        resolution_mm=1.0,
        ret_merged=False,
    ):
    import hashlib
    dataset_str = '_'.join(sorted(dataset_names))
    hash_suffix = hashlib.md5(dataset_str.encode()).hexdigest()[:8]
    t_vals_dicts, layer_positions = cached(f"localizers_{checkpoint_name}_{hash_suffix}_{fwhm_mm}_{resolution_mm}")(
        _localizers
    )(checkpoint_name, dataset_names, device=device, frames_per_video=frames_per_video, video_fps=video_fps, fwhm_mm=fwhm_mm, resolution_mm=resolution_mm)

    if ret_merged:
        merged_t_vals_dict = {}
        for t_vals_dict in t_vals_dicts:
            merged_t_vals_dict.update(t_vals_dict)
        t_vals_dicts = merged_t_vals_dict

    return t_vals_dicts, layer_positions


def get_localizer_model(rois, ckpt_name, t_perc=100, t_thres=None, fwhm_mm=2.0, resolution_mm=1.0):
    t_vals_dicts, layer_positions = localizers(ckpt_name, fwhm_mm=fwhm_mm, resolution_mm=resolution_mm)

    # merge t_vals_dicts
    t_val_dict = {}
    for t_vals in t_vals_dicts:
        t_val_dict.update(t_vals)

    def _tval_filter(t_val):
        if t_perc < 100:
            threshold = np.percentile(t_val.flatten(), 100-t_perc)
            mask = t_val >= threshold

        if t_thres is not None:
            mask = mask & (t_val >= t_thres) 
        return mask

    ret = []
    for roi in rois:
        if roi == "face":
            t_vals = t_val_dict["Faces_moving"]
        elif roi == "place":
            t_vals = t_val_dict["Scenes_moving"]
        elif roi == "body":
            t_vals = t_val_dict["Bodies_moving"]
        elif roi == "object":
            t_vals = t_val_dict["Objects_moving"]
        elif roi == "v6":
            t_vals = t_val_dict["V6"]
        elif roi == "psts":
            t_vals = t_val_dict["pSTS"]
        elif roi == "mt":
            t_vals = t_val_dict["MT-Huk"]
        else:
            raise ValueError(f"Unknown roi: {roi}")
        masks = [_tval_filter(t_val) for t_val in t_vals]  # layers
        ret.append(masks)
    return ret

def get_localizer_human(rois):
    from validate.rois.glasser import get_region_voxels
    ret = []
    for roi in rois:
        if roi == "face":
            mask = get_region_voxels(["FFC"])
        elif roi == "v6":
            mask = get_region_voxels(["V6"])
        elif roi == "psts":
            mask = get_region_voxels(["STSdp", "STSvp"])
        elif roi == "mt":
            mask = get_region_voxels(["MT"])
        else:
            raise ValueError(f"Unknown roi: {roi}")
        ret.append(mask)
    return ret


            
            