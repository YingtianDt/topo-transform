from utils import cached
from models import vit_transform
from validate.floc import *
from validate import load_transformed_model
from validate.correction import fdr, fwe

from .common import *


FLOC_DATASETS = ['vpnl', 'kanwisher', 'pitzalis', 'biomotion', 'pitcher', 'robert']
LOCALIZER_RERUN = False

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
        p_vals_dicts = []
        for dataset_name in dataset_names:
            if dataset_name in ["vpnl", "kanwisher"]:
                t_vals_dict, p_vals_dict = localize_categories(
                    model, 
                    transform, 
                    dataset_name=dataset_name,
                    batch_size=batch_size, 
                    device=device,
                    frames_per_video=frames_per_video,
                    video_fps=video_fps,
                    ret_pvals=True,
                )
            elif dataset_name == "motion":
                t_vals_dict, p_vals_dict = localize_motion(
                    model, 
                    transform, 
                    batch_size=batch_size, 
                    device=device,
                    frames_per_video=frames_per_video,
                    video_fps=video_fps,
                    ret_pvals=True,
                )
            elif dataset_name == "pitzalis":
                t_vals_dict, p_vals_dict = localize_v6(
                    model,
                    transform,
                    batch_size=batch_size,
                    device=device,
                    frames_per_video=frames_per_video,
                    video_fps=video_fps,
                    ret_pvals=True,
                )
            elif dataset_name == "biomotion":
                t_vals_dict, p_vals_dict = localize_psts(
                    model,
                    transform,
                    batch_size=batch_size,
                    device=device,
                    frames_per_video=frames_per_video,
                    video_fps=video_fps,
                    ret_pvals=True,
                )
            elif dataset_name == "temporal":
                t_vals_dict, p_vals_dict = localize_temporal(
                    model,
                    transform,
                    batch_size=batch_size,
                    device=device,
                    frames_per_video=frames_per_video,
                    video_fps=video_fps,
                    ret_pvals=True,
                )
            elif dataset_name == "pitcher":
                print("For Pitcher dataset, using duration = 3 seconds")
                t_vals_dict, p_vals_dict = localize_pitcher(
                    model,
                    transform,
                    batch_size=batch_size,
                    device=device,
                    frames_per_video=video_fps*3,
                    video_fps=video_fps,
                    ret_pvals=True,
                )
            elif dataset_name == "robert":
                print("For Robert dataset, using duration = 3 seconds")
                t_vals_dict, p_vals_dict = localize_robert(
                    model,
                    transform,
                    batch_size=batch_size,
                    device=device,
                    frames_per_video=video_fps*3,
                    video_fps=video_fps,
                    ret_pvals=True,
                )
            else:
                raise ValueError(f"Unknown dataset_name: {dataset_name}")

            t_vals_dicts.append(t_vals_dict)
            p_vals_dicts.append(p_vals_dict)

        return t_vals_dicts, p_vals_dicts, layer_positions


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
    t_vals_dicts, p_vals_dicts, layer_positions = cached(
        f"localizers_{checkpoint_name}_{hash_suffix}_{fwhm_mm}_{resolution_mm}",
        rerun=LOCALIZER_RERUN
    )(_localizers)(checkpoint_name, dataset_names, device=device, frames_per_video=frames_per_video, video_fps=video_fps, fwhm_mm=fwhm_mm, resolution_mm=resolution_mm)

    # p val correction
    tmp = []    
    for p_vals_dict in p_vals_dicts:
        fdr_p_vals_dict = {}
        for roi_name, p_vals in p_vals_dict.items():
            shapes = [p_val.shape for p_val in p_vals]
            fdr_p_vals = [fwe(p_val) for p_val in p_vals]
            fdr_p_vals_dict[roi_name] = [fdr_p_val.reshape(shape) for fdr_p_val, shape in zip(fdr_p_vals, shapes)]
        tmp.append(fdr_p_vals_dict)
    p_vals_dicts = tmp

    if ret_merged:
        merged_t_vals_dict = {}
        for t_vals_dict in t_vals_dicts:
            merged_t_vals_dict.update(t_vals_dict)
        t_vals_dicts = merged_t_vals_dict

        merged_p_vals_dict = {}
        for p_vals_dict in p_vals_dicts:
            merged_p_vals_dict.update(p_vals_dict)
        p_vals_dicts = merged_p_vals_dict

    return t_vals_dicts, p_vals_dicts, layer_positions


def get_localizer_model(rois, ckpt_name, p_thres=LOCALIZER_P_THRESHOLD, t_thres=0, fwhm_mm=2.0, resolution_mm=1.0):
    t_vals_dicts, p_vals_dicts, layer_positions = localizers(ckpt_name, fwhm_mm=fwhm_mm, resolution_mm=resolution_mm)

    # merge p_vals_dicts
    p_val_dict = {}
    for p_vals in p_vals_dicts:
        p_val_dict.update(p_vals)

    # merge t_vals_dicts
    t_val_dict = {}
    for t_vals in t_vals_dicts:
        t_val_dict.update(t_vals)

    def _filter(p_val, t_val):
        mask = (p_val < p_thres) & (t_val > t_thres)
        return mask

    ret = []
    for roi in rois:
        if roi == "face":
            p_vals = p_val_dict["face"]
            t_vals = t_val_dict["face"]
        elif roi == "place":
            p_vals = p_val_dict["place"]
            t_vals = t_val_dict["place"]
        elif roi == "body":
            p_vals = p_val_dict["body"]
            t_vals = t_val_dict["body"]
        elif roi == "object":
            p_vals = p_val_dict["object"]
            t_vals = t_val_dict["object"]
        elif roi == "v6":
            p_vals = p_val_dict["V6"]
            t_vals = t_val_dict["V6"]   
        elif roi == "psts":
            p_vals = p_val_dict["pSTS"]
            t_vals = t_val_dict["pSTS"]
        elif roi == "mt":
            p_vals = p_val_dict["MT-Huk"]
            t_vals = t_val_dict["MT-Huk"]
        else:
            raise ValueError(f"Unknown roi: {roi}")
        masks = [_filter(p_val, t_val) for p_val, t_val in zip(p_vals, t_vals)]  # layers
        ret.append(masks)
    return ret


def get_localizer_human(rois):
    from validate.rois import glasser, visf

    ret = []
    for roi in rois:
        if roi == "face":
            mask = visf.get_region_voxels("faces")
        elif roi == "place":
            mask = visf.get_region_voxels("places")
        elif roi == "body":
            mask = visf.get_region_voxels("bodies")
        elif roi == "character":
            mask = visf.get_region_voxels("characters")
        elif roi == "v6":
            mask = glasser.get_region_voxels(["V6"])
        elif roi == "psts":
            mask = glasser.get_region_voxels(["TPOJ2","TPOJ1"])
        elif roi == "mt":
            mask = visf.get_region_voxels("hMT")
        else:
            raise ValueError(f"Unknown roi: {roi}")

        ret.append(mask)
    return ret