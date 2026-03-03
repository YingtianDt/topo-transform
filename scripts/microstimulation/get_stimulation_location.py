from utils import cached
from models import vit_transform
from validate.floc import *
from validate.floc.utils.cluster import find_patches
from validate import load_transformed_model
from validate.correction import fdr, fwe

from scripts.common import *
from scripts.get_localizers import localizers


def get_patches(rois, ckpt_name, p_thres=LOCALIZER_P_THRESHOLD, t_thres=LOCALIZER_T_THRESHOLD, fwhm_mm=2.0, resolution_mm=1.0):
    t_vals_dicts, p_vals_dicts, layer_positions = localizers(ckpt_name, fwhm_mm=fwhm_mm, resolution_mm=resolution_mm)

    # merge p_vals_dicts
    p_val_dict = {}
    for p_vals in p_vals_dicts:
        p_val_dict.update(p_vals)

    # merge t_vals_dicts
    t_val_dict = {}
    for t_vals in t_vals_dicts:
        t_val_dict.update(t_vals)

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
        layer_patches = [find_patches(p_val, t_val) for p_val, t_val in zip(p_vals, t_vals)]  # layers
        # assume single layer
        assert len(layer_patches) == 1
        patches = layer_patches[0]
        ret.append(patches)
    return ret

def get_patch_based_stimulation_location(rois, ckpt_name, fwhm_mm=2.0, resolution_mm=1.0):
    # choose the geometric center of the largest patch as stimulation location for each ROI
    patches = get_patches(rois, ckpt_name, fwhm_mm=fwhm_mm, resolution_mm=resolution_mm)
    locations = []
    for patch in patches:
        if len(patch) == 0:
            locations.append(None)
            continue
        largest_patch = max(patch, key=lambda x: x['size'])
        coords = largest_patch['coordinates']  # (N, 2)
        center = coords.mean(axis=0)  # (x, y)
        locations.append(center)
    return locations
    
def get_selectivity_based_stimulation_locations(roi, ckpt_name, fwhm_mm=2.0, resolution_mm=1.0, num_samples=20, seed=42):
    if roi == 'face-afraz':
        t_vals_dicts, p_vals_dicts, layer_positions = localizers(ckpt_name, ['afraz'], fwhm_mm=fwhm_mm, resolution_mm=resolution_mm)
        t_vals_dict = t_vals_dicts[0]
        p_vals_dict = p_vals_dicts[0]
        t_vals = t_vals_dict['face_vs_nonface'][0]

        # select from face region
        t_vals_dict, p_vals_dict, layer_positions = localizers(ckpt_name, ret_merged=True)
        p_vals_f = p_vals_dict['face'][0]
        t_vals[p_vals_f > LOCALIZER_P_THRESHOLD] = -100  # mask out non-significant locations

    elif roi == 'face':
        t_vals_dict, p_vals_dict, layer_positions = localizers(ckpt_name, ret_merged=True)
        t_vals = t_vals_dict['face']
        t_vals = t_vals[0]
        t_vals[:, :, -71:] = -100  # mask out non-central locations

    t_vals = t_vals.flatten()
    positions = layer_positions[0]  # (x, y)

    # sample randomly num_samples points from the suprathreshold locations
    suprathreshold_indices = np.where(t_vals > 0)[0]
    if len(suprathreshold_indices) == 0:
        raise ValueError(f"No suprathreshold locations found for ROI: {roi}")
    np.random.seed(seed)
    sampled_indices = np.random.choice(suprathreshold_indices, size=min(num_samples, len(suprathreshold_indices)), replace=False)
    sampled_t_vals = t_vals[sampled_indices]
    sampled_positions = positions[sampled_indices]

    # sort by t_vals descending
    sorted_indices = np.argsort(-sampled_t_vals)
    sampled_t_vals = sampled_t_vals[sorted_indices]
    sampled_positions = sampled_positions[sorted_indices]

    locations = sampled_positions.numpy()
    selecitivities = sampled_t_vals
    
    return locations, selecitivities

def get_random_stimulation_locations(model, num_samples=100, seed=42, fwhm_mm=2.0, resolution_mm=1.0):
    with model.smoothing_enabled(
            fwhm_mm=fwhm_mm, 
            resolution_mm=resolution_mm, 
        ):
        if model.smoothing:
            layer_positions = [lp.coordinates.cpu() for lp in model.smoothed_layer_positions]
        else:
            layer_positions = [lp.coordinates.cpu() for lp in model.layer_positions]


    # constraint to category-selective regions
    regions = [
        'Faces_localizer',
        'Scenes_localizer',
        'Bodies_localizer',
        'Objects_localizer',
    ]
    mask = torch.zeros(layer_positions[0].shape[0], dtype=torch.bool)
    ckpt_name = model.name
    t_vals_dict, p_vals_dict, layer_positions = localizers(ckpt_name, ret_merged=True)
    for region in regions:
        p_vals = p_vals_dict[region][0].flatten()
        mask |= (p_vals < LOCALIZER_P_THRESHOLD)

    positions = layer_positions[0]  # (x, y)
    num_neurons = positions.shape[0]
    np.random.seed(seed)
    valid_indices = torch.where(mask)[0].numpy()
    if len(valid_indices) == 0:
        raise ValueError("No suprathreshold locations found for random stimulation locations.")
    sampled_indices = np.random.choice(valid_indices, size=min(num_samples, len(valid_indices)), replace=False)
    sampled_positions = positions[sampled_indices]
    locations = sampled_positions.numpy()
    return locations, sampled_indices
