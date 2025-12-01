from utils import cached
from models import vit_transform
from validate.floc import *
from validate.floc.utils.cluster import find_patches
from validate import load_transformed_model
from validate.correction import fdr, fwe

from .common import *


def get_patches(rois, ckpt_name, p_thres=LOCALIZER_P_THRESHOLD, t_thres=0, fwhm_mm=2.0, resolution_mm=1.0):
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

def get_stimulation_location(rois, ckpt_name, fwhm_mm=2.0, resolution_mm=1.0):
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
        locations.append((int(center[0]), int(center[1])))
    return locations
    