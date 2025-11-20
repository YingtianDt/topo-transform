import libpysal as lp
from .utils import *
from validate.floc import *
from tqdm import tqdm

from validate.rois.nsd import get_region_voxels
NSD_HIGH = get_region_voxels(["high-ventral", "high-lateral", "high-dorsal"])
human_lh_adj_list, human_rh_adj_list = compute_nsd_high_adjacency_list()

def validate_smoothness(
        model,
        transform,
        dataset_name,
        batch_size=32,
        device='cuda',
        frames_per_video=24,
        video_fps=12,
        fwhm_mm=2.0,
        resolution_mm=1.0,
    ):

    print("Assuming single sheet and smoothing...")
    
    # model
    with model.smoothing_enabled(fwhm_mm=fwhm_mm, resolution_mm=resolution_mm):
        model_t_val_dict = validate_floc(
            model,
            transform,
            dataset_names=[dataset_name],
            batch_size=batch_size,
            device=device,
            frames_per_video=frames_per_video,
            video_fps=video_fps,
        )[0]

        model_sheet_dims = model.smoothed_layer_positions[0].dims
        model_w = lp.weights.lat2W(*model_sheet_dims[-2:], rook=False)
        model_cates = list(model_t_val_dict.keys())
        model_t_val_dict = {k: v[0] for k, v in model_t_val_dict.items()} # first layer only

    # human
    human_t_val_dict = validate_floc_human(
        dataset_names=[dataset_name],
    )[0]
    human_lh_t_val_dict, human_rh_t_val_dict = _filter_high_regions(human_t_val_dict)
    human_lh_adj_list, human_rh_adj_list = compute_nsd_high_adjacency_list()
    human_cates = list(human_lh_t_val_dict.keys())

    common_cates = list(set(model_cates) & set(human_cates))
    print(f"Common categories: {common_cates}")
    smoothness_results = {}
    for cate in common_cates:
        model_t_vals = model_t_val_dict[cate]
        human_lh_t_vals = human_lh_t_val_dict[cate]
        human_rh_t_vals = human_rh_t_val_dict[cate]

        model_smoothness = Moran(model_t_vals, model_w).I
        human_lh_smoothness = compute_morans_I_neural(human_lh_t_vals, human_lh_adj_list)
        human_rh_smoothness = compute_morans_I_neural(human_rh_t_vals, human_rh_adj_list)
        human_smoothness = (human_lh_smoothness + human_rh_smoothness) / 2.0

        print(f"Cate: {cate}, Model smoothness: {model_smoothness:.4f}, Human smoothness: {human_smoothness:.4f}")

        smoothness_results[cate] = {
            'model_smoothness': model_smoothness,
            'human_smoothness': human_smoothness,
        }

    return smoothness_results

def _filter_high_regions(t_val_dict):
    sel = NSD_HIGH
    sel_lh = sel[:len(sel)//2]
    sel_rh = sel[len(sel)//2:]
    lh_t_val_dict = {}
    rh_t_val_dict = {}
    num_layers = len(next(iter(t_val_dict.values())))
    for cat_name in t_val_dict:
        t_vals = t_val_dict[cat_name]
        lh_t_vals = t_vals[:len(t_vals)//2][sel_lh]
        rh_t_vals = t_vals[len(t_vals)//2:][sel_rh]
        if cat_name not in lh_t_val_dict:
            lh_t_val_dict[cat_name] = []
            rh_t_val_dict[cat_name] = []
        lh_t_val_dict[cat_name] = lh_t_vals
        rh_t_val_dict[cat_name] = rh_t_vals
    return lh_t_val_dict, rh_t_val_dict

def compute_activity_smoothness_neural(activity):
    from .utils import compute_morans_I_neural
    sel = NSD_HIGH
    sel_lh = sel[:len(sel)//2]
    sel_rh = sel[len(sel)//2:]
    lh_activity = activity[:, :len(sel)//2][:, sel_lh]
    rh_activity = activity[:, len(sel)//2:][:, sel_rh]
    B = lh_activity.shape[0]
    smoothness = []
    for b in range(B):
        lh_s = compute_morans_I_neural(lh_activity[b], human_lh_adj_list)
        rh_s = compute_morans_I_neural(rh_activity[b], human_rh_adj_list)
        smoothness.append((lh_s + rh_s) / 2.0)
    smoothness = np.array(smoothness)
    return smoothness

def compute_activity_smoothness_model(activity, model):
    import torch
    from libpysal import weights
    from esda.moran import Moran
    model_sheet_dims = model.smoothed_layer_positions[0].dims

    if isinstance(activity, torch.Tensor):
        activity = activity.cpu().numpy()

    B = activity.shape[0]
    model_w = weights.lat2W(*model_sheet_dims[-2:], rook=False)
    smoothness = []
    for b in tqdm(range(B), desc="Computing model smoothness"):
        s = Moran(activity[b], model_w).I
        smoothness.append(s)
    smoothness = np.array(smoothness)
    return smoothness