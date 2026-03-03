import libpysal as lp
from .utils import *
from validate.floc import *
from tqdm import tqdm

from validate.rois.nsd import get_region_voxels
NSD_HIGH = get_region_voxels(
    ["high-ventral", "high-lateral", "high-dorsal"]
)
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

    # # plot example
    # example_cate = 'Scenes_static'
    # example_model_t_vals = model_t_val_dict[example_cate].flatten()
    # example_human_t_vals = human_t_val_dict[example_cate]
    # example_human_t_vals[~NSD_HIGH] = np.nan 
    # with model.smoothing_enabled(fwhm_mm=fwhm_mm, resolution_mm=resolution_mm):
    #     positions = model.smoothed_layer_positions[0].coordinates.numpy()

    # import matplotlib.pyplot as plt
    # from nilearn import plotting, surface, datasets
    
    # # plot model
    # model_v_max_abs = np.max(np.abs(example_model_t_vals))
    # plt.scatter(
    #     positions[:, 0], positions[:, 1], c=example_model_t_vals, cmap='bwr', s=5,
    #     marker='s', vmax=model_v_max_abs, vmin=-model_v_max_abs
    # )
    # plt.axis('equal')
    # plt.colorbar(label='t-value')
    # plt.title(f'Model t-values for {example_cate}')
    # plt.savefig('model_example_tvals.png', dpi=400)
    # plt.close()

    # # plot human
    # human_v_max_abs = np.nanmax(np.abs(example_human_t_vals))
    # example_fsaverage = datasets.fetch_surf_fsaverage('fsaverage5')
    # plotting.plot_surf_stat_map(
    #     example_fsaverage.flat_left,
    #     example_human_t_vals[:len(example_human_t_vals)//2],
    #     hemi='left',
    #     title=f'Human t-values for {example_cate} (LH)',
    #     colorbar=True,
    #     cmap='bwr',
    #     view='dorsal',
    #     vmin=-human_v_max_abs, vmax=human_v_max_abs,
    # )

    # plt.savefig('human_example_tvals_lh.png', dpi=400)
    # plt.close()

    # # plot human
    # human_v_max_abs = np.nanmax(np.abs(example_human_t_vals))
    # example_fsaverage = datasets.fetch_surf_fsaverage('fsaverage5')
    # plotting.plot_surf_stat_map(
    #     example_fsaverage.infl_left,
    #     example_human_t_vals[:len(example_human_t_vals)//2],
    #     hemi='left',
    #     title=f'Human t-values for {example_cate} (LH)',
    #     colorbar=True,
    #     cmap='bwr',
    #     view='lateral',
    #     vmin=-human_v_max_abs, vmax=human_v_max_abs,
    # )

    # plt.savefig('human_example_tvals_lh2.png', dpi=400)
    # plt.close()

    # breakpoint()


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

        model_smoothness = _moran_ignore_nans(model_t_vals, model_w)
        human_lh_smoothness = compute_morans_I_neural(human_lh_t_vals, human_lh_adj_list)
        human_rh_smoothness = compute_morans_I_neural(human_rh_t_vals, human_rh_adj_list)
        human_smoothness = (human_lh_smoothness + human_rh_smoothness) / 2.0

        print(f"Cate: {cate}, Model smoothness: {model_smoothness:.4f}, Human smoothness: {human_smoothness:.4f}")

        smoothness_results[cate] = {
            'model_smoothness': model_smoothness,
            'human_smoothness': human_smoothness,
        }

    return smoothness_results

def _moran_ignore_nans(values, w):
    mask = np.isfinite(values)
    if mask.all():
        return Moran(values, w).I
    kept = np.flatnonzero(mask)
    if kept.size < 2:
        return np.nan
    old_to_new = {int(old): new for new, old in enumerate(kept)}
    neighbors = {}
    weights = {}
    for old_idx in kept:
        old_idx = int(old_idx)
        nbrs = []
        wts = []
        for nbr, wt in zip(w.neighbors.get(old_idx, []), w.weights.get(old_idx, [])):
            if nbr in old_to_new:
                nbrs.append(old_to_new[nbr])
                wts.append(wt)
        neighbors[old_to_new[old_idx]] = nbrs
        weights[old_to_new[old_idx]] = wts
    w_sub = lp.weights.W(neighbors, weights, silence_warnings=True)
    return Moran(values[mask], w_sub).I

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
    model_w = weights.lat2W(*model_sheet_dims[-2:])
    smoothness = []
    for b in tqdm(range(B), desc="Computing model smoothness"):
        s = _moran_ignore_nans(activity[b], model_w)
        smoothness.append(s)
    smoothness = np.array(smoothness)
    return smoothness
