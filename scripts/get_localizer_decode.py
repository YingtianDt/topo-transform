import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import config
from data.neural_data.collections.tang2025 import get_compilation
from data.neural_data import Assembly, TemporalAssemblyDataset, get_data_loader
from validate.neural_decoding import decode, make_decoder, run_features
from validate.rois import nsd
from validate import load_transformed_model
from topo import TopoTransformedVJEPA
from models import vit_transform
from collections import defaultdict

from utils import cached
from .common import *
from .get_localizers import localizers, get_localizer_human, get_localizer_model


class Extractor:
    def __init__(self):
        self.do_transform = True

    def activate_transform(self):
        self.do_transform = True

    def deactivate_transform(self):
        self.do_transform = False

    def __call__(self, model, inputs):
        with torch.no_grad():
            layer_features, layer_positions = model(inputs, do_transform=self.do_transform)
        return [lf.mean(dim=1) for lf in layer_features]  # average over time dimension


def _localizer_decode(ckpt_name, rois, num_splits, fwhm_mm, resolution_mm, t_percentage=1, t_threshold=None):
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, epoch = load_transformed_model(checkpoint_name=ckpt_name, device=device)
    model.eval()

    masks_model = get_localizer_model(rois, ckpt_name, t_perc=t_percentage, t_thres=t_threshold, fwhm_mm=fwhm_mm, resolution_mm=resolution_mm)
    masks_human = get_localizer_human(rois)

    batch_size = 16
    ratios = (.9, .1, 0.)
    mask = nsd.get_region_voxels(['high-ventral', 'high-lateral', 'high-dorsal'])
    print("Considering high-level cortex regions only. Voxel counts:", mask.sum())

    decoder = make_decoder(test_type='regress', device=device)

    scores = defaultdict(list)

    with model.smoothing_enabled(fwhm_mm=fwhm_mm, resolution_mm=resolution_mm):
        for split in range(num_splits):
            print(f"=== Split {split+1}/{num_splits} ===")
            seed = 42 + split   
            trainset, testset, _ = get_compilation(vit_transform, ratios=ratios, seed=seed, type='clip')

            train_loader = get_data_loader(trainset, batch_size=batch_size, shuffle=False, num_workers=batch_size)
            test_loader = get_data_loader(testset, batch_size=batch_size, shuffle=False, num_workers=batch_size)
            extractor = Extractor()

            train_model_activities, train_neural_activities = run_features(
                model,
                train_loader,
                get_features=extractor,
                device=device,
            )

            test_model_activities, test_neural_activities = run_features(
                model,
                test_loader,
                get_features=extractor,
                device=device,
            )

            train_model_activities = train_model_activities[0]
            test_model_activities = test_model_activities[0]

            # compute RSA between model and neural activations

            for i, roi, mask_model, mask_human in zip(range(len(rois)), rois, masks_model, masks_human):
                print(f"Processing ROI: {roi}")

                # Model activations for this ROI
                train_model_acts_roi = train_model_activities[:, mask_model[0]]  # shape: (n_samples, n_units)
                test_model_acts_roi = test_model_activities[:, mask_model[0]]  # shape: (n_samples, n_units)

                # Human neural activations for this ROI
                train_neural_acts_roi = train_neural_activities[:, 0, mask_human]  # shape: (n_samples, n_voxels)
                test_neural_acts_roi = test_neural_activities[:, 0, mask_human]  # shape: (n_samples, n_voxels)

                # remove nans
                nans = np.isnan(train_model_acts_roi).any(axis=0)
                train_model_acts_roi = train_model_acts_roi[:, ~nans]
                test_model_acts_roi = test_model_acts_roi[:, ~nans]
                
                decoder = make_decoder('regress', device)

                decode_scores = decode(
                    [[train_model_acts_roi]],
                    [train_neural_acts_roi],
                    [[test_model_acts_roi]],
                    [test_neural_acts_roi],
                    decoder
                )

                mean_score = decode_scores.mean()
                print(f"Decoding score for ROI {roi}: {mean_score:.4f}")

                scores[roi].append(mean_score)
        
    scores = {roi: np.array(vals) for roi, vals in scores.items()}  # Convert lists to arrays
    return scores

def localizer_decode(ckpt_name, rois, num_splits=1, fwhm_mm=2.0, resolution_mm=1.0):
    import hashlib
    rois_code = hashlib.md5('_'.join(sorted(rois)).encode()).hexdigest()[:8]
    return cached(f"localizer_decode_splits{num_splits}_{ckpt_name}_rois{rois_code}_fwhm{fwhm_mm}_res{resolution_mm}")(_localizer_decode)(ckpt_name, rois=rois, num_splits=num_splits, fwhm_mm=fwhm_mm, resolution_mm=resolution_mm)

if __name__ == "__main__":
    # Example usage
    ckpt_name = MODEL_CKPT
    num_splits = 1  # Adjust as needed

    rois = [
        'face',
        'v6',
        'psts',
        'mt',
    ]

    scores  = localizer_decode(ckpt_name, rois, num_splits=num_splits)