
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import config
from data import AFRAZ2006
from data.neural_data import get_data_loader
from validate.rois import nsd
from validate import load_transformed_model
from models import vit_transform

from utils import cached
from scripts.common import *

from .utils import Extractor, _Dataset
from .get_behaviour_decoder import get_decoder
from .get_stimulation_location import *
from topo.perturbation import MicroStimulation
from validate.neural_decoding import run_features, make_decoder


STIMULATION_PARAMETERS = {
    # "Microstimulation consisted of bipolar current pulses of 50mA delivered at 200 Hz (refs 19, 20).
    'current_pulse_mA': 5000,
    'pulse_rate_Hz': 2000,
}

def _test_stimulation(ckpt_name, dataset_name):
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, _ = load_transformed_model(checkpoint_name=ckpt_name, device=device)
    model.eval()

    decoder = get_decoder(ckpt_name, dataset_name)

    stimulation = MicroStimulation(model)
    perturbation_params = STIMULATION_PARAMETERS

    batch_size = 64

    if dataset_name == "afraz2006":
        dataset = AFRAZ2006(transforms=vit_transform).valset
        roi = "face-afraz"
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

    shuffle = False
    assert shuffle == False, "Shuffle must be False for evaluation"
    val = _Dataset(dataset)
    val_loader = get_data_loader(val, batch_size=batch_size, shuffle=shuffle, num_workers=batch_size)
    label_signal_levels = dataset.label_signal_levels()
    
    extractor = Extractor()

    ret = {}

    # pre stimulation
    val_features, val_labels = run_features(model, val_loader, extractor, device=device)
    val_pred = decoder.predict_proba(torch.from_numpy(val_features))
    
    ret['pre_stimulation'] = val_pred[:,1]
    ret['label_signal_levels'] = label_signal_levels

    # post stimulation
    # stimulation_locations, selecitivities = get_selectivity_based_stimulation_locations(roi, ckpt_name, num_samples=50)
    stimulation_locations, sampled_indices = get_random_stimulation_locations(model)

    val_pred_list = []
    for location in stimulation_locations:
        print(location)
        stimulation.perturb(location, perturbation_params)
        val_features, val_labels = run_features(model, val_loader, extractor, device=device)
        val_pred = decoder.predict_proba(torch.from_numpy(val_features))
        val_pred_list.append(val_pred[:,1])
        stimulation.clear()

    ret['post_stimulation'] = val_pred_list
    ret['stimulation_locations'] = stimulation_locations
    ret['sampled_indices'] = sampled_indices
    # ret['selecitivities'] = selecitivities

    return ret


def _test_stimulation(ckpt_name, dataset_name):
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, _ = load_transformed_model(checkpoint_name=ckpt_name, device=device)
    model.eval()

    decoder = get_decoder(ckpt_name, dataset_name)

    stimulation = MicroStimulation(model)
    perturbation_params = STIMULATION_PARAMETERS

    batch_size = 64

    if dataset_name == "afraz2006":
        dataset = AFRAZ2006(transforms=vit_transform).valset
        roi = "face"
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

    shuffle = False
    assert shuffle == False, "Shuffle must be False for evaluation"
    val = _Dataset(dataset)
    val_loader = get_data_loader(val, batch_size=batch_size, shuffle=shuffle, num_workers=batch_size)
    label_signal_levels = dataset.label_signal_levels()
    
    extractor = Extractor()

    ret = {}

    # pre stimulation
    val_features, val_labels = run_features(model, val_loader, extractor, device=device)
    val_features_pre = val_features.copy()
    
    # post stimulation
    stimulation_locations, selecitivities = get_selectivity_based_stimulation_locations(roi, ckpt_name, num_samples=5)

    from scripts.get_localizers import localizers
    t_vals_dict, p_vals_dict, layer_positions = localizers(ckpt_name, dataset_names=['vpnl'], ret_merged=True, resolution_mm=0.0, fwhm_mm=0.0)
    t_vals_f = t_vals_dict['face'][0]
    face_selectivities = t_vals_f.flatten()[-200704:]
    # stimulation_locations = get_random_stimulation_locations(model)

    val_pred_list = []
    for location in stimulation_locations:
        print(location)
        stimulation.perturb(location, perturbation_params)
        val_features, val_labels = run_features(model, val_loader, extractor, device=device)
        val_features_post = val_features.copy()

        # compute difference in features
        feature_diff = val_features_post - val_features_pre
        positions = model.layer_positions[0].coordinates.cpu().numpy()[-200704:]  # (N, 2)

        plt.scatter(positions[:,0], positions[:,1], c=feature_diff.mean(axis=0), cmap='bwr', s=1)
        plt.colorbar(label='Mean Feature Change')
        plt.title(f'Feature Change due to Stimulation at {location}')
        plt.xlabel('X Position (mm)')
        plt.ylabel('Y Position (mm)')
        plt.savefig(f'stimulation_feature_change_{location[0]:.2f}_{location[1]:.2f}.png')
        plt.close()

        # selectivity vs feature change correlation
        from scipy.stats import pearsonr
        mean_feature_change = feature_diff.mean(axis=0).flatten()
        corr, pval = pearsonr(face_selectivities, mean_feature_change)
        print(f"Selectivity vs Feature Change Correlation: r={corr:.4f}, p={pval:.4e}")
        plt.scatter(face_selectivities, mean_feature_change, s=1)
        plt.xlabel('Selectivity')
        plt.ylabel('Mean Feature Change')
        plt.title(f'Selectivity vs Feature Change (r={corr:.2f})')
        plt.savefig(f'stimulation_selectivity_vs_feature_change.png')
        plt.close()

        breakpoint()

    return ret



def test_stimulation(ckpt_name, dataset_name):
    return cached(f"test_stimulation_{dataset_name}_{ckpt_name}", rerun=True)(_test_stimulation)(ckpt_name, dataset_name)


if __name__ == "__main__":
    ckpt_name = MODEL_CKPT
    dataset_name = "afraz2006"
    decoder = test_stimulation(ckpt_name, dataset_name)