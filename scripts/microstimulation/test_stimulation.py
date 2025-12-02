
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import config
from data import AFRAZ2006
from data.neural_data import get_data_loader
from validate.neural_decoding import run_features, make_decoder
from validate.rois import nsd
from validate import load_transformed_model
from models import vit_transform

from utils import cached
from scripts.common import *

from .utils import Extractor, _Dataset
from .get_behaviour_decoder import get_decoder
from .get_stimulation_location import get_selectivity_based_stimulation_locations
from topo.perturbation import MicroStimulation


STIMULATION_PARAMETERS = {
    # "Microstimulation consisted of bipolar current pulses of 50mA delivered at 200 Hz (refs 19, 20).
    'current_pulse_mA': 500,
    'pulse_rate_Hz': 200,
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
        dataset = AFRAZ2006(transforms=vit_transform)
        roi = "face"
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

    shuffle = False
    assert shuffle == False, "Shuffle must be False for evaluation"
    val = _Dataset(dataset.valset)
    val_loader = get_data_loader(val, batch_size=batch_size, shuffle=shuffle, num_workers=batch_size)
    label_signal_levels = dataset.valset.label_signal_levels()
    
    extractor = Extractor()

    ret = {}

    # pre stimulation
    val_features, val_labels = run_features(model, val_loader, extractor, device=device)
    val_pred = decoder.predict(torch.from_numpy(val_features))
    
    ret['pre_stimulation'] = val_pred
    ret['label_signal_levels'] = label_signal_levels

    # post stimulation
    stimulation_locations, selecitivities = get_selectivity_based_stimulation_locations([roi], ckpt_name)
    stimulation_locations = stimulation_locations[0]  # single ROI
    selecitivities = selecitivities[0]

    val_pred_list = []
    for location in stimulation_locations:
        stimulation.perturb(location, perturbation_params)
        val_features, val_labels = run_features(model, val_loader, extractor, device=device)
        val_pred = decoder.predict(torch.from_numpy(val_features))
        val_pred_list.append(val_pred)
        stimulation.clear()

        break # only do one location for testing purposes

    ret['post_stimulation'] = val_pred_list
    ret['stimulation_locations'] = stimulation_locations
    ret['selecitivities'] = selecitivities

    return ret


def test_stimulation(ckpt_name, dataset_name):
    return cached(f"test_stimulation_{dataset_name}_{ckpt_name}", rerun=True)(_test_stimulation)(ckpt_name, dataset_name)


if __name__ == "__main__":
    ckpt_name = MODEL_CKPT
    dataset_name = "afraz2006"
    decoder = test_stimulation(ckpt_name, dataset_name)