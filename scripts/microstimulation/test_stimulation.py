
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

from .get_behaviour_decoder import get_decoder
from .get_stimulation_location import get_stimulation_location
from topo.perturbation import MicroStimulation


class _Dataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data, label, _ = self.dataset[idx]
        return data, label

# do no transform and return the last layer features
class Extractor:
    def __call__(self, model, inputs):
        with torch.no_grad():
            layer_features, layer_positions = model(inputs, do_transform=False)  # pre-transform
        return layer_features[-1].mean(dim=1)  # average over time dimension

def _test_stimulation(ckpt_name, dataset_name):
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, _ = load_transformed_model(checkpoint_name=ckpt_name, device=device)
    model.eval()

    decoder = get_decoder(ckpt_name, dataset_name)

    stimulation = MicroStimulation(model)
    perturbation_params = {
        "current_pulse_mA": 50,
        "pulse_rate_Hz": 200,
    }

    batch_size = 16

    if dataset_name == "afraz2006":
        dataset = AFRAZ2006(transforms=vit_transform)
        roi = "face"
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

    val = _Dataset(dataset.valset)
    val_loader = get_data_loader(val, batch_size=batch_size, shuffle=False, num_workers=batch_size)
    noise_levels = dataset.valset.noise_levels()
    
    extractor = Extractor()

    # pre stimulation

    val_features, val_labels = run_features(model, val_loader, extractor, device=device)
    val_pred = decoder.predict(torch.from_numpy(val_features))
    breakpoint()

def test_stimulation(ckpt_name, dataset_name):
    return cached(f"test_stimulation_{dataset_name}_{ckpt_name}")(_test_stimulation)(ckpt_name, dataset_name)


if __name__ == "__main__":
    ckpt_name = MODEL_CKPT
    dataset_name = "afraz2006"
    decoder = test_stimulation(ckpt_name, dataset_name)