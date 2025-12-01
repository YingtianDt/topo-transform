
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

def _get_decoder(ckpt_name, dataset_name):
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, _ = load_transformed_model(checkpoint_name=ckpt_name, device=device)
    model.eval()

    batch_size = 16

    if dataset_name == "afraz2006":
        dataset = AFRAZ2006(transforms=vit_transform)
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

    tr = _Dataset(dataset.trainset)
    train_loader = get_data_loader(tr, batch_size=batch_size, shuffle=True, num_workers=batch_size)
    
    extractor = Extractor()
    train_features, train_labels = run_features(model, train_loader, extractor, device=device)
    decoder = make_decoder(test_type='classify', device=device)
    decoder.fit(train_features, train_labels)
    print(f"Mode {mode}: {validated_score}")

    breakpoint()


def get_decoder(ckpt_name, dataset_name):
    return cached(f"get_decoder_{dataset_name}_{ckpt_name}")(_get_decoder)(ckpt_name, dataset_name)


if __name__ == "__main__":
    ckpt_name = MODEL_CKPT
    dataset_name = "afraz2006"
    decoder = get_decoder(ckpt_name, dataset_name)