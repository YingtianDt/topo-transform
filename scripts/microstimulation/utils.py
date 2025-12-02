import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import config


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
            last_layer_features = model.model(inputs)  # vjepa forward pass
            return last_layer_features