from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def ensure_dir(path):
    if path is None:
        return None
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def savefig(path, **kwargs):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, **kwargs)
    plt.close()
    return path


def to_numpy(array):
    if torch.is_tensor(array):
        return array.detach().cpu().numpy()
    return np.asarray(array)
