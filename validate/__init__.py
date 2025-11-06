import config

import torch
from topo import TopoTransformedVJEPA
from .autocorr import validate_autocorr
from .floc import *

def load_transformed_model(checkpoint_name=None, checkpoint_path=None, device='cuda'):
    """Load a trained TopoTransformedVJEPA model."""
    assert checkpoint_path is not None or checkpoint_name is not None, \
        "Either checkpoint_name or checkpoint_path must be provided."
    if checkpoint_path is None:
        checkpoint_path = config.CACHE_DIR / "checkpoints" / checkpoint_name
    model = TopoTransformedVJEPA()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['transformed_model_state_dict'])
    model.to(device)
    epoch = checkpoint.get('epoch', None)
    return model, epoch


if __name__ == "__main__":
    ckpt_pth = config.CACHE_DIR / "checkpoints" / "best_transformed_model_vjepa_4_8_12_16_20_lr0p1_bs32.pt"
    model, epoch = load_transformed_model(checkpoint_path=ckpt_pth, device='cpu')
