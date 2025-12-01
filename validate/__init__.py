import config

import torch
from topo import TopoTransformedVJEPA
from .autocorr import validate_autocorr
from .floc import *
from .invertible import validate_invertibility


def load_transformed_model(layer_indices=[14,18,22], checkpoint_name=None, checkpoint_path=None, device='cuda'):
    """Load a trained TopoTransformedVJEPA model."""
    assert checkpoint_path is not None or checkpoint_name is not None, \
        "Either checkpoint_name or checkpoint_path must be provided."
    if checkpoint_path is None:
        checkpoint_path = config.CACHE_DIR / "checkpoints" / checkpoint_name
    model = TopoTransformedVJEPA(layer_indices=[18], inf_neighborhood=False)
    model.name = checkpoint_name if checkpoint_name is not None else str(checkpoint_path.stem)
    model.to(device)
    return model, 300


if __name__ == "__main__":
    ckpt_pth = config.CACHE_DIR / "checkpoints" / "best_transformed_model_vjepa_4_8_12_16_20_lr0p1_bs32.pt"
    model, epoch = load_transformed_model(checkpoint_path=ckpt_pth, device='cpu')
