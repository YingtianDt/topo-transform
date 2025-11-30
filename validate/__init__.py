import config

import torch
from topo import TopoTransformedVJEPA, TopoTransformedTDANN
from .autocorr import validate_autocorr
from .floc import *
from .invertible import validate_invertibility


def load_transformed_model(checkpoint_name, device='cuda'):
    """Load a trained TopoTransformedVJEPA model."""

    if "tdann." in checkpoint_name:
        seed = int(checkpoint_name.split("_seed")[-1].split(".")[0])
        model = TopoTransformedTDANN(seed=seed)
        model.name = checkpoint_name
        checkpoint_path = config.CACHE_DIR / "checkpoints" / checkpoint_name.replace("tdann.", "tdann/")
        model.model.load_pretrained_weights(checkpoint_path)
        print(f"Loaded TopoTransformedTDANN model from {checkpoint_path}.")
        epoch = None
    elif checkpoint_name == "swapopt":
    #     model = TopoTransformedVJEPA(layer_indices=[14, 18, 22] if not is_swapopt else [18], seed=42, swapopt=is_swapopt, inf_neighborhood=not is_swapopt)
        layer_indices = [18]
        model = TopoTransformedVJEPA(layer_indices=layer_indices, swapopt=True, inf_neighborhood=False)
        model.name = "swapopt"
        epoch = None
    else:
        layer_indices=[14,18,22]
        checkpoint_path = config.CACHE_DIR / "checkpoints" / checkpoint_name
        model = TopoTransformedVJEPA(layer_indices=layer_indices)
        model.name = checkpoint_name if checkpoint_name is not None else str(checkpoint_path.stem)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        msg = model.load_state_dict(checkpoint['transformed_model_state_dict'], strict=False)
        epoch = checkpoint.get('epoch', None)
        print(f"Loaded TopoTransformedVJEPA model from {checkpoint_path} (epoch {epoch}).")
        print(msg)

    model.to(device)
    
    return model, epoch


if __name__ == "__main__":
    model, epoch = load_transformed_model("tdann/model_final_checkpoint_phase199_seed1.torch", device='cpu')
