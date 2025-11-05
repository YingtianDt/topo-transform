import os
import numpy as np
import torch
from torch import nn

from config import CACHE_DIR
from .positions import create_position_dicts
from .loss import SpatialCorrelationLoss
from spacetorch.models.positions import LayerPositions


POSITION_DIR = CACHE_DIR / "positions"
POSITION_DIR.mkdir(exist_ok=True, parents=True)


# here we just use a heuristic configuration
# in TDANN:
# RETINA 0.0475 / 2.4   = 0.0198  
# V1     1.626  / 35.75 = 0.0455  
# V2     3.977  / 35    = 0.1136  
# V4     2.545  / 22.4  = 0.1136  
# VTC    31.818 / 70.0  = 0.4545
def _get_tissue_configs(num_layers, min_neighborhood_size=0.05, max_neighborhood_size=0.5):
    layer_tissue_sizes = []
    layer_neighborhood_widths = []
    
    base_size = 30
    # Exponential interpolation across layers
    for i in range(num_layers):
        t = i / (num_layers - 1) if num_layers > 1 else 0.0
        size = min_neighborhood_size * (max_neighborhood_size / min_neighborhood_size) ** t * base_size
        layer_tissue_sizes.append(base_size)
        layer_neighborhood_widths.append(size)
    
    return layer_tissue_sizes, layer_neighborhood_widths

class TopoTransformedModel(nn.Module):
    def __init__(self, name, model, extractor, transform, rebuild=False, seed=42):
        super().__init__()

        # Freeze backbone model
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        self.model = model

        self.name = name
        self.extractor = extractor
        self.transform = transform
        self.layer_dims = extractor.layer_dims
        self.layer_names = extractor.layer_names
        self.num_layers = extractor.num_target_layers
        self.layer_config_dir = (POSITION_DIR / name)

        if not self.layer_config_dir.exists() or rebuild:
            print("Generating layer positions...")
            layer_tissue_sizes, layer_neighborhood_widths = _get_tissue_configs(self.num_layers)

            layer_dims = {name: v for name, v in zip(self.layer_names, self.layer_dims)}
            layer_tissue_sizes = {name: v for name, v in zip(self.layer_names, layer_tissue_sizes)}
            layer_neighborhood_widths = {name: v for name, v in zip(self.layer_names, layer_neighborhood_widths)}

            np.random.seed(seed)
            self.layer_positions = create_position_dicts(
                self.layer_names,
                layer_dims,
                layer_tissue_sizes, 
                layer_neighborhood_widths,
                self.layer_config_dir,
            )
        else:
            print("Loading layer positions...")
            self.layer_positions = []
            for layer_name in self.layer_names:
                file_path = self.layer_config_dir / f"{layer_name}.pkl"
                assert file_path.exists()
                layer_position = LayerPositions.load(file_path)
                self.layer_positions.append(layer_position)

        # store as buffers
        for i, layer_position in enumerate(self.layer_positions):
            coords = torch.from_numpy(layer_position.coordinates).float()
            neigh = torch.from_numpy(layer_position.neighborhood_indices).long()

            # Register buffers so they move automatically with .to(device)
            self.register_buffer(f"layer_{i}_coordinates", coords)
            self.register_buffer(f"layer_{i}_neighborhood_indices", neigh)

            # Optionally store reference for convenience
            layer_position.coordinates = getattr(self, f"layer_{i}_coordinates")
            layer_position.neighborhood_indices = getattr(self, f"layer_{i}_neighborhood_indices")

    def forward(self, inputs):
        with torch.no_grad():
            layer_features = self.extractor.extract_features(self.model, inputs)
        layer_features = self.transform(layer_features)
        return layer_features, self.layer_positions


class TopoTransformedVJEPA(TopoTransformedModel):
    def __init__(self, layer_indices=range(4,24,4)):
        from models import VJEPA
        from .features import VJEPAFeatureExtractor
        from .layer import TopoTransform
        name = 'vjepa'
        if list(layer_indices) == list(range(0,24)):
            name += '_full'
        else:
            for layer_index in layer_indices:
                name += f'_{layer_index}'
        model = VJEPA()
        extractor = VJEPAFeatureExtractor(layer_indices=layer_indices)
        transform = TopoTransform(layer_dims=extractor.layer_dims)
        super().__init__(name, model, extractor, transform)