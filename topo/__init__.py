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


# Here we just use a heuristic configuration
# 
# in TDANN:
# RETINA 0.0475 / 2.4   = 0.0198  
# V1     1.626  / 35.75 = 0.0455  
# V2     3.977  / 35    = 0.1136  
# V4     2.545  / 22.4  = 0.1136  
# VTC    31.818 / 70.0  = 0.4545
#
# with Yash's electrode data alignment:
# blocks.[0,1] = retina
# blocks.[2,3,4,5] = V1
# blocks.[6,7] = V2
# blocks.[8,9,10,11,12,13] = V4v/d
# blocks.[14,15,16,17,18,19,20,21,22,23] = higher visual cortex

NEIGHBORHOOD_WIDTHS = {
    "retina": 0.0475,
    "V1": 1.626,
    "V2": 3.977,
    "V4": 2.545,
    "VTC": 31.818,
}
TISSUE_SIZES = {
    "retina": 2.4,
    "V1": 35.75,
    "V2": 35.0,
    "V4": 22.4,
    "VTC": 70.0,
}
VJEPA_LAYER_ASSIGNMENTS = {
    "retina": [0, 1],
    "V1": [2, 3, 4, 5],
    "V2": [6, 7],
    "V4": [8, 9, 10, 11, 12, 13],
    "VTC": [14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
}
INITIAL_RF_OVERLAP = 0.1

def _get_tissue_configs(layer_indices, layer_assignments, exponentially_interpolate=False, constant_rf_overlap=False):
    layer_tissue_sizes = []
    layer_neighborhood_widths = []
    layer_rf_overlaps = []

    num_stages = len(layer_assignments)
    areas = list(layer_assignments.keys())
    
    for s in range(num_stages):

        if s == 0:
            tissue_size_start = TISSUE_SIZES['retina']
            neighborhood_start = NEIGHBORHOOD_WIDTHS['retina']
            rf_overlap_start = INITIAL_RF_OVERLAP / 2
        else:
            area_start = areas[s - 1]
            tissue_size_start = TISSUE_SIZES[area_start]
            neighborhood_start = NEIGHBORHOOD_WIDTHS[area_start]
            rf_overlap_start = rf_overlap

        area_end = areas[s]
        layers = layer_assignments[area_end]
        tissue_size_end = TISSUE_SIZES[area_end]
        neighborhood_end = NEIGHBORHOOD_WIDTHS[area_end]
        num_layers = len(layers)
        rf_overlap_end = min(rf_overlap_start * 2, 1) if not constant_rf_overlap else rf_overlap_start

        rf_overlap = rf_overlap_end

        if exponentially_interpolate:
            tissue_sizes = [tissue_size_end] * num_layers
            neighborhood_widths = _exponentially_interpolate(neighborhood_start, neighborhood_end, num_layers)
            rf_overlaps = _exponentially_interpolate(rf_overlap_start, rf_overlap_end, num_layers)
        else:
            tissue_sizes = [tissue_size_end] * num_layers
            neighborhood_widths = [neighborhood_end] * num_layers
            rf_overlaps = [rf_overlap_end] * num_layers

        layer_tissue_sizes.extend(tissue_sizes)
        layer_neighborhood_widths.extend(neighborhood_widths)
        layer_rf_overlaps.extend(rf_overlaps)
    
    layer_tissue_sizes = [layer_tissue_sizes[i] for i in layer_indices]
    layer_neighborhood_widths = [layer_neighborhood_widths[i] for i in layer_indices]
    layer_rf_overlaps = [layer_rf_overlaps[i] for i in layer_indices]
    
    print("Layer tissue sizes:", layer_tissue_sizes)
    print("Layer neighborhood widths:", layer_neighborhood_widths)
    print("Layer RF overlaps:", layer_rf_overlaps)

    return layer_tissue_sizes, layer_neighborhood_widths, layer_rf_overlaps


def _exponentially_interpolate(start, end, num_points, lower_bound=0.01):
    if num_points == 1:
        return [start]
    if start == 0:
        start = lower_bound
    if end == 0:
        end = lower_bound

    return [start * (end / start) ** ((i+1) / (num_points)) for i in range(num_points)]


class TopoTransformedModel(nn.Module):
    def __init__(self, name, model, extractor, layer_positions, transform, rebuild=False, seed=42):
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
        self.layer_positions = layer_positions

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
    def __init__(self, layer_indices=range(8,24), exponentially_interpolate=False, 
                constant_rf_overlap=False, rebuild=False, seed=42):
        from models import VJEPA
        from .features import VJEPAFeatureExtractor
        from .layer import TopoTransform
        
        name = 'vjepa'
        if list(layer_indices) == list(range(0,24)):
            name += '_full'
        else:
            for layer_index in layer_indices:
                name += f'_{layer_index}'

        if exponentially_interpolate:
            name += '_interp'

        if constant_rf_overlap:
            name += '_constRF'
        
        model = VJEPA()
        extractor = VJEPAFeatureExtractor(layer_indices=layer_indices)
        transform = TopoTransform(layer_dims=extractor.layer_dims)

        layer_config_dir = (POSITION_DIR / name)

        if not layer_config_dir.exists() or rebuild:
            print("Generating layer positions...")
            layer_tissue_sizes, layer_neighborhood_widths, layer_rf_overlaps = _get_tissue_configs(
                layer_indices, VJEPA_LAYER_ASSIGNMENTS, exponentially_interpolate=exponentially_interpolate, 
                constant_rf_overlap=constant_rf_overlap
            )

            layer_dims = {name: v for name, v in zip(extractor.layer_names, extractor.layer_dims)}
            layer_tissue_sizes = {name: v for name, v in zip(extractor.layer_names, layer_tissue_sizes)}
            layer_neighborhood_widths = {name: v for name, v in zip(extractor.layer_names, layer_neighborhood_widths)}
            layer_rf_overlaps = {name: v for name, v in zip(extractor.layer_names, layer_rf_overlaps)}

            np.random.seed(seed)
            layer_positions = create_position_dicts(
                extractor.layer_names,
                layer_dims,
                layer_tissue_sizes, 
                layer_neighborhood_widths,
                layer_rf_overlaps,
                layer_config_dir,
            )
        else:
            print("Loading layer positions...")
            layer_positions = []
            for layer_name in extractor.layer_names:
                file_path = layer_config_dir / f"{layer_name}.pkl"
                assert file_path.exists()
                layer_position = LayerPositions.load(file_path)
                layer_positions.append(layer_position)

        super().__init__(name, model, extractor, layer_positions, transform, rebuild, seed)