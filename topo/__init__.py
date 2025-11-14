import os
import numpy as np
import torch
from torch import nn

from config import CACHE_DIR
from .positions import create_position_dicts, LayerPositions
from .tissue import _get_tissue_configs_v3, _get_tissue_configs, VJEPA_LAYER_ASSIGNMENTS
from .loss import SpatialCorrelationLoss
from .smoothing import NeuronSmoothing
from spacetorch.models.positions import LayerPositions


POSITION_DIR = CACHE_DIR / "positions"
POSITION_DIR.mkdir(exist_ok=True, parents=True)


class TopoTransformedModel(nn.Module):
    def __init__(self, name, model, extractor, layer_positions, transform, rebuild=False, seed=42):
        super().__init__()

        # Freeze backbone model
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        self.model = model

        self.name = name
        self.transform = transform
        self.set_extractor(extractor)
        self.layer_positions = layer_positions
        self.output_layer_indices = range(self.num_layers)

        # store as buffers
        for i, layer_position in enumerate(self.layer_positions):
            coords = torch.from_numpy(layer_position.coordinates).float()
            neigh = torch.from_numpy(layer_position.neighborhood_indices).long()

            # Register buffers so they move automatically with .to(device)
            self.register_buffer(f"layer_{i}_coordinates", coords)
            self.register_buffer(f"layer_{i}_neighborhood_indices", neigh)

            layer_position.coordinates = coords
            layer_position.neighborhood_indices = neigh

    def set_extractor(self, extractor):
        self.extractor = extractor
        self.layer_names = extractor.layer_names
        self.layer_dims = extractor.layer_dims
        self.num_layers = extractor.num_target_layers

    def set_layer_names(self, layer_names):
        self.output_layer_indices = [self.layer_names.index(name) for name in layer_names]
        self.extractor.set_layer_names(layer_names)
        self.set_extractor(self.extractor)

    @staticmethod
    def smooth(layer_features, layer_positions, fwhm_mm=2, resolution_mm=1):
        smoothing = NeuronSmoothing(fwhm_mm=fwhm_mm, resolution_mm=resolution_mm)
        ret_features = []
        ret_positions = []
        for layer_feat, layer_pos in zip(layer_features, layer_positions):
            positions = layer_pos.coordinates
            features_smoothed, positions_smoothed, grid_dims = smoothing(layer_feat, positions)
            layer_pos_smoothed = LayerPositions(
                name=layer_pos.name,
                dims=grid_dims,
                coordinates=positions_smoothed,
                neighborhood_indices=None,
                neighborhood_width=None,
            )
            ret_features.append(features_smoothed)
            ret_positions.append(layer_pos_smoothed)
        return ret_features, ret_positions 

    def forward(self, inputs, smoothing=False):
        with torch.no_grad():
            layer_features = self.extractor.extract_features(self.model, inputs)
        layer_features = self.transform(layer_features)
        layer_positions = [self.layer_positions[i] for i in self.output_layer_indices]

        if smoothing:
            layer_features, layer_positions = self.smooth(layer_features, layer_positions)

        return layer_features, layer_positions


class TopoTransformedVJEPA(TopoTransformedModel):
    def __init__(self, layer_indices=[14,18,22], smoothing=False, exponentially_interpolate=False, 
                constant_rf_overlap=False, rebuild=False, single_sheet=True, seed=42):
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
        
        if single_sheet:
            name += '_single'

        self.single_sheet = single_sheet
        self.smoothing = smoothing
        
        model = VJEPA()
        extractor = VJEPAFeatureExtractor(layer_indices=layer_indices)
        transform = TopoTransform(layer_dims=extractor.layer_dims)

        layer_config_dir = (POSITION_DIR / name)

        if not layer_config_dir.exists() or rebuild:
            print("Generating layer positions...")
            get_tissue_configs = _get_tissue_configs_v3 if single_sheet else _get_tissue_configs
            layer_tissue_sizes, layer_neighborhood_widths, layer_rf_overlaps = get_tissue_configs(
                layer_indices, layer_assignments=VJEPA_LAYER_ASSIGNMENTS, exponentially_interpolate=exponentially_interpolate, 
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
                save_dir=layer_config_dir,
                single_sheet=single_sheet,
            )
        else:
            print("Loading layer positions...")
            layer_positions = []
            if not single_sheet:
                for layer_name in extractor.layer_names:
                    file_path = layer_config_dir / f"{layer_name}.pkl"
                    assert file_path.exists()
                    layer_position = LayerPositions.load(file_path)
                    layer_positions.append(layer_position)
            else:
                file_path = layer_config_dir / "single_sheet.pkl"
                assert file_path.exists()
                for _ in extractor.layer_names:
                    layer_position = LayerPositions.load(file_path)
                    layer_positions.append(layer_position)

        super().__init__(name, model, extractor, layer_positions, transform, rebuild, seed)

    def forward(self, inputs, smoothing=False):
        layer_features, layer_positions = super().forward(inputs, smoothing=False)
        if self.single_sheet:
            # concatenate features along width
            concatenated_features = []
            for feat in layer_features:
                concatenated_features.append(feat)  # list of (B, C, H, W)
            concatenated_features = torch.cat(concatenated_features, dim=-1)
            layer_features = [concatenated_features]
            layer_positions = layer_positions[:1]

        if smoothing or (self.smoothing and not self.training):
            layer_features, layer_positions = self.smooth(layer_features, layer_positions)

        return layer_features, layer_positions