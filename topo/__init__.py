import os
import numpy as np
import torch
from torch import nn
from contextlib import contextmanager

from config import CACHE_DIR, POSITION_DIR
from .positions import create_position_dicts, LayerPositions
from .tissue import _get_tissue_configs_v3, _get_tissue_configs, VJEPA_LAYER_ASSIGNMENTS
from .loss import *
from .smoothing import NeuronSmoothing
from spacetorch.models.positions import LayerPositions


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
        self.output_layer_indices = range(self.num_layers)
        self.layer_positions = layer_positions

        # smoothing
        self.smoothing = False
        self.fwhm_mm = 2
        self.resolution_mm = 1

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

    def smooth(self, layer_features, layer_positions):
        smoothing = NeuronSmoothing(fwhm_mm=self.fwhm_mm, resolution_mm=self.resolution_mm)
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

    @contextmanager
    def smoothing_enabled(self, fwhm_mm=None, resolution_mm=None):
        """Context manager to temporarily enable smoothing."""
        original_smoothing = self.smoothing
        old_fwhm = self.fwhm_mm
        old_resolution = self.resolution_mm
        try:
            if resolution_mm == 0.0:
                self.smoothing = False
                self.smoothed_layer_positions = self.layer_positions
                self.fwhm_mm = None
                self.resolution_mm = None
            else:
                self.smoothing = True
                if fwhm_mm is not None:
                    self.fwhm_mm = fwhm_mm
                if resolution_mm is not None:
                    self.resolution_mm = resolution_mm

                self.smoothed_layer_positions = []
                for layer_position in self.layer_positions:
                    position_smoothed, grid_dims = NeuronSmoothing.get_grid_positions(layer_position.coordinates, resolution_mm=resolution_mm)
                    self.smoothed_layer_positions.append(
                        LayerPositions(
                            name=layer_position.name,
                            dims=grid_dims,
                            coordinates=position_smoothed,
                            neighborhood_indices=None,  # to be computed later if needed
                            neighborhood_width=layer_position.neighborhood_width,
                        )
                    )

                    if hasattr(self, "single_sheet") and self.single_sheet:
                        self.smoothed_layer_positions = self.smoothed_layer_positions[:1]

            yield self
        finally:
            self.smoothing = original_smoothing
            self.fwhm_mm = old_fwhm
            self.resolution_mm = old_resolution
            del self.smoothed_layer_positions

    def forward(self, inputs, do_transform=True):
        with torch.no_grad():
            layer_features = self.extractor.extract_features(self.model, inputs)
        if do_transform:
            layer_features = self.transform(layer_features)
        layer_positions = [self.layer_positions[i] for i in self.output_layer_indices]

        if self.smoothing:
            layer_features, layer_positions = self.smooth(layer_features, layer_positions)

        return layer_features, layer_positions


class TopoTransformedVJEPA(TopoTransformedModel):
    def __init__(self, layer_indices=[14,18,22], smoothing=False, exponentially_interpolate=False, 
                constant_rf_overlap=False, rebuild=False, single_sheet=True, large_neighborhood=False, inf_neighborhood=True, seed=42, swapopt=False):
        from models import VJEPA, VJEPASwapopt
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

        if large_neighborhood:
            name += '_neighbL'

        if inf_neighborhood:
            name += '_neighbInf'

        self.single_sheet = single_sheet
        self.smoothing = smoothing
        
        self.swapopt = swapopt
        model = VJEPA() if not swapopt else VJEPASwapopt()
        extractor = VJEPAFeatureExtractor(layer_indices=layer_indices)
        transform = TopoTransform(layer_dims=extractor.layer_dims)

        layer_config_dir = (POSITION_DIR / name)
        print(layer_config_dir)

        if not layer_config_dir.exists() or rebuild:
            print("Generating layer positions...")
            get_tissue_configs = _get_tissue_configs_v3 if single_sheet else _get_tissue_configs
            layer_tissue_sizes, layer_neighborhood_widths, layer_rf_overlaps = get_tissue_configs(
                layer_indices, layer_assignments=VJEPA_LAYER_ASSIGNMENTS, exponentially_interpolate=exponentially_interpolate, 
                constant_rf_overlap=constant_rf_overlap, large_neighborhood=large_neighborhood, inf_neighborhood=inf_neighborhood
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
                inf_neighborhood=inf_neighborhood,
            )
        else:
            print("Loading layer positions...")
            layer_positions = []
            if not single_sheet:
                for layer_name in extractor.layer_names:
                    file_path = layer_config_dir / f"{layer_name}.npz"
                    assert file_path.exists()
                    layer_position = LayerPositions.load(file_path)
                    layer_positions.append(layer_position)
            else:
                if swapopt:
                    file_path = layer_config_dir / "backbone.blocks.14.attn.npz"
                else:
                    file_path = layer_config_dir / "single_sheet.pkl"
                assert file_path.exists()
                for _ in extractor.layer_names:
                    layer_position = LayerPositions.load(file_path)
                    layer_positions.append(layer_position)

        super().__init__(name, model, extractor, layer_positions, transform, rebuild, seed)

    def forward(self, inputs, do_transform=True):
        with torch.no_grad():
            layer_features = self.extractor.extract_features(self.model, inputs)

        if do_transform and not self.swapopt:
            layer_features = self.transform(layer_features)

        if self.single_sheet:
            # concatenate features along width
            concatenated_features = []
            for feat in layer_features:
                concatenated_features.append(feat)  # list of (B, C, H, W)
            concatenated_features = torch.cat(concatenated_features, dim=-1)
            layer_features = [concatenated_features]
            layer_positions = self.layer_positions
        else:
            layer_features = [self.layer_features[i] for i in self.output_layer_indices]

        if self.smoothing:
            layer_features, layer_positions = self.smooth(layer_features, layer_positions)

        return layer_features, layer_positions
    

class TopoTransformedTDANN(TopoTransformedModel):
    def __init__(self, seed=0):
        from models import TDANN
        from .features import TDANNFeatureExtractor
        
        model = TDANN()
        extractor = TDANNFeatureExtractor()

        layer_config_dir = (POSITION_DIR / name)
        print(layer_config_dir)

        print("Loading layer positions...")
        layer_positions = []
        file_path = layer_config_dir / "single_sheet.npz"
        assert file_path.exists()
        for _ in extractor.layer_names:
            layer_position = LayerPositions.load(file_path)
            layer_positions.append(layer_position)

        super().__init__(name, model, extractor, layer_positions, transform=None, rebuild=None, seed=42)

    def forward(self, inputs, do_transform=True):
        with torch.no_grad():
            layer_features = self.extractor.extract_features(self.model, inputs)

        if do_transform:
            pass

        # concatenate features along width
        concatenated_features = []
        for feat in layer_features:
            concatenated_features.append(feat)  # list of (B, C, H, W)
        concatenated_features = torch.cat(concatenated_features, dim=-1)
        layer_features = [concatenated_features]
        layer_positions = self.layer_positions

        if self.smoothing:
            layer_features, layer_positions = self.smooth(layer_features, layer_positions)

        return layer_features, layer_positions