"""
Saves initial positions for each layer of a given model, initialized retinotopically
"""
import math
from pathlib import Path
import argparse
from dataclasses import dataclass
import logging
import numpy as np
from typing import Dict, List, Any

from spacetorch.models.positions import (
    LayerPositions,
)
from spacetorch.types import Dims, LayerString, VVSRegion
from spacetorch.utils.spatial_utils import (
    collapse_and_trim_neighborhoods,
    jitter_positions,
    place_conv,
    precompute_neighborhoods,
)

# script constants
POS_VERSION = 3  # increment this every time the position scheme changes

# set up logger
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)-15s %(levelname)s:%(message)s"
)
logger = logging.getLogger(__name__)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    return parser


@dataclass
class LayerPlacement:
    """A layer placement describes the high-level parameters for how units in a given
    layer should be arranged.

    Attributes:
        name: the name of the layer, e.g., "layer4.1"
        tissue_size: total extent of the layer tissue in mm, e.g., 10mm
        dims: expected dimensionality of the outputs in this layer, e.g., (128, 28, 28)
            means that the outputs for this layer come from 128 distinct kernels and
            (28 * 28 = 784) spatial positions
        rf_overlap: The units at each spatial position occupy a square subregion of the tissue.
            This parameter controls how much adjacent subregions overlap
        neighborhood_width: size in mm of a "neighborhood" of units. During training,
            only units in the same neighborhood participate in computation of the spatial
            cost
    """

    name: str
    tissue_size: float
    dims: Dims
    rf_overlap: float
    neighborhood_width: float


def get_placement_configs(
    layers: List[LayerString],
    layer_dims: Dict[LayerString, Dims],
    layer_tissue_sizes: Dict[LayerString, float],
    layer_neighborhood_widths: Dict[LayerString, float], 
    layer_rf_overlaps: Dict[LayerString, float]
) -> List[LayerPlacement]:
    placement_configs: List[LayerPlacement] = [
        LayerPlacement(
            name=layer,
            tissue_size=layer_tissue_sizes[layer],
            dims=layer_dims[layer],
            neighborhood_width=layer_neighborhood_widths[layer],
            rf_overlap=layer_rf_overlaps[layer],
        )
        for layer in layers
    ]

    return placement_configs


def create_position_dict(cfg: LayerPlacement, inf_neighborhood: bool = False) -> Dict[str, Any]:
    positions, rf_radius = place_conv(
        dims=cfg.dims,
        pos_lims=(0, cfg.tissue_size),
        offset_pattern="random",
        rf_overlap=cfg.rf_overlap,
        return_rf_radius=True,
    )

    positions = jitter_positions(positions, jitter=0.3)

    neighborhood_list = precompute_neighborhoods(
        positions, radius=cfg.neighborhood_width / 2, n_neighborhoods=20_000,
        inf_neighborhood=inf_neighborhood
    )

    neighborhoods = collapse_and_trim_neighborhoods(
        neighborhood_list, keep_fraction=0.95, keep_limit=500, target_shape=None
    )

    return {
        "positions": positions,
        "neighborhoods": neighborhoods,
        "radius": cfg.neighborhood_width / 2,
        "dims": cfg.dims,
    }

def create_position_dict_single_sheet(cfgs: List[LayerPlacement]) -> Dict[str, Any]:
    print(cfgs)

    C, H, W = cfgs[0].dims
    overlap = cfgs[0].rf_overlap
    for cfg in cfgs[1:]:
        assert cfg.dims == (C, H, W), "All layers must have the same spatial dimensions for single sheet arrangement"
        assert math.isclose(cfg.rf_overlap, overlap), "All layers must have the same rf_overlap for single sheet arrangement"
    num_layers = len(cfgs)

    position_lists = []
    for i, cfg in enumerate(cfgs):
        positions, rf_radius = place_conv(
            dims=cfg.dims,
            pos_lims=(0, cfg.tissue_size),
            offset_pattern="random",
            rf_overlap=cfg.rf_overlap,
            return_rf_radius=True,
        )
        positions[:, 0] += i * cfg.tissue_size
        position_lists.append(positions)

    positions = np.concatenate(position_lists, axis=0)
    positions = jitter_positions(positions, jitter=0.3)

    neighborhood_list = precompute_neighborhoods(
        positions, radius=cfg.neighborhood_width / 2, n_neighborhoods=20_000
    )

    neighborhoods = collapse_and_trim_neighborhoods(
        neighborhood_list, keep_fraction=0.95, keep_limit=500, target_shape=None
    )

    return {
        "positions": positions,
        "neighborhoods": neighborhoods,
        "radius": cfg.neighborhood_width / 2,
        "dims": (C, H, W * num_layers),
    }

def create_position_dicts(
    layers: List[LayerString],
    layer_dims: Dict[LayerString, Dims],
    layer_tissue_sizes: Dict[LayerString, float],
    layer_neighborhood_widths: Dict[LayerString, float], 
    layer_rf_overlaps: Dict[LayerString, float],
    single_sheet: bool = False,
    inf_neighborhood: bool = False,
    save_dir = None,
):

    placement_configs = get_placement_configs(
        layers,
        layer_dims, 
        layer_tissue_sizes,
        layer_neighborhood_widths,
        layer_rf_overlaps,
    )

    if not single_sheet:
        # save each placement config
        ret = []
        for cfg in placement_configs:
            position_dict = create_position_dict(cfg, inf_neighborhood=inf_neighborhood)
            layer_positions = LayerPositions(
                name=cfg.name,
                dims=cfg.dims,
                coordinates=position_dict["positions"],
                neighborhood_indices=position_dict["neighborhoods"],
                neighborhood_width=position_dict["radius"] * 2,
            )
            if save_dir is not None:
                save_dir.mkdir(exist_ok=True, parents=True)
                logger.info(f"Saving to {save_dir}: {cfg.name}.pkl")
                layer_positions.save(save_dir)
            
            ret.append(layer_positions)

        if save_dir is not None:
            version_file = save_dir / "version.txt"
            with open(version_file, "w") as stream:
                stream.write(str(POS_VERSION))

    else:
        position_dict = create_position_dict_single_sheet(placement_configs)
        layer_positions = LayerPositions(
            name="single_sheet",
            dims=position_dict["dims"],
            coordinates=position_dict["positions"],
            neighborhood_indices=position_dict["neighborhoods"],
            neighborhood_width=position_dict["radius"] * 2,
        )
        if save_dir is not None:
            save_dir.mkdir(exist_ok=True, parents=True)
            logger.info(f"Saving to {save_dir}: single_sheet.pkl")
            layer_positions.save(save_dir)

        print(layer_positions)

        ret = [layer_positions]

    return ret


if __name__ == "__main__":
    # ret = create_position_dicts(
    #     ["layer1", "layer2", "layer3"],
    #     {"layer1": (1024, 14, 14), "layer2": (1024, 14, 14), "layer3": (1024, 14, 14)},
    #     {"layer1": 70, "layer2": 70, "layer3": 70},
    #     {"layer1": 31.818, "layer2": 31.818, "layer3": 31.818},
    #     {"layer1": 0.1, "layer2": 0.1, "layer3": 0.1},
    #     single_sheet=True,
    #     save_dir=Path("/ccn2/u/ynshah/tdann-transform/"),
    # )

    import numpy as np
    from pathlib import Path
    import pickle


    x = Path("/ccn2/u/ynshah/tdann-transform/single_sheet.pkl")

    with x.open("rb") as stream:
        y = pickle.load(stream)

    d = {
        "name": y.name,
        "dims": y.dims,
        "coordinates": y.coordinates,
        "neighborhood_indices": y.neighborhood_indices,
        "neighborhood_width": y.neighborhood_width,
    }

    np.savez("/ccn2/u/ynshah/tdann-transform/single_sheet.npz", **d)