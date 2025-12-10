import config

import os
import torch
from torchvision import transforms

from validate import *
from models import vit_transform

def crop_config_id(ckpt_name):
    ckpt_name = ckpt_name.replace("best_transformed_model_", "")
    ckpt_name = ckpt_name.replace("checkpoint_transformed_model_", "")

    if ckpt_name.startswith("best_transformed_model_"):
        ckpt_name = ckpt_name[len("best_transformed_model_"):-3]
    elif ckpt_name.startswith("checkpoint_transformed_model_"):
        ckpt_name = ckpt_name[len("checkpoint_transformed_model_"):]
        ckpt_name.split("_epoch")[0]
    else:
        ckpt_name = ckpt_name[:-3]
    return ckpt_name


if __name__ == '__main__':
    model_name = "best_transformed_model_global_vjepa_14_18_22_single_neighbInf_kinetics400_lr1e-4_bs32_sd42.pt"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, epoch = load_transformed_model(checkpoint_name=model_name, device=device)

    config_id = crop_config_id(model_name)

    figure_dir = config.CACHE_DIR / "figures" / config_id
    figure_dir.mkdir(parents=True, exist_ok=True)

    viz_params = {
        # 'topk_percent': 1,
    }

    # validate_invertibility(model, vit_transform)
    with model.smoothing_enabled(fwhm_mm=1.0, resolution_mm=1.0):
        validate_floc(
            model, 
            vit_transform, 
            dataset_names=[
                "vpnl", 
                "biomotion", 
                "kanwisher", 
                "pitzalis",
                # "motion", 
                # "temporal",
                # "pitcher",
                # "robert",
            ],
            viz_dir=figure_dir,
            viz_params=viz_params,
            batch_size=32,
            device=device,
            viz_patches=True,
            plot_individual=True,
            plot_aggregate=False,
        )