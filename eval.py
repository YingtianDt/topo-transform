import config

import os
import torch
from torchvision import transforms

from validate import load_transformed_model, validate_autocorr, validate_floc
from validate.floc import VPNL, KANWISHER


vit_transform = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.Lambda(lambda img: img/255.0),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

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
    model_name = "best_transformed_model_vjepa_4_8_12_16_20_lr1e-4_bs32.pt"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, epoch = load_transformed_model(checkpoint_name=model_name, device=device)
    config_id = crop_config_id(model_name)

    figure_dir = config.CACHE_DIR / "figures" / config_id
    figure_dir.mkdir(parents=True, exist_ok=True)

    validate_floc(model, vit_transform, data_path=VPNL, viz_dir=figure_dir, epoch=epoch)
    validate_floc(model, vit_transform, data_path=KANWISHER, viz_dir=figure_dir, epoch=epoch)