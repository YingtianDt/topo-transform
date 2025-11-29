from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import config
from data import Kinetics400
from topo import TopoTransformedVJEPA, TopoTransformedTDANN

from utils import cached

def _validate_features(ckpt_name):

    vit_transform = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.Lambda(lambda img: img/255.0),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])  

    is_swapopt = ckpt_name == "swapopt"
    is_tdann = ckpt_name == "tdann"

    # Load data
    data = Kinetics400(train_transforms=vit_transform, test_transforms=vit_transform, fps=8 if is_swapopt else 12)
    val_loader = DataLoader(data.valset, batch_size=32, shuffle=False, num_workers=4)

    # Load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if is_tdann:
        model = TopoTransformedTDANN(seed=0)
    else:
        model = TopoTransformedVJEPA(layer_indices=[14, 18, 22] if not is_swapopt else [18], seed=42, swapopt=is_swapopt, inf_neighborhood=not is_swapopt)

    # Load checkpoint
    if not is_swapopt or not is_tdann:
        checkpoint_path = config.CACHE_DIR / "checkpoints" / ckpt_name
        if checkpoint_path.exists():
            ckpt = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(ckpt['transformed_model_state_dict'])
            print(f"Loaded checkpoint from {checkpoint_path}")

    model = model.to(device)
    model.eval()

    # Extract features
    all_features = []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Extracting features'):
            videos = batch[0].to(device)
            features, positions = model(videos)
            all_features.append([f.cpu() for f in features])
    all_features = [torch.cat([batch[l] for batch in all_features], dim=0) for l in range(len(all_features[0]))]
    return all_features, positions

def validate_features(ckpt_name: str):
    return cached(f"validate_features_{ckpt_name}")(_validate_features)(ckpt_name)