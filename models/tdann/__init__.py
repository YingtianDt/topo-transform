import torch
import re
import torch.nn as nn
from torchvision.models import resnet18


class TDANN(nn.Module):
    def __init__(self, pretrained=True, seed=0) -> None:
        super(TDANN, self).__init__()
        
        self.model = resnet18()

        if pretrained:
            model_path = f"/ccn2/u/ynshah/spacetorchv2/checkpoints/tdann/spatial/model_final_checkpoint_phase199_seed{seed}.torch"
            ckpt = torch.load(model_path, map_location="cpu")
            model_params = ckpt["classy_state_dict"]["base_model"]["model"]["trunk"]

            adjusted_model_params = {}
            pattern = re.compile(r"(\d+)_(\d+)")

            for key, value in model_params.items():
                new_key = pattern.sub(r'\1.\2', key)
                adjusted_model_params[new_key] = value

            adjusted_model_params = {k.replace("base_model.", ""): v for k, v in adjusted_model_params.items()}
            adjusted_model_params = {k.replace("_feature_blocks.", ""): v for k, v in adjusted_model_params.items()}

            msg = self.model.load_state_dict(adjusted_model_params, strict=False)
            print(("Pretrained weights found at {} and loaded with msg: {}".format(model_path, msg)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) == 5:
            # just pick the first frame
            x = x[:, 0, :, :, :]
        return self.model(x)
