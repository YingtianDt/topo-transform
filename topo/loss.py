from torch import nn

from spacetorch.losses.losses_torch import SpatialCorrelationLossModule


class SpatialCorrelationLoss(nn.Module):
    def __init__(self, num_layers: int, neighborhoods_per_batch: int = 16):
        super().__init__()
        self.layer_losses = nn.ModuleList([
            SpatialCorrelationLossModule(neighborhoods_per_batch=neighborhoods_per_batch)
            for _ in range(num_layers)
        ])
        self.num_layers = num_layers

    def forward(self, layer_features, layer_positions):
        loss = 0
        assert len(layer_features) == len(layer_positions) == self.num_layers
            
        for layer_feature, layer_position, layer_loss in zip(layer_features, layer_positions, self.layer_losses):
            has_time = layer_feature.ndim == 5  # BTCHW

            if has_time:
                B, T, C, H, W = layer_feature.shape
                feature = layer_feature.reshape(B*T, C*H*W)
            else:
                B, C, H, W = layer_feature.shape
                feature = layer_feature.reshape(B, C*H*W)

            loss += layer_loss(
                feature,
                layer_position.coordinates,
                layer_position.neighborhood_indices,
            )

        return loss / self.num_layers