import torch
from torch import nn
import numpy as np


VJEPA_LAYERS = [f'backbone.blocks.{i}' for i in range(24)]
UNIFORMER_LAYERS = [
    *[f'model.blocks1.{i}' for i in range(0,  5)],
    *[f'model.blocks2.{i}' for i in range(0,  8)],
    *[f'model.blocks3.{i}' for i in range(0,  20)],
    *[f'model.blocks4.{i}' for i in range(0,  7)],
]

class FeatureExtractor:
    def __init__(self):
        pass

    def extract_features(self, model, inputs):
        raise NotImplementedError

    def __call__(self, model, inputs):
        model.eval()
        with torch.no_grad():
            features = self.extract_features(model, inputs)
        return features

class LayerFeatureExtractor(FeatureExtractor):
    def __init__(self, layer_names):
        super(LayerFeatureExtractor, self).__init__()
        self.layer_names = layer_names
        self.outputs = {}
        self.hooks = []

    def _get_layer(self, model, layer_name):
        modules = layer_name.split('.')
        layer = model
        for mod in modules:
            if mod.isdigit():
                layer = layer[int(mod)]
            else:
                layer = getattr(layer, mod)
        return layer

    def _hook_fn(self, layer_name):
        def hook(module, input, output):
            self.outputs[layer_name] = output
        return hook

    def register_hooks(self, model):
        for layer_name in self.layer_names:
            layer = self._get_layer(model, layer_name)
            hook = layer.register_forward_hook(self._hook_fn(layer_name))
            self.hooks.append(hook)

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def extract_features(self, model, inputs):
        self.register_hooks(model)
        model(inputs)
        self.remove_hooks()
        features = [self.outputs[name] for name in self.layer_names]
        return features

class VJEPAFeatureExtractor(LayerFeatureExtractor):
    def __init__(self, layer_names=None, layer_indices=None, ret_type='tchw'):
        if layer_names is None:
            if layer_indices is not None:
                layer_names = [VJEPA_LAYERS[i] for i in layer_indices]
            else:
                layer_names = VJEPA_LAYERS

        super().__init__(layer_names)
        self.ret_type = ret_type

    @property
    def layer_dims(self):
        return [(1024, 14, 14) for _ in range(self.num_target_layers)]

    @property
    def num_target_layers(self):
        return len(self.layer_names)

    def extract_features(self, model, inputs):
        if inputs.ndim == 4:
            # img to single-frame video
            inputs = inputs[:,None].repeat(1,2,1,1,1)

        features = super().extract_features(model, inputs)
        features = [self._process_feature(feat) for feat in features]
        return features

    def _process_feature(self, feature):
        # feature: B x THW x C
        B, THW, C = feature.shape
        feature = feature.reshape(B, -1, 14, 14, C)  # B x T x H x W x C
        feature = feature.permute(0, 1, 4, 2, 3)  # B x T x C x H x W
        if self.ret_type == 'tchw':
            return feature
        elif self.ret_type == 'chw':
            feature = feature.mean(dim=1)  # B x C x H x W

