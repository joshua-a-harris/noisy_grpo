import os
import logging

logger = logging.getLogger(__name__)


class NoiseModelMixin:
    """Interface for setting and collecting noise from model layer"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.debug = os.environ.get("DEBUG") == "1"
        self.noise_layer = None

    def set_layer_noise(self, layer_idx, noise_scaling_factor):
        self.model.layers[layer_idx].set_noise_mode(True, noise_scaling_factor=noise_scaling_factor)
        self.noise_layer = layer_idx

    def set_current_noise(self, current_noise):
        self.model.layers[self.noise_layer].current_noise = current_noise

    def get_current_noise(self):
        current_noise = self.model.layers[self.noise_layer].current_noise
        return current_noise

