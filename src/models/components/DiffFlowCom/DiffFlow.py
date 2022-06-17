from .DiffFlow_base import BaseModel
import torch
import logging
from src import utils

log = utils.get_logger(__name__)


class DiffFlow(BaseModel):
    def __init__(
            self,
            data_shape,
            timestamp,
            condition,
            diffusion,
            score_net: torch.nn.Module,
            drift_net: torch.nn.Module,
    ):
        super().__init__(data_shape, drift_net, score_net)
        # logging.info("******************************Diff Flow")
        timestamp = timestamp()
        condition = condition()
        diffusion = diffusion()
        log.info(timestamp)
        log.info(condition)
        log.info(diffusion)
        self.register_buffer("timestamps", timestamp)
        self.register_buffer("diffusion", diffusion)
        self.register_buffer("condition", condition)
        assert self.timestamps.shape == self.diffusion.shape
        self.register_buffer("delta_t", self.timestamps[1:] - self.timestamps[:-1])

    def forward(self, x):
        return super().forward(x, self.timestamps, self.diffusion, self.condition)

    def backward(self, z):
        return super().backward(z, self.timestamps, self.diffusion, self.condition)

    def sample(self, n_samples):
        z = self._distribution.sample(n_samples).view(-1, *self.data_shape)
        x, _ = self.backward(z)
        return x

    def sample_noise(self, n_samples):
        return self._distribution.sample(n_samples).view(-1, *self.data_shape)

    def noise_log_prob(self, z):
        return self._distribution.log_prob(z)
