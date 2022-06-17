import torch
from pytorch_lightning import LightningModule
import logging
from typing import Any, List


class DiffFlowModule(LightningModule):
    def __init__(
            self,
            net: torch.nn.Module,
            # data_shape,
            # timestamp,
            # diffusion,
            # condition,
            # drift_net,
            # score_net,
            # lr: float = 0.001,
            # weight_decay: float = 0.0005,
    ):
        super().__init__()
        self.net = net
        # self.timestap = timestamp()
        # self.condition = condition()
        # self.diffusion = diffusion()
        # logging.info(self.timestap)
        # logging.info(self.condition)
        # logging.info(self.diffusion)
        # self.net = net
        # logging.info(net)

    def forward(self, x: torch.Tensor):
        logging.info('-' * 10 + 'DiffFlowModule forward' + '-' * 10)
        return self.net(x)

    def training_step(self, batch: Any, batch_idx: int):
        return batch

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

