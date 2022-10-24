from typing import Any, List

import numpy as np
import torch
import wandb
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric
from torchmetrics.classification.accuracy import Accuracy
from gluonts.torch.util import copy_parameters
from src import utils
import hydra
import omegaconf
import pyrootutils
import ipdb
import time

log = utils.get_pylogger(__name__)


class GestureTimeGradLightingModule(LightningModule):
    """Example of LightningModule for MNIST classification.

    A LightningModule organizes your PyTorch code into 6 sections:
        - Computations (init).
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Prediction Loop (predict_step)
        - Optimizers and LR Schedulers (configure_optimizers)

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
            self,
            train_net: torch.nn.Module,
            prediction_net: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            is_predict_stage=False
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.is_predict_stage = False
        self.save_hyperparameters(logger=True, ignore=["train_net"])
        self.save_hyperparameters(logger=True, ignore=["prediction_net"])

        self.train_net = train_net
        self.prediction_net = prediction_net
        self.train_step_count = 1
        self.train_time_list = []
        self.generation_time_list = []
        self.train_init_time = 0
        self.train_elapsed_time = 0
        self.generation_init_time = 0
        self.generation_elapsed_time = 0

    def forward(self, x: torch.Tensor, cond: torch.Tensor):
        trainer = self.trainer
        return self.train_net(trainer, x, cond)

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn'timestep store accuracy from these checks
        # self.val_acc_best.reset()
        pass

    def on_train_epoch_start(self):
        self.train_init_time = time.time()

    def train_step(self, batch: Any):
        x = batch["x"]  # the output of body pose corresponding to the condition [80,95,45]
        cond = batch["cond"]  # [80,95,927]
        likelihoods, mean_loss = self.forward(x, cond)
        return likelihoods, mean_loss

    def training_step(self, batch: Any, batch_idx: int):
        # log.info(f"-------------------training_step----------------")
        likelihoods, loss = self.train_step(batch)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        # log.info(f"train_step count: {self.train_step_count}  train mean_loss: {loss}")
        self.train_step_count += 1
        return loss
        # return {"likelihoods": likelihoods, "mean_loss": mean_loss}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        # self.train_acc.reset()
        self.train_step_count = 0
        self.train_elapsed_time = time.time() - self.train_init_time
        self.log("TrainingTime", self.train_elapsed_time, on_step=False, on_epoch=True, prog_bar=True)
        self.train_time_list.append(self.train_elapsed_time)
        # log.info(f"training Time: {self.train_time_list}")

    def on_train_end(self) -> None:
        mean_train_time = np.mean(self.train_time_list)
        std_train_time = np.std(self.train_time_list)
        log.info(f"Mean training epoch time : {mean_train_time}")
        log.info(f"Std training epoch time : {std_train_time}")
        # self.log("mean_train_time", float(mean_train_time), on_step=False, on_epoch=True, prog_bar=True)
        # self.log("std_train_time", float(std_train_time), on_step=False, on_epoch=True, prog_bar=True)

    def validation_step(self, batch: Any, batch_idx: int):
        # log.info(f"-------------------validation_step----------------")
        likelihoods, loss = self.train_step(batch)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        # log.info(f"val mean_loss: {loss}")
        return {"loss": loss, "likelihoods": likelihoods}

    def validation_epoch_end(self, outputs: List[Any]):
        pass

    def on_test_start(self):
        # log.info('-----------------on_test_start--------------')
        copy_parameters(self.train_net, self.prediction_net)
        self.is_predict_stage = False
        # for name, train_net_para in self.train_net.named_parameters():
        #     log.info(f'train_net_para: {name}:\n {train_net_para.data}')
        # log.info('\n')
        # for name, prediction_net_para in self.prediction_net.named_parameters():
        #     # log.info(f'prediction_net_para: {name}:\n {prediction_net_para.data}')
        #     log.info(f'prediction_net_para: {name}:\n {prediction_net_para.data}')

    def on_test_epoch_start(self) -> None:
        self.generation_init_time = time.time()

    def test_step(self, batch: Any, batch_idx: int):
        # ipdb.set_trace()
        autoreg_all = batch["autoreg"].cuda()  # [20, 400, 45]
        # log.info(f'test_step -> autoreg_all shape: {autoreg_all.shape} \n {autoreg_all}')
        control_all = batch["control"].cuda()  # [80,400,27]
        trainer = self.trainer
        output, generation_elapse_time_list = self.prediction_net.forward(autoreg_all, control_all, trainer)
        mean_generation_time = torch.mean(torch.tensor(generation_elapse_time_list))
        std_generation_time = torch.std(torch.tensor(generation_elapse_time_list))
        self.log("mean_generation_time", mean_generation_time, on_step=False, on_epoch=True, prog_bar=True)
        self.log("std_generation_time", std_generation_time, on_step=False, on_epoch=True, prog_bar=True)
        return output

    def test_epoch_end(self, outputs: List[Any]):
        # self.test_acc.reset()
        self.generation_elapsed_time = time.time() - self.generation_init_time
        self.generation_time_list.append(self.generation_elapsed_time)
        self.log("GenerationTime", self.generation_elapsed_time, on_step=False, on_epoch=True, prog_bar=True)
        log.info(f"Generation Time: {self.generation_time_list}")

    def on_test_end(self) -> None:
        mean_generation_time = np.mean(self.generation_time_list)
        std_generation_time = np.std(self.generation_time_list)
        log.info(f"Mean generation epoch time : {mean_generation_time}")
        log.info(f"Std generation epoch time : {std_generation_time}")

        # self.log("mean_generation_time", float(mean_generation_time), on_step=False, on_epoch=True, prog_bar=True)
        # self.log("std_generation_time", float(std_generation_time), on_step=False, on_epoch=True, prog_bar=True)

    def on_predict_epoch_start(self) -> None:
        log.info('On_prediction_Epoch_Start------------')

    def on_predict_epoch_end(self, results: List[Any]) -> None:
        pass

    def predict_step(self, batch: Any, batch_idx: int):
        log.info('-----------predict_step----------------')
        autoreg_all = batch["autoreg"].cuda()  # [20, 400, 45]
        # log.info(f'test_step -> autoreg_all shape: {autoreg_all.shape} \n {autoreg_all}')
        control_all = batch["control"].cuda()  # [80,400,27]
        self.is_predict_stage = True
        trainer = self.trainer
        output = self.prediction_net.forward(autoreg_all, control_all, trainer, self.is_predict_stage)
        return output

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        return {
            "optimizer": self.hparams.optimizer(params=self.parameters()),
        }


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "model" / "gesture_diffusion_lightningmodule.yaml")
    _ = hydra.utils.instantiate(cfg)
