import torch
from typing import Optional, Tuple
from pytorch_lightning import LightningDataModule
import logging


class DiffFlowDataModule(LightningDataModule):

    def __init__(self,
                 data_dir: str = "data/",
                 train_val_test_split: Tuple[int, int, int] = (55_000, 5_000, 10_000),
                 batch_size: int = 64,
                 num_workers: int = 0,
                 pin_memory: bool = False, ):
        super().__init__()
        logging.info('DiffFlowDataModule')

    def prepare_data(self):
        logging.info('prepare_data')

    def setup(self, stage: Optional[str] = None):
        logging.info('setup')

    def train_dataloader(self):
        return None
