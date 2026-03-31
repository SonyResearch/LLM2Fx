import os
import json
import torch
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset, concatenate_datasets
from typing import Any, ClassVar, List, Optional, Union
import pytorch_lightning as pl
from .pt_dataset import AudioFXDataset
from torch.utils.data import DataLoader

class RegressionDataModule(pl.LightningDataModule):
    def __init__(
        self,
        audio_dir: str,
        fx_config_type: str,
        batch_size: int = 256,
        num_workers: int = 8,
    ):
        super().__init__()
        self.audio_dir = audio_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.fx_config_type = fx_config_type

    def prepare_data(self) -> None:
        """Validate that required files exist."""
        self.train_hf = load_dataset("#", split="train")
        self.val_hf = load_dataset("#", split="test")
        self.train_hf.shuffle()
        self.val_hf.shuffle()
        print(len(self.train_hf), len(self.val_hf))

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.train_dataset = AudioFXDataset(
                db=self.train_hf,
                audio_dir=self.audio_dir,
                fx_config_type = self.fx_config_type
            )
            # Load validation data
            self.val_dataset = AudioFXDataset(
                db=self.val_hf,
                audio_dir=self.audio_dir,
                fx_config_type = self.fx_config_type
            )
        elif stage == "test":
            self.test_dataset = AudioFXDataset(
                db=self.val_hf,
                audio_dir=self.audio_dir,
                fx_config_type = self.fx_config_type
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=True,
            persistent_workers=True if self.num_workers > 0 else False,
            worker_init_fn=self._worker_init_fn if self.num_workers > 0 else None
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
            persistent_workers=True if self.num_workers > 0 else False,
            worker_init_fn=self._worker_init_fn if self.num_workers > 0 else None
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
            persistent_workers=True if self.num_workers > 0 else False,
            worker_init_fn=self._worker_init_fn if self.num_workers > 0 else None
        )

    @staticmethod
    def _worker_init_fn(worker_id: int) -> None:
        """Initialize worker with different random seed."""
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
