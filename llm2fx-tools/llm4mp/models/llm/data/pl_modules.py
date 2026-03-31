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

class AudioFXDataModule(pl.LightningDataModule):
    def __init__(
        self,
        db_name: str,
        audio_dir: str,
        model_path: str,
        batch_size: int = 256,
        num_workers: int = 8,
        max_length: int = 512,
        use_cot: bool = True,
        use_chat: bool = True,
        online_sampling: bool = False,
        cache_dir: Optional[str] = None,
    ):
        super().__init__()
        self.audio_dir = audio_dir
        self.model_path = model_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.cache_dir = cache_dir
        self.db_name = db_name
        self.use_cot = use_cot
        self.use_chat = use_chat
        self.online_sampling = online_sampling

    def prepare_data(self) -> None:
        """Validate that required files exist."""
        self.train_db = load_dataset(self.db_name, split="train")
        self.val_db = load_dataset(self.db_name, split="test")
        print(len(self.train_db), len(self.val_db))

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.train_dataset = AudioFXDataset(
                db=self.train_db,
                audio_dir=self.audio_dir,
                model_path=self.model_path,
                cache_dir=self.cache_dir,
                split="train",
                use_cot=self.use_cot,
                use_chat=self.use_chat,
                online_sampling=self.online_sampling
            )
            # Load validation data
            self.val_dataset = AudioFXDataset(
                db=self.val_db,
                audio_dir=self.audio_dir,
                model_path=self.model_path,
                cache_dir=self.cache_dir,
                split="test",
                use_cot=self.use_cot,
                use_chat=self.use_chat,
                online_sampling=self.online_sampling
            )
        elif stage == "test":
            self.test_dataset = AudioFXDataset(
                db=self.val_db,
                audio_dir=self.audio_dir,
                model_path=self.model_path,
                cache_dir=self.cache_dir,
                split="test",
                use_cot=self.use_cot,
                use_chat=self.use_chat,
                online_sampling=self.online_sampling
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
            batch_size=self.batch_size,
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
