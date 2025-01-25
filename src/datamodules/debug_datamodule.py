from __future__ import annotations

from typing import Dict, Optional

import torch
from torch.utils.data import Dataset

from src.datamodules.abstract_datamodule import BaseDataModule
from src.evaluation.one_step.main import OneStepAggregator
from src.utilities.utils import (
    get_logger,
)


log = get_logger(__name__)


class DebugDataModule(BaseDataModule):
    def __init__(
        self,
        length: int = 100,
        channels: int = 2,
        height: int = 10,
        width: int = 10,
        window: int = 1,
        horizon: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters()

    def setup(self, stage: Optional[str] = None):
        train_len = int(0.8 * self.hparams.length)
        val_len = int(0.1 * self.hparams.length)
        test_len = self.hparams.length - train_len - val_len
        ds_kwargs = dict(
            channels=self.hparams.channels,
            height=self.hparams.height,
            width=self.hparams.width,
            window=self.hparams.window,
            horizon=self.hparams.horizon,
        )
        self._data_train = DebugDataset(length=train_len, **ds_kwargs)
        self._data_val = DebugDataset(length=val_len, **ds_kwargs)
        self._data_test = DebugDataset(length=test_len, **ds_kwargs)

    @property
    def sigma_data(self) -> float:
        return 1.0

    def get_epoch_aggregators(
        self,
        split: str,
        is_ensemble: bool,
        dataloader_idx: int = 0,
        experiment_type: str = None,
        device: torch.device = None,
        verbose: bool = True,
    ) -> Dict[str, OneStepAggregator]:
        getattr(self, f"_data_{split}")

        split_horizon = self.get_horizon(split, dataloader_idx)
        if "interpolation" in experiment_type.lower():
            horizon_range = range(1, split_horizon)
        else:
            horizon_range = range(1, split_horizon + 1)

        aggr_kwargs = dict(is_ensemble=is_ensemble)
        one_step_kwargs = {
            **aggr_kwargs,
            "record_rmse": True,
            "record_normed": False,
            "use_snapshot_aggregator": False,
        }
        aggregators_all = {}
        for h in horizon_range:
            aggregators_all[f"t{h}"] = OneStepAggregator(name=f"t{h}", verbose=verbose and (h == 1), **one_step_kwargs)

        return aggregators_all


class DebugDataset(Dataset):
    def __init__(
        self,
        length: int = 100,
        channels: int = 2,
        height: int = 10,
        width: int = 10,
        window: int = 1,
        horizon: int = 1,
    ):
        self.length = length - horizon
        self.channels = channels
        self.height = height
        self.width = width
        self.window = window
        self.horizon = horizon
        # self.data = torch.randn(self.length, self.channels, self.height, self.width)

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int):
        return dict(
            dynamics=torch.randn(self.window + self.horizon, self.channels, self.height, self.width),
            dynamical_condition=torch.randn(self.window + self.horizon, 3, self.height, self.width),
            static_condition=torch.randn(2, self.height, self.width),
        )
        # return dict(dynamics=self.data[idx:idx + self.horizon + self.window])
        # return self.data[idx:idx + self.horizon + self.window]

    @property
    def loss_weights_tensor(self) -> Optional[torch.Tensor]:
        return torch.randn(self.height, self.width)
