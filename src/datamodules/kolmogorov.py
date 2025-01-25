from __future__ import annotations

import os
from os.path import join
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import xarray as xr
from einops import rearrange
from torch import Tensor

from src.datamodules.abstract_datamodule import BaseDataModule
from src.evaluation.aggregators.main import OneStepAggregator
from src.utilities.normalization import StandardNormalizer
from src.utilities.utils import (
    get_logger,
)


log = get_logger(__name__)


class KolmogorovDataModule(BaseDataModule):
    def __init__(
        self,
        data_dir: str,
        filename: str = "kolmogorov-32-250-256-256.nc",
        test_filename: str = None,
        window: int = 1,
        horizon: int = 1,
        lookback_window: int = None,  # If not None, return extra lookback window of data for input.
        prediction_horizon: int = None,  # None means use horizon and no auto-regressive prediction
        prediction_horizon_long: int = None,  # None means use horizon and no auto-regressive prediction
        channels: List[str] = ["streamfunction"],  # any subset of ["streamfunction", "vorticity"]
        downsampling_method: str = "coarsen",  # "interp" or "nearest" or "coarsen"
        spatial_downsampling_factor: int = 1,
        num_val_trajectories: int = 2,
        num_test_trajectories: int = 5,
        discard_first_n: int = 40,
        shift_test_times_by: int = 0,
        subsample_valid: int = 5,
        subsample_predict: int = None,
        max_num_train_samples: int = None,
        data_split_seed: int = 77,  # Seed for the random number generator, do not change for reproducibility
        cheat_validation: bool = False,
        add_noise_level: float = 0.0,
        standardize: bool = False,
        **kwargs,
    ):
        super().__init__(data_dir=data_dir, **kwargs)
        self.save_hyperparameters()
        assert os.path.exists(
            str(data_dir)
        ), f"Data directory does not exist: ``{data_dir}``. os.path.exists(data_dir)={os.path.exists(data_dir)}"
        filename = self.hparams.filename = str(filename)
        self.filepath = join(data_dir, filename)
        log.info(f"Using data file: {self.filepath}. Test file: {test_filename}")
        self.test_filepath = join(data_dir, test_filename) if test_filename is not None else None
        if "inits" in filename:
            # e.g. "kolmogorov-N256-n_inits32-T1000.nc"
            _, dim_x, n_trajs, n_timesteps = filename.strip(".nc").split("-")
            dim_x = dim_y = int(dim_x.strip("N"))
            n_trajs = int(n_trajs.strip("n_inits"))
            n_timesteps = int(n_timesteps.strip("T"))
        else:
            # e.g. "kolmogorov-32-250-256-256.nc"
            n_trajs, n_timesteps, dim_x, dim_y = filename.strip(".nc").split("-")[1:]
            n_trajs, n_timesteps = int(n_trajs), int(n_timesteps)

        if self.hparams.debug_mode:
            self.hparams.spatial_downsampling_factor = 4
            self.hparams.max_num_train_samples = 100
            self.hparams.subsample_valid = 80

        self.prediction_horizon = prediction_horizon or horizon
        self.prediction_horizon_long = prediction_horizon_long  # or n_timesteps - int(discard_first_n) - window - 2
        # Split the data into train, val, and test trajectories
        self.train_slice = slice(0, n_trajs - num_val_trajectories - num_test_trajectories)
        if cheat_validation:
            log.info(
                f"Cheating validation set by using first {num_val_trajectories} training trajectories for validation"
            )
            self.val_slice = slice(0, num_val_trajectories)
        else:
            self.val_slice = slice(
                n_trajs - num_val_trajectories - num_test_trajectories, n_trajs - num_test_trajectories
            )

        self.test_slice = slice(n_trajs - num_test_trajectories, None)
        self._sigma_data = self._climatology = None

        self._default_filepath = join(data_dir, "kolmogorov-32-250-256-256.nc")
        self._default_train_slice = slice(0, 32 - 5 - 2)

        # def __post_init__(self):
        # super().__post_init__()
        self.hparams.window = int(self.hparams.window)
        self.hparams.horizon = int(self.hparams.horizon)
        if self.hparams.lookback_window is not None:
            if not isinstance(self.hparams.lookback_window, (float, int)):
                assert len(self.hparams.lookback_window) == 2, f"{self.hparams.lookback_window}"
                if self.hparams.lookback_window[0] is None:
                    assert self.hparams.lookback_window[1] is not None, f"{self.hparams.lookback_window}"
                    self.hparams.lookback_window = (
                        -self.hparams.horizon + self.hparams.lookback_window[1] + 1,
                        self.hparams.lookback_window[1],
                    )
                    log.info(f"Setting lookback_window to {self.hparams.lookback_window}")
            else:
                self.hparams.lookback_window = (-self.hparams.lookback_window + 1, 0)

    def get_horizon(self, split: str, dataloader_idx: int = 0) -> int:
        if split in ["val", "validate"] and dataloader_idx == 1:
            return self.prediction_horizon_long
        assert dataloader_idx in [0, None], f"Invalid dataloader_idx: {dataloader_idx}. (split={split})"
        if split in ["predict", "test"] + self.test_set_names:
            return self.prediction_horizon_long
        elif split in ["val", "validate"]:
            return self.prediction_horizon
        else:
            assert split in ["train", "fit"], f"Invalid split: {split}"
            return self.hparams.horizon

    def get_split_dataset(
        self, ds: xr.Dataset, split: str, slice_: slice, dataloader_idx: int = 0, downsample=False, **kwargs
    ) -> KolmogorowFlowDataset:
        ds = ds.isel(init=slice_)
        if downsample:
            ds = self._downsample(ds)
        horizon = self.get_horizon(split, dataloader_idx)
        kwargs["lookback_window"] = self.hparams.lookback_window
        kwargs["add_noise_level"] = self.hparams.add_noise_level
        if split in ["test", "predict"]:
            kwargs["shift_times_by"] = self.hparams.shift_test_times_by
        return KolmogorowFlowDataset(ds, window=self.hparams.window, horizon=horizon, dataset_id=split, **kwargs)

    def _downsample(self, xr_dataset, factor: int = None) -> Tensor:
        factor = factor or self.hparams.spatial_downsampling_factor
        if factor > 1:
            if self.hparams.downsampling_method == "interp":
                new_x = xr_dataset.x[::factor]
                new_y = xr_dataset.y[::factor]
                xr_dataset = xr_dataset.interp(x=new_x, y=new_y)
            elif self.hparams.downsampling_method == "coarsen":
                xr_dataset = xr_dataset.coarsen(x=factor, y=factor, boundary="trim").mean()
            elif self.hparams.downsampling_method == "nearest":
                xr_dataset = xr_dataset.isel(
                    x=slice(None, None, factor), y=slice(None, None, factor)
                )  # nearest neighbor
            else:
                raise ValueError(f"Invalid downsampling method: {self.hparams.downsampling_method}")
            log.info(f"Downsampled the spatial dimensions by a factor of {factor} to dims: {xr_dataset.dims}")

        # Select the channels
        xr_dataset = xr_dataset[set(self.hparams.channels)]
        return xr_dataset  # xyz

    def open_postprocessed_dataset(self, filepath: str = None, downsample=True) -> xr.Dataset:
        filepath = filepath or self.filepath
        log.info(f"OPENING {filepath}")
        xr_dataset = xr.open_dataset(filepath)
        # Discard the first n timesteps
        xr_dataset = xr_dataset.isel(time=slice(self.hparams.discard_first_n, None))
        # Downsample the spatial dimensions
        if downsample:
            xr_dataset = self._downsample(xr_dataset)

        return xr_dataset

    def setup(self, stage: Optional[str] = None):
        """Load data. Set internal variables: self._data_train, self._data_val, self._data_test."""
        xr_train_dataset = None
        if stage in ["test", "predict"] and self.test_filepath is not None:
            xr_eval_dataset = self.open_postprocessed_dataset(self.test_filepath, downsample=False)
        else:
            xr_train_dataset = self.open_postprocessed_dataset()

        # Set normalizer if needed
        if self.hparams.standardize:
            if xr_train_dataset is None:
                xr_train_dataset = self.open_postprocessed_dataset()
            train_data = xr_train_dataset.isel(init=self.train_slice).to_array().values
            mean, std = torch.tensor(train_data.mean()), torch.tensor(train_data.std())
            self.normalizer = StandardNormalizer(means=mean, stds=std)
            self._sigma_data = 1.0  # Inside model, data is normalized with mean=0, std=1
        # Set the correct tensor datasets for the train, val, and test sets
        ds_splits = dict()
        if stage in ["fit", None]:
            if self._sigma_data is None:
                self._sigma_data = xr_train_dataset.isel(init=self.train_slice).to_array().std().item()
            log.info(f"Computed sigma_data={self._sigma_data} in setup()")
            ds_splits["train"] = self.get_split_dataset(
                xr_train_dataset, "fit", self.train_slice, max_num_samples=self.hparams.max_num_train_samples
            )
        if stage in ["fit", "validate", None]:
            ds_splits["val"] = [
                self.get_split_dataset(xr_train_dataset, "val", self.val_slice, subsample=self.hparams.subsample_valid)
            ]
            if (
                self.get_horizon("val", dataloader_idx=1) is not None
                and "interpolation" not in self.experiment_class_name.lower()
            ):
                log.info(f"Using long inference horizon={self.get_horizon('val', dataloader_idx=1)} for validation")
                ds_splits["val"] += [
                    self.get_split_dataset(
                        xr_train_dataset,
                        "val",
                        self.val_slice,
                        dataloader_idx=1,
                        max_num_samples=3,
                        subsample=self.hparams.subsample_valid,
                    )
                ]

        if stage in ["test", "predict"]:
            kwargs = dict(downsample=True)  # downsample after selecting the slice
            if self.test_filepath is not None:
                if self.hparams.filename == self.hparams.test_filename:
                    assert (
                        self.filepath == self.test_filepath
                    ), f"Test filepath is the same as the training filepath: {self.test_filepath}"
                    assert (
                        "kolmogorov-N256-n_inits16-T250.nc" not in self.filepath
                    ), f"train_filename={self.hparams.filename} and test_filename={self.hparams.test_filename} are the same"
                    test_slice = self.test_slice
                else:
                    log.info(f"Using a different file for loading test data: {self.test_filepath}")
                    if self.hparams.test_filename == "kolmogorov-N256-n_inits16-T250.nc":
                        test_slice = slice(None, 2)
                    elif self.hparams.test_filename == "kolmogorov-N256-n_inits32-T1000.nc":
                        test_slice = slice(0, 1)
                    elif self.hparams.test_filename == "kolmogorov-32-250-256-256.nc":
                        test_slice = slice(32 - 5, None)
                    else:
                        raise ValueError(
                            f"Unknown test_filename: ``{self.hparams.test_filename}``. Please add it to the test set."
                        )
                if self.hparams.test_filename == "kolmogorov-N256-n_inits32-T1000.nc":
                    kwargs["subsample"] = 50 if stage == "test" else self.hparams.subsample_predict or 350
                elif self.hparams.test_filename == "kolmogorov-N256-n_inits16-T250.nc":
                    kwargs["subsample"] = 50 if stage == "test" else self.hparams.subsample_predict or 200
            else:
                test_slice = self.test_slice

            if stage == "test":
                ds_splits["test"] = self.get_split_dataset(xr_eval_dataset, "test", test_slice, **kwargs)

            if stage == "predict":
                ds_splits["predict"] = self.get_split_dataset(xr_eval_dataset, "predict", test_slice, **kwargs)

        for split, split_ds in ds_splits.items():
            # Save the tensor dataset to self._data_{split}
            setattr(self, f"_data_{split}", split_ds)

        # Print sizes of the datasets (how many examples)
        self.print_data_sizes(stage)

    @property
    def sigma_data(self) -> float:
        if self._sigma_data is None:
            xr_dataset = self.open_postprocessed_dataset(self._default_filepath)
            self._sigma_data = xr_dataset.isel(init=self._default_train_slice).to_array().std().item()
            log.info(f"Computed sigma_data={self._sigma_data} in sigma_data()")
        return self._sigma_data

    @property
    def climatology(self) -> np.ndarray:
        if self._climatology is None:
            xr_dataset = self.open_postprocessed_dataset(self._default_filepath)
            climatology = xr_dataset.isel(init=self._default_train_slice).mean(dim=["init", "time"]).to_array().values
            self._climatology = torch.from_numpy(climatology).float()
            log.info(f"Computed climatology={self._climatology} in climatology(), shape={self._climatology.shape}")
        return self._climatology

    @property
    def validation_set_names(self) -> List[str]:
        return ["val", "inference"] if len(self._data_val) > 1 else ["val"]

    @property
    def test_set_names(self):
        if self.test_filepath is not None:
            if self.hparams.test_filename == "kolmogorov-N256-n_inits16-T250.nc":
                return ["test-wx"]
            elif self.hparams.test_filename == "kolmogorov-N256-n_inits32-T1000.nc":
                return ["test-wx2"]
            elif self.hparams.test_filename == "kolmogorov-32-250-256-256.nc":
                return ["test"]
            else:
                raise ValueError(
                    f"Unknown test_filename: {self.hparams.test_filename}. Please add it to the test_set_names property."
                )
        return ["test"]

    def get_epoch_aggregators(
        self,
        split: str,
        is_ensemble: bool,
        dataloader_idx: int = 0,
        experiment_type: str = None,
        device: torch.device = None,
        verbose: bool = True,
    ) -> Dict[str, OneStepAggregator]:
        aggr_kwargs = dict(is_ensemble=is_ensemble)
        one_step_kwargs = {
            **aggr_kwargs,
            "record_rmse": True,
            "use_snapshot_aggregator": False,
            "record_normed": self.hparams.standardize,
            "record_abs_values": True,  # will record mean and std of the absolute values of preds and targets
        }
        horizon = self.get_horizon(split, dataloader_idx)
        # todo: replace experiment_type with self.experiment_class_name
        if "interpolation" in experiment_type.lower():
            horizon_range = range(1, horizon)
        else:
            horizon_range = range(1, horizon + 1)

        aggregators = {f"t{h}": OneStepAggregator(**one_step_kwargs, name=f"t{h}") for h in horizon_range}
        return aggregators


class KolmogorowFlowDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset: xr.Dataset,
        window: int,
        horizon: int,
        lookback_window: Tuple[int] = None,  # If not None, return extra lookback window of data for input.
        add_noise_level: float = 0.0,
        shift_times_by: int = 0,
        subsample: int = 1,
        max_num_samples: int = None,
        dataset_id: str = None,
    ):
        super().__init__()
        self.dataset_id = dataset_id
        self.subsample = subsample
        self.window_size = window + horizon
        self.lookback_window = lookback_window
        self.shift_times_by = shift_times_by
        if self.shift_times_by > 0:
            log.info(f"[ds.id={dataset_id}] Shifting times by {self.shift_times_by}")

        self.n_trajs = len(dataset.init)
        self.length_per_traj = len(dataset.time) - horizon
        self.length = max_num_samples or self.length_per_traj * self.n_trajs // subsample
        self.div_mod_rh = self.length * self.subsample // self.n_trajs

        self.np_data = dataset.to_array().values.astype(np.float32)
        self.np_data = rearrange(self.np_data, "c traj time x y -> traj time c x y")
        if add_noise_level > 0.0:
            log.info(f"[ds.id={dataset_id}] Adding noise to the data with level={add_noise_level}")
            noise = np.random.normal(
                0, add_noise_level, self.np_data.shape
            )  # same as np.random.randn(*self.np_data.shape) * add_noise_level
            self.np_data += noise

        # log.info(f"Length per traj: {length_per_traj}, n_trajs: {self.n_trajs},  window_size: {self.window_size}, subsample: {subsample}, len(time): {len(dataset.time)}, length: {self.length}")
        # Print some mappings of idx to traj_idx and time_idx
        # idxs  = [0, 1, 2, 3, 4] + [length_per_traj - 3, length_per_traj - 2, length_per_traj - 1, length_per_traj, length_per_traj + 1] if length_per_traj <= self.length and self.length > 10 else range(self.length)
        # log.info(f"Shape data: {self.np_data.shape}")
        # for idx in idxs:
        # traj_idx, time_idx = self.idx_to_traj_idx(idx*self.subsample)
        # log.info(f"{dataset_id}: idx={idx}, traj_idx={traj_idx}, time_idx={time_idx}")
        # print(f"\t shape={self[idx]['dynamics'].shape}")

    def __len__(self):
        return self.length

    def idx_to_traj_idx(self, idx):
        """Convert the dataset index to the trajectory index and the time index"""
        traj_idx, time_idx = divmod(idx, self.div_mod_rh)
        return traj_idx, time_idx

    def __getitem__(self, idx):
        idx = idx * self.subsample
        traj_idx, time_idx = self.idx_to_traj_idx(idx)
        if self.shift_times_by > 0 and time_idx + self.shift_times_by < self.length_per_traj:
            # print(f"time_idx={time_idx}, idx={idx}. setting t_idx to {time_idx + self.shift_times_by} (from {time_idx})")
            time_idx += self.shift_times_by
        if self.lookback_window is not None:
            if time_idx + self.lookback_window[0] < 0:
                print(
                    f"left_t={time_idx + self.lookback_window[0]}, idx={idx}. setting t_idx to {time_idx + abs(self.lookback_window[0])} (from {time_idx})"
                )
                time_idx = time_idx + abs(self.lookback_window[0])
            left_t = time_idx + self.lookback_window[0]
            right_t = time_idx + self.lookback_window[1] + 1
            arrays = {"lookback": self.np_data[traj_idx, left_t:right_t, ...]}
        else:
            arrays = dict()

        arrays["dynamics"] = self.np_data[traj_idx, time_idx : time_idx + self.window_size, ...]
        return arrays


if __name__ == "__main__":
    filepath = "/Users/sruhlingcach/PycharmProjects/rdm-mi/data/kolmogorov-32-250-256-256.nc"

    # Add src to the path
    import sys

    sys.path.append("/Users/sruhlingcach/PycharmProjects/rdm-mi")
    sys.path.append("/Users/sruhlingcach/PycharmProjects/rdm-mi/src")
    dm = KolmogorovDataModule(
        data_dir=os.path.dirname(filepath),
        filename=os.path.basename(filepath),
        window=1,
        horizon=1,
        prediction_horizon_long=250,
        channels=["streamfunction"],
        num_val_trajectories=2,
        num_test_trajectories=5,
        discard_first_n=40,
        subsample_valid=5,
        max_num_train_samples=None,
        data_split_seed=77,
    )
    dm.setup("fit")
