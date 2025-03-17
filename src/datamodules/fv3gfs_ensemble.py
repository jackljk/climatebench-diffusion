from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import hydra
import torch
from tensordict import TensorDict
from torch import Tensor

from src.ace_inference.core.dataset.config import XarrayDataConfig
from src.ace_inference.core.dataset.getters import get_dataset
from src.ace_inference.core.dataset.requirements import DataRequirements
from src.ace_inference.core.dataset.xarray import XarrayDatasetSalva
from src.ace_inference.core.prescriber import Prescriber
from src.ace_inference.core.typing_ import Slice
from src.datamodules.abstract_datamodule import BaseDataModule
from src.evaluation.aggregators.main import OneStepAggregator
from src.evaluation.aggregators.time_mean import TimeMeanAggregator
from src.utilities.normalization import get_normalizer
from src.utilities.packer import Packer
from src.utilities.utils import get_logger, raise_error_if_invalid_type, to_torch_and_device


log = get_logger(__name__)


class FV3GFSEnsembleDataModule(BaseDataModule):
    def __init__(
        self,
        data_dir: str,
        in_names: List[str],
        out_names: List[str],
        forcing_names: List[str],
        loss_latitude_weighting: bool = False,
        auxiliary_names: List[str] = None,
        window: int = 1,
        horizon: int = 1,
        prediction_horizon: int = None,  # None means use horizon
        prediction_horizon_long: int = None,  # None means use horizon
        prescriber: Optional[Prescriber] = None,
        multistep_strategy: Optional[str] = None,
        data_dir_stats: Optional[str] = None,
        max_train_samples: Optional[int] = None,
        max_val_samples: Optional[int] = None,
        training_sub_paths: Optional[List[str]] = None,
        **kwargs,
    ):
        raise_error_if_invalid_type(data_dir, possible_types=[str], name="data_dir")
        if not os.path.isdir(data_dir):
            raise ValueError(f"Data dir={data_dir} not found.")
        # Pass explicit optimal parameters for local dataloading
        super().__init__(data_dir=data_dir, **kwargs)
        self.save_hyperparameters(ignore=["prescriber"])

        forcing_names = forcing_names or []
        auxiliary_names = auxiliary_names or []
        data_dir_stats = data_dir_stats or data_dir
        path_mean = Path(data_dir_stats) / "centering.nc"
        path_std = Path(data_dir_stats) / "scaling.nc"
        if not path_mean.exists() or not path_std.exists():
            raise FileNotFoundError(f"Could not find normalization files at ``{path_mean}`` and/or ``{path_std}``")
        self.train_dir = Path(data_dir) / "train"
        self.validation_dir = Path(data_dir) / "validation" / "ic_0011"

        non_forcing_names = [n for n in self.all_names if n not in forcing_names]
        self.normalizer = get_normalizer(path_mean, path_std, names=non_forcing_names)
        channel_axis = -3
        self.in_packer = Packer(in_names, axis=channel_axis)
        self.out_packer = Packer(out_names, axis=channel_axis)

        self.forcing_normalizer = get_normalizer(path_mean, path_std, names=forcing_names)
        self.forcing_packer = Packer(forcing_names, axis=channel_axis)
        if prescriber is not None:
            if not isinstance(prescriber, Prescriber):
                prescriber = hydra.utils.instantiate(prescriber)
            log.info(f"Prescribing ``{prescriber.prescribed_name}`` using mask ``{prescriber.mask_name}``")
        self.prescriber = prescriber

        if self.hparams.debug_mode:
            log.info("Running in debug mode")
            self.hparams.training_sub_paths = ["ic_0001"]
            self.hparams.max_train_samples = 80
            self.hparams.max_val_samples = 10

    def _check_args(self):
        h = self.hparams.horizon
        w = self.hparams.window
        assert isinstance(h, list) or h >= 0, f"horizon must be >= 0 or a list, but is {h}"
        assert w == 1, f"window must be 1, but is {w}"

    @property
    def all_names(self):
        forcing_names = self.hparams.forcing_names or []
        aux_names = self.hparams.auxiliary_names or []
        if self.hparams.prescriber is not None:
            aux_names = set(aux_names).union([self.hparams.prescriber.mask_name])

        all_names = list(
            set(self.hparams.in_names).union(self.hparams.out_names).union(forcing_names).union(aux_names)
        )
        return all_names

    def _create_ds(self, split: str, dataloader_idx: Optional[int] = None, **kwargs) -> Optional[XarrayDatasetSalva]:
        kwargs = kwargs.copy()
        kwargs["split_id"] = split
        horizon = self.get_horizon(split, dataloader_idx)
        n_valid_samples = self.hparams.max_val_samples
        n_samples = None
        min_idx_shift = 0
        sub_paths = None

        if split == "train":
            if self.hparams.max_train_samples is not None:
                log.info(f"Limiting training samples to {self.hparams.max_train_samples}")
                n_samples = self.hparams.max_train_samples
            if self.hparams.training_sub_paths is not None:
                sub_paths = self.hparams.training_sub_paths
                log.info(f"Limiting training sub-paths to {sub_paths}")

        elif split == "val" and n_valid_samples is not None:
            log.info(f"Limiting validation samples to {n_valid_samples}")
            n_samples = n_valid_samples if dataloader_idx in [0, None] else 8

        elif split == "test" and n_valid_samples is not None:
            log.info(f"Limiting test samples to val samples {n_valid_samples}:{n_valid_samples*3}")
            n_samples = n_valid_samples * 2
            min_idx_shift = n_valid_samples  # preclude test samples

        elif split == "predict":
            n_samples = 1

        # Create requirements object
        requirements_dict = {
            "names": list(self.all_names),
            "n_timesteps": self.hparams.window + horizon,
            # "in_names": list(self.hparams.in_names),
            # "out_names": list(self.hparams.out_names),
        }

        data_requirements = DataRequirements(**requirements_dict)

        # Create dataset configs
        data_path = self.train_dir if split == "train" else self.validation_dir
        configs = []

        # Handle ensemble data organization
        if split == "train":
            # For training data, we might have multiple ensemble members as subdirectories
            # If sub_paths is None and we're in train split, automatically discover subdirectories
            if sub_paths is None and data_path.exists():
                # Get all subdirectories in the data path that contain .nc files
                discovered_sub_paths = []
                for subdir in data_path.iterdir():
                    if subdir.is_dir() and list(subdir.glob("*.nc")):
                        discovered_sub_paths.append(subdir.name)

                if discovered_sub_paths:
                    log.info(f"Automatically discovered ensemble members: {discovered_sub_paths}")
                    sub_paths = discovered_sub_paths

            # Create configs for each ensemble member if subdirectories exist
            if sub_paths:
                # Sort sub_paths to ensure consistent order
                sub_paths = sorted(sub_paths)
                for sub_path in sub_paths:
                    ensemble_path = data_path / sub_path
                    if ensemble_path.exists():
                        config = XarrayDataConfig(
                            data_path=str(ensemble_path),
                            file_pattern="*.nc",
                            n_repeats=1,
                            engine="netcdf4",
                            spatial_dimensions="latlon",
                            subset=Slice() if n_samples is None else Slice(0, n_samples * 10, 10),
                            infer_timestep=True,
                        )
                        configs.append(config)
                    else:
                        log.warning(f"Ensemble member path not found: {ensemble_path}")

        # If no configs were created yet (non-training split or no subdirectories), use single path
        if not configs:
            config = XarrayDataConfig(
                data_path=str(data_path),
                file_pattern="*.nc",
                n_repeats=1,
                engine="netcdf4",
                spatial_dimensions="latlon",
                subset=Slice() if n_samples is None else Slice(0, n_samples * 10, 10),
                infer_timestep=True,
            )
            configs.append(config)

        # Standard keyword arguments for XarrayDatasetSalva
        kwargs_final = {
            "forcing_names": self.hparams.forcing_names,
            "forcing_packer": self.forcing_packer,
            "forcing_normalizer": self.forcing_normalizer,
            "loss_latitude_weighting": self.hparams.loss_latitude_weighting,
            "min_idx_shift": min_idx_shift,
            "dataset_class": XarrayDatasetSalva,
            # Don't pass sub_paths here since we're handling them with multiple configs
        }

        # Combine all kwargs
        kwargs_final.update(kwargs)

        # Get dataset
        ds, properties = get_dataset(configs, data_requirements, **kwargs_final)

        # Propagate properties for backward compatibility with the old implementation
        # This ensures that area_weights and other attributes are available to the rest of the code
        if hasattr(ds, "properties"):
            ds.area_weights = properties.horizontal_coordinates.area_weights
            ds.metadata = properties.variable_metadata
            ds.sigma_coordinates = properties.vertical_coordinate
            ds.horizontal_coordinates = properties.horizontal_coordinates

            # Add loss_weights_tensor directly if missing
            if not hasattr(ds, "loss_weights_tensor") and self.hparams.loss_latitude_weighting:
                ds.loss_weights_tensor = ds.area_weights

        return ds

    def setup(self, stage: Optional[str] = None):
        shared_dset_kwargs = dict()
        if stage in (None, "fit"):
            self._data_train = self._create_ds(split="train", **shared_dset_kwargs)

        if stage in (None, "fit", "validate"):
            self._data_val = [self._create_ds(split="val", **shared_dset_kwargs)]
            if self.hparams.horizon > 0 and self.hparams.prediction_horizon_long is not None:
                # Add a validation dataloader for running inference
                log.info(
                    f"Adding a validation dataset for inference with horizon {self.hparams.prediction_horizon_long}"
                )
                self._data_val += [self._create_ds(split="val", dataloader_idx=1, **shared_dset_kwargs)]

        if stage in (None, "test"):
            self._data_test = self._create_ds(split="test", **shared_dset_kwargs)
        if stage == "predict":
            self._data_predict = self._create_ds(split="predict", **shared_dset_kwargs)

        # Print sizes of the datasets (how many examples)
        self.print_data_sizes(stage)

    def boundary_conditions(
        self,
        preds: Union[Tensor, TensorDict],
        targets: Union[Tensor, TensorDict],
        data: Any = None,
        metadata: Any = None,
        time: float = None,
    ) -> Union[Tensor, TensorDict]:
        """Return predictions that satisfy the boundary conditions for a given item (batch element)."""
        if self.prescriber is None:
            return super().boundary_conditions(preds, targets, data, time)
        else:
            return self.prescriber(gen_norm=preds, target_norm=targets, data=data)

    @property
    def validation_set_names(self) -> List[str]:
        return ["val", "inference"] if len(self._data_val) > 1 else ["val"]

    def get_horizon(self, split: str, dataloader_idx: int = 0) -> int:
        if split in ["val", "validate"] and dataloader_idx == 1:
            return self.hparams.prediction_horizon_long
        assert dataloader_idx in [0, None], f"Invalid dataloader_idx: {dataloader_idx}"
        if split in ["predict", "test"]:
            return self.hparams.prediction_horizon_long or self.hparams.horizon
        elif split in ["val", "validate"]:
            return self.hparams.prediction_horizon or self.hparams.horizon
        else:
            assert split in ["train", "fit"], f"Invalid split: {split}"
            return self.hparams.horizon

    def get_epoch_aggregators(
        self,
        split: str,
        is_ensemble: bool,
        dataloader_idx: int = 0,
        experiment_type: str = None,
        device: torch.device = None,
        verbose: bool = True,
        save_to_path: str = None,
    ) -> Dict[str, OneStepAggregator]:
        assert dataloader_idx in [0, 1], f"Invalid dataloader_idx: {dataloader_idx}"
        split_ds = getattr(self, f"_data_{split}")
        if split == "val" and isinstance(split_ds, list):
            split_ds = split_ds[0]  # just need it for the area weights
        is_inference_val = split == "val" and dataloader_idx == 1
        use_full_rollout = is_inference_val or split == "test"
        if "interpolation" in experiment_type.lower():
            horizon_range = range(1, self.hparams.horizon)
        else:
            split_horizon = self.get_horizon(split, dataloader_idx)
            horizon_range = range(1, split_horizon + 1)

        # Get area weights - handle both old and new dataset formats
        if hasattr(split_ds, "area_weights"):
            area_weights = to_torch_and_device(split_ds.area_weights, device)
        elif hasattr(split_ds, "properties"):
            area_weights = to_torch_and_device(split_ds.properties.horizontal_coordinates.area_weights, device)
        else:
            # Default to None if not found
            log.warning("No area weights found in dataset")
            area_weights = None

        aggr_kwargs = dict(area_weights=area_weights, is_ensemble=is_ensemble)
        aggregators = {}
        if use_full_rollout or "interpolation" in experiment_type.lower():
            save_snapshots = True
        else:
            save_snapshots = False
        # we want to save at most 10 snapshots, including 1st, 10th, 20th, and last
        max_h = horizon_range[-1]
        if "interpolation" in experiment_type.lower():
            snapshot_horizons = [1, max_h // 2]
        elif max_h <= 10:
            snapshot_horizons = [1, max_h]
        elif max_h <= 50:
            snapshot_horizons = [1, 5, 12, 20, 32, 40, max_h]
        elif max_h <= 100:
            snapshot_horizons = [1, 12, 20, 40, 60, 80, max_h]
        elif max_h <= 200:
            snapshot_horizons = [1, 12, 20, 40, 80, 120, max_h]
        elif max_h <= 460:
            snapshot_horizons = [1, 12, 20, 120, 240, 420, max_h]
        elif max_h <= 500:
            snapshot_horizons = [1, 12, 20, 120, 240, 420, max_h]
        elif max_h <= 1460:
            snapshot_horizons = [1, 12, 20, 120, 500, 1460]
        elif max_h == 14600:
            snapshot_horizons = [40, 120, 240, 360, 420, 500, 1000, 2000, 4000, 8000, 12000, 14600]
        else:
            snapshot_horizons = []
        snaps_vars = [
            "air_temperature_7",
            "specific_total_water_7",
            "specific_total_water_0",
            "specific_total_water_7_normed",
            "air_temperature_0",
        ]
        for h in horizon_range:
            save_snaps_h = save_snapshots and h in snapshot_horizons
            aggregators[f"t{h}"] = OneStepAggregator(
                use_snapshot_aggregator=save_snaps_h,
                snapshot_var_names=snaps_vars,
                verbose=verbose and (h == 1),
                record_normed=h <= 10 or h % 50 == 0 or save_snaps_h,
                record_abs_values=h % 50 == 1,
                name=f"t{h}",
                **aggr_kwargs,
            )
        if use_full_rollout:
            aggregators["time_mean"] = TimeMeanAggregator(**aggr_kwargs, name="time_mean")
        return aggregators
