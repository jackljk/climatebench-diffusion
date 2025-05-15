from __future__ import annotations

import copy
import os
from abc import abstractmethod
from collections import defaultdict
from os.path import join
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import dask
import numpy as np
import torch
import xarray as xr
import xbatcher
from dask.distributed import Client, LocalCluster

from src.datamodules.abstract_datamodule import BaseDataModule
from src.evaluation.aggregators.main import ListAggregator, OneStepAggregator
from src.evaluation.aggregators.save_data import SaveToDiskAggregator
from src.evaluation.metrics_wb import get_lat_weights
from src.utilities.normalization import get_normalizer
from src.utilities.packer import Packer
from src.utilities.utils import (
    get_logger,
    raise_error_if_invalid_type,
    raise_error_if_invalid_value,
    subsample_preselected_indices,
    to_torch_and_device,
)


log = get_logger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ERA5DataModuleBase(BaseDataModule):
    def __init__(
        self,
        data_dir: str,
        data_dir_stats: Optional[str] = None,
        text_data_path: Optional[str] = None,
        dataset: str = "1959-2022-1h-240x121_equiangular_with_poles_conservative.zarr",
        train_slice: Optional[slice] = slice("2015-01-01", "2018-12-31"),
        val_slice: Optional[slice] = slice("2019-01-01", "2019-12-31"),
        test_slice: Optional[slice] = slice("2020-01-01", "2020-12-31"),
        predict_slice: Optional[slice] = slice("2020-03-01", "2020-12-31", 96),
        hourly_resolution: int = 1,
        possible_initial_times: Optional[List[str]] = None,
        subsample_valid: int = 1,
        window: int = 1,  # Number of time steps to use in the input
        horizon: int = 1,  # Number of time steps to predict into the future
        multi_horizon: bool = True,
        prediction_horizon: int = None,  # None means use horizon and no auto-regressive prediction
        prediction_horizon_long: int = None,  # None means use horizon and no auto-regressive prediction
        static_fields: Sequence[str] = (
            "land_sea_mask",
            "soil_type",
            "geopotential_at_surface",
            "lat_lon_embeddings",
            # todo: add time (local time of day + day of year) as input
        ),
        spatial_crop_inputs: Optional[Dict[str, slice]] = None,
        spatial_crop_outputs: Optional[Dict[str, slice]] = None,
        spatial_crop_during_training: bool = False,  # only valid if spatial_crop_outputs is not None
        output_mask_area: Optional[str] = None,
        loss_latitude_weighting: bool = True,
        loss_pressure_weighting: bool = False,
        loss_pressure_weighting_levels: Union[str, List[int]] = "era5",  # can be "era5", "wb", or a list of levels
        loss_pressure_weighting_divide_by: str = "mean",  # can be "mean" or "sum"
        loss_surface_vars_weighting: (
            str | None
        ) = None,  # todo: implement a 1/IFS ENS performance weighting (GenCast reports success with this)
        text_period_start: Optional[str] = None,  # If None, use all text data
        text_period_end: Optional[str] = None,  # If None, use all text data. The end date is inclusive.
        shift_text_date: int = 0,
        text_history: int = 0,
        text_conditioning: str = "time",
        text_skip_missing_dates: bool = False,  # If false, use null embeddings for missing dates
        return_future_date_for_training: bool = False,
        # set to true if self.model.predict_non_spatial_condition=True
        normalize_std_fname: str = "std",  # use std_rescaled for residual prediction, std for direct prediction
        use_dask: bool = False,
        num_dask_workers: int = 16,
        dask_scheduler: str = "threads",  # can be "threads", "processes", "synchronous", "distributed"
        dask_cache_size: str = "10GB",
        lat_lon_format: str = "lon_lat",
        text_type: str = "tf-idf",  # can be tf-idf, bert, bow
        log_metrics: bool = True,
        log_normed: bool = True,
        log_images: bool = True,
        log_spectra: bool = False,
        every_nth_epoch_snapshot: int = 8,
        max_val_samples: int = None,
        **kwargs,
    ):
        """

        Args:
            data_dir (str): Path to the directory containing the zarr dataset (or a ``dataset`` subdirectory)
            data_dir_stats (str): Path to the directory containing the normalization statistics
            text_data_path (str): Path to the text data file (if using text embeddings)
            dataset (str): Name of the weatherbench2 dataset
            train_slice: slice for the training period
            val_slice: slice for the validation period
            test_slice:  slice for the test period
            predict_slice:  slice for the prediction period
            hourly_resolution:  1 for hourly, 6 for 6-hourly etc.
            possible_initial_times: Possible initial times for the prediction (e.g. ["00:00", "06:00", "12:00", "18:00"]), only use if hourly_resolution = 1
            subsample_valid: Subsample the validation set by this factor (for faster validation).
            window: The number of time steps to use in the input
            horizon: The number of time steps to predict into the future during training
            multi_horizon:
            prediction_horizon: The number of time steps to predict into the future during validation
            prediction_horizon_long: The number of time steps to predict into the future during inference/testing
            static_fields: The names of the static fields to include as conditional inputs
            spatial_crop_inputs: A dictionary of slices to crop the input fields by, if desired
            spatial_crop_outputs: A dictionary of slices to crop the output/target fields by, if desired
            spatial_crop_during_training: Only applies if spatial_crop_outputs is not None.
                If True, the spatial crop is applied during training, otherwise only during validation and testing.
            output_mask_area: Only applies if spatial_crop_outputs is not None. Used to use the spatial_crop_outputs
                mask over specific areas (e.g. "land" or "ocean").
            loss_latitude_weighting: Whether to weight the loss by cosine of latitude
            loss_pressure_weighting: Whether to weight the loss proportionally to the pressure levels (more weight to near-surface vars which have larger pressure levels)
            loss_surface_vars_weighting: How to weight the loss for surface variables. Can be "graphcast" or None
            loss_pressure_weighting_levels: Only applies if loss_pressure_weighting=True.
                Can be "era5", "wb", or a list of levels. If "era5", uses the ERA5 pressure levels. If "wb", uses the WeatherBench pressure levels.
                These levels are used to compute the normalization weights for the pressure levels.
            loss_pressure_weighting_divide_by: Only applies if loss_pressure_weighting=True. Can be "mean" or "sum".
                The pressure weighting is divided by the mean or sum of the pressure levels.
            shift_text_date:  Number of days to shift the text dates by (if positive, this uses future information!)
            text_conditioning: How to condition the text embeddings. Can be "time" or "cross_attn".
            normalize_std_fname: Which standard deviation file to use for normalization
            use_dask:
            lat_lon_format:
            num_dask_workers:
            text_type: Which type of text embeddings to use. Can be "tf-idf", "bert", or "bow"
            log_metrics: Whether to log metrics (e.g. RMSE, CRPS, etc.)
            log_images: Whether to log images (e.g. global predictions, targets, bias)
            log_spectra: Whether to log power spectra. If "targets", logs the target spectra. If true, logs predictions spectra.
            **kwargs:

        Note:
            For Autoregressive training you need to make sure that:
                - input_vars == output_vars
                - You may want to set spatial_crop_during_training=False
        """
        raise_error_if_invalid_type(data_dir, possible_types=[str], name="data_dir")
        raise_error_if_invalid_value(lat_lon_format, possible_values=["lat_lon", "lon_lat"], name="lat_lon_format")
        possible_text_conds = ["time"]  # cross_attn
        raise_error_if_invalid_value(text_conditioning, possible_values=possible_text_conds, name="text_conditioning")
        assert hourly_resolution >= 1, f"Invalid hourly_resolution: {hourly_resolution}"
        if "weatherbench2" not in data_dir:
            for name in ["weatherbench2", "weatherbench-2"]:
                if os.path.isdir(join(data_dir, name)):
                    data_dir = join(data_dir, name)
                    break
        if not data_dir.endswith(".zarr") and dataset not in data_dir:
            data_dir = join(data_dir, dataset)
        assert data_dir.endswith(".zarr"), f"Invalid data_dir: {data_dir}"
        self.zarr_path = data_dir
        dataset = os.path.basename(data_dir)
        if isinstance(predict_slice, str) and "slice" in predict_slice:
            predict_slice = eval(predict_slice)
        super().__init__(data_dir=data_dir, **kwargs)
        self.save_hyperparameters()
        if self.hparams.debug_mode:
            log.info("------------------ Running in debug mode -------------------")
            self.hparams.train_slice = slice("2015-01-01", "2015-01-05")
            self.hparams.val_slice = slice("2015-02-01", "2015-02-10")
            self.hparams.subsample_valid = 6
        if use_dask and False:
            from dask.cache import Cache as dask_Cache

            # comment these the next two lines out to disable Dask's cache
            log.info(f"Registering Dask cache with size: {dask_cache_size}")
            cache = dask_Cache(dask_cache_size)  # dask_Cache(1e10)  # 10gb cache
            cache.register()

        if self.hparams.log_spectra == "targets":
            # Don't log metrics/images when only wanting to log target spectra
            self.hparams.log_metrics = self.hparams.log_images = False

        # Infer hourly resolution of dataset
        if "-6h-" in dataset:
            if hourly_resolution == 6:
                hourly_resolution = 1
            else:
                raise ValueError(f"Invalid hourly resolution: {hourly_resolution} for dataset: {dataset}")
        elif "-1h-" in dataset:
            pass
        else:
            raise ValueError(f"Could not infer hourly resolution from dataset: {dataset}")

        # Normalization
        if data_dir_stats is None:
            if (Path(data_dir) / "statistics").exists():
                data_dir_stats = Path(data_dir) / "statistics"
            elif (Path(data_dir).parent / "statistics").exists():
                data_dir_stats = Path(data_dir).parent / "statistics"
            else:
                raise FileNotFoundError("Please specify ``data_dir_stats``. Could not find statistics directory.")
        else:
            data_dir_stats = Path(data_dir_stats)

        if "era5" not in normalize_std_fname:
            normalize_std_fname = f"era5_{normalize_std_fname}"
        normalize_std_fname += ".nc" if not normalize_std_fname.endswith(".nc") else ""

        path_mean = data_dir_stats / "era5_mean.nc"
        path_std = data_dir_stats / normalize_std_fname
        if not path_mean.exists() or not path_std.exists():
            raise FileNotFoundError(f"Could not find normalization files at ``{path_mean}`` and/or ``{path_std}``")
        self._normalizer_files = dict(mean=path_mean, std=path_std)

        self._latitude, self._longitude, self._split_to_time = None, None, dict()
        if spatial_crop_inputs is not None:
            spatial_crop_inputs = dict(**spatial_crop_inputs)
            for k, v in spatial_crop_inputs.items():
                if isinstance(v, Sequence) and len(v) == 2:
                    spatial_crop_inputs[k] = slice(*[int(x) for x in v])

        if spatial_crop_outputs is not None:
            spatial_crop_outputs = dict(**spatial_crop_outputs)
            for k, v in spatial_crop_outputs.items():
                if isinstance(v, Sequence) and len(v) == 2:
                    spatial_crop_outputs[k] = slice(*[int(x) for x in v])
            crop_lats, crop_lons = spatial_crop_outputs.get("latitude"), spatial_crop_outputs.get("longitude")
            if crop_lats == slice(10, 70) and crop_lons == slice(190, 310):
                self.crop_name = "NA"  # "north_america"
            elif crop_lats == slice(24, 50) and crop_lons == slice(235, 295):
                self.crop_name = "ConUS"  # "united_states"
            else:
                raise ValueError(f"Plese give your crop a name. Current crop: {spatial_crop_inputs}")
        else:
            self.crop_name = ""  # "global"

        self.spatial_crop_inputs = spatial_crop_inputs
        self.spatial_crop_outputs = spatial_crop_outputs

        if self.spatial_crop_inputs is not None and self.spatial_crop_outputs is not None:
            # Check that output crop is a subset of input crop
            for k, v in self.spatial_crop_outputs.items():
                crop_inputs_k = self.spatial_crop_inputs.get(k)
                if isinstance(v, slice):
                    assert v.start >= crop_inputs_k.start, f"Invalid crop for {k}: {v}"
                    assert v.stop <= crop_inputs_k.stop, f"Invalid crop for {k}: {v}"

    @property
    def dataset_identifier(self) -> str:
        iden = f"ERA5_horizon{self.hparams.horizon}"
        return iden

    @property
    def sigma_data(self) -> float:
        return 1.0

    def __del__(self):
        # Close the Dask client when the dataset is no longer needed
        if hasattr(self, "client"):
            self.client.close()

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

    def _check_args(self):
        super()._check_args()
        h = self.hparams.horizon
        w = self.hparams.window
        assert isinstance(h, list) or h > 0, f"horizon must be > 0 or a list, but is {h}"
        assert w == 1, f"window must be > 0, but is {w}"
        data_dir = self.zarr_path  # .zmetadata needs to be in the same directory
        if os.path.isdir(data_dir):  # Check local directory
            assert os.path.isfile(join(data_dir, ".zmetadata")), f"Could not find .zmetadata in data_dir: {data_dir}"

    def get_split_dataset(self, split: str, time_slice: slice, **kwargs) -> ERA5DatasetBase:
        assert split in ["fit", "train", "validate", "val", "test", "predict"], f"Invalid split: {split}"
        # gs_path = f"gs://weatherbench2/datasets/era5/{os.path.basename(self.hparams.data_dir)}"
        # Open local dataset. chunks=None is important for speed
        if self.hparams.use_dask and self.hparams.num_dask_workers is not None:
            # Configure a proper client for better resource management
            if not hasattr(self, "client") or self.client.status != "running":
                n_workers = min(self.hparams.num_dask_workers, os.cpu_count())
                threads_per_worker = max(1, os.cpu_count() // n_workers)
                n_workers, threads_per_worker = 2, 6
                memory_limit = "32GB"  # "16GB"  # "4GB"  # Adjust based on your system

                cluster = LocalCluster(
                    n_workers=n_workers,
                    threads_per_worker=threads_per_worker,
                    memory_limit=memory_limit,
                    processes=True,  # True for better isolation
                    # scheduler_port=0  # Dynamically assigned port
                )
                self.client = Client(cluster)
                log.info(f"Created Dask cluster with {n_workers=}, {threads_per_worker=}, {memory_limit=}")

                # Set task scheduling parameters for better performance
                dask.config.set(
                    {
                        # "distributed.scheduler.work-stealing": True,
                        "distributed.scheduler.work-stealing": False,  # Disable work stealing for large chunks
                        "distributed.worker.memory.target": 0.85,  # Target 85% memory utilization
                        "distributed.worker.memory.spill": 0.9,  # Spill at 90% utilization
                        "distributed.worker.memory.pause": 0.95,  # Pause at 95% utilization
                    }
                )
            elif False:
                dask_scheduler = self.hparams.dask_scheduler
                # self.client = Client(scheduler=dask_scheduler, n_workers=self.hparams.num_dask_workers, threads_per_worker=1)
                dask.config.set(scheduler=dask_scheduler, num_workers=self.hparams.num_dask_workers)
                log.info(
                    f"Using Dask with {self.hparams.num_dask_workers} workers and scheduler: ``{dask_scheduler}``"
                )

        if self.hparams.use_dask:
            horizon = self.get_horizon(split, kwargs.get("dataloader_idx"))
            chunks = {
                "time": 8,  # horizon + self.hparams.window, # 12, #24, #48,  #min(32, horizon*2),
                "latitude": -1,  # "auto",  #121,  # 180 / 1.5 = 120 + 1 for the poles
                "longitude": -1,  # 240  # 360 / 1.5 = 240
                "level": -1,  # 13,  # len(self.all_levels)
            }
            # chunks = {}  # "auto"
            log.info(f"Using optimized Dask chunks: {chunks} for horizon={horizon}")
        else:
            chunks = None
        ds = xr.open_zarr(self.zarr_path, decode_times=True, chunks=chunks, mask_and_scale=False, consolidated=True)
        ds = ds.sel(time=time_slice)
        if self.spatial_crop_inputs is not None:
            log.info(f"Applying spatial crop to inputs: {self.spatial_crop_inputs}")
            ds = ds.sel(**self.spatial_crop_inputs)
            log.info(f"New shape after spatial crop: {ds.dims}")

        kwargs["window"] = self.hparams.window
        kwargs["static_fields"] = self.hparams.static_fields
        kwargs["use_dask"] = self.hparams.use_dask
        kwargs["hourly_resolution"] = self.hparams.hourly_resolution
        kwargs["possible_initial_times"] = self.hparams.possible_initial_times
        kwargs["spatial_crop_outputs"] = self.spatial_crop_outputs
        kwargs["output_mask_area"] = self.hparams.output_mask_area
        kwargs["text_skip_missing_dates"] = self.hparams.text_skip_missing_dates
        if split in ["fit", "train"]:
            # Set loss weights
            kwargs["loss_latitude_weighting"] = self.hparams.loss_latitude_weighting
            kwargs["loss_pressure_weighting"] = self.hparams.loss_pressure_weighting
            kwargs["loss_pressure_weighting_levels"] = self.hparams.loss_pressure_weighting_levels
            kwargs["loss_pressure_weighting_divide_by"] = self.hparams.loss_pressure_weighting_divide_by
            kwargs["loss_surface_vars_weighting"] = self.hparams.loss_surface_vars_weighting
            kwargs["spatial_crop_during_training"] = self.hparams.spatial_crop_during_training

        if self.hparams.lat_lon_format == "lat_lon":
            ds = ds.transpose(..., "latitude", "longitude")  # Don't put this before spatial crop! (it will be slower)
            kwargs["lat_lon_format"] = ("latitude", "longitude")
        else:
            kwargs["lat_lon_format"] = ("longitude", "latitude")
        dset = self._get_split_dataset(ds, split, **kwargs)

        if self._latitude is None:
            self._latitude = ds.latitude  # dset.dataset.latitude
            self._longitude = ds.longitude
        split_id = split
        if "dataloader_idx" in kwargs and kwargs["dataloader_idx"] > 0:
            split_id += f"_{kwargs['dataloader_idx']}"
        self._split_to_time[split_id] = ds.time
        return dset

    @abstractmethod
    def _get_split_dataset(
        self, ds, split: str, time_slice: slice, dataloader_idx: int = 0, **kwargs
    ) -> ERA5DatasetBase:
        raise NotImplementedError

    def setup(self, stage: Optional[str] = None):
        """Load data. Set internal variables: self._data_train, self._data_val, self._data_test."""
        # Set the correct tensor datasets for the train, val, and test sets
        ds_splits = dict()
        if stage in ["fit", "validate", None]:
            ds_splits["train"] = self.get_split_dataset("fit", self.train_slice)
            val_kwargs = dict(split="validate", time_slice=self.val_slice, subsample=self.hparams.subsample_valid)
            ds_splits["val"] = [self.get_split_dataset(**val_kwargs, max_num_samples=self.hparams.max_val_samples)]
            if self.get_horizon("val", dataloader_idx=1) is not None:
                log.info(f"Using long inference horizon={self.get_horizon('val', dataloader_idx=1)} for validation")
                ds_splits["val"] += [self.get_split_dataset(**val_kwargs, max_num_samples=16, dataloader_idx=1)]

        if stage in ["test", None]:
            ds_splits["test"] = self.get_split_dataset("test", self.test_slice)
        if stage == "predict":
            ds_splits["predict"] = self.get_split_dataset("predict", self.predict_slice)

        for split, split_ds in ds_splits.items():
            if split_ds is None:
                continue
            # Save the tensor dataset to self._data_{split}
            setattr(self, f"_data_{split}", split_ds)
            assert getattr(self, f"_data_{split}") is not None, f"Could not create {split} dataset"

        # Print sizes of the datasets (how many examples)
        self.print_data_sizes(stage)

    @property
    def validation_set_names(self) -> List[str]:
        return ["val", "inference"] if len(self._data_val) > 1 else ["val"]

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

        hourly_res = self.hparams.hourly_resolution
        if "interpolation" in experiment_type.lower():
            split_horizon = self.hparams.horizon
            horizon_range = range(1, split_horizon)
        else:
            split_horizon = self.get_horizon(split, dataloader_idx)
            horizon_range = range(1, split_horizon + 1)

        aggregators_all = defaultdict(list)
        area_weights = to_torch_and_device(split_ds.area_weights_tensor, device)
        aggr_kwargs = dict(area_weights=area_weights, is_ensemble=is_ensemble)
        aggr_kwargs["coords"] = {"latitude": self._latitude, "longitude": self._longitude}
        record_normed = self.hparams.log_normed and split_horizon <= 80  # save logging space for huge horizons
        record_abs_values = True if split_horizon <= 80 else False
        record_rmse = True
        if split_ds.mask is not None:
            masks = [None, split_ds.mask]
            assert self.crop_name != "", f"Please give your crop a name. Current crop: {self.spatial_crop_outputs}"
            mask_names = ["", f"{self.crop_name}/"]
        else:
            masks = [None]
            mask_names = [""]
        if record_rmse and verbose:
            log.info(f"Recording normed metrics for {split=}, {dataloader_idx=}, {split_horizon=}")

        for mask, mask_name in zip(masks, mask_names):
            aggr_kwargs["mask"] = mask

            snapshot_horizons_hours = [1 * hourly_res, 24, 5 * 24, 10 * 24, split_horizon * hourly_res]
            if not self.hparams.log_images:
                use_snapshot_aggregator = False
            elif split == "val" and dataloader_idx == 1:
                assert len(self._data_val) > 1, "Full rollout is only supported for inference"
                # Save some example snapshots from the full rollout
                use_snapshot_aggregator = True if mask is None else False
            else:
                use_snapshot_aggregator = mask is None
            spectra_horizons_hours = snapshot_horizons_hours + [12, 3 * 24, 7 * 24, 14 * 24]

            # name=f"t{h * hourly_res}" is used for logging the appropriate lead time regardless of the hourly_res or
            # temporal resolution/subsampling of the dataset
            snapshot_var_names = ["temperature_850", "2m_temperature", "10m_u_component_of_wind"]
            # snapshot_var_names += ["10m_v_component_of_wind", "mean_sea_level_pressure"]
            snapshot_var_names = [f"{vr}_normed" for vr in snapshot_var_names] + ["geopotential_500"]
            snapshot_kwargs = {
                "var_names": snapshot_var_names,
                "preprocess_fn": lambda x: np.moveaxis(np.flip(x, axis=-1), -1, -2),
                "every_nth_epoch": self.hparams.every_nth_epoch_snapshot,
            }
            spectra_names = ["2m_temperature", "10m_u_component_of_wind", "mean_sea_level_pressure"]
            spectra_levels = [100, 500, 700, 850, 1000]
            spectra_names += [f"temperature_{lev}" for lev in spectra_levels]
            spectra_names += [f"geopotential_{lev}" for lev in spectra_levels]
            spectra_names += [f"u_component_of_wind_{lev}" for lev in spectra_levels]
            spectra_names += [f"v_component_of_wind_{lev}" for lev in spectra_levels]
            spectra_names += [f"specific_humidity_{lev}" for lev in spectra_levels]
            spectra_kwargs = {"var_names": spectra_names, "spectra_type": "zonal_60_90"}
            for h in horizon_range:
                h_to_hours = h * hourly_res
                record_spectra = self.hparams.log_spectra if h_to_hours in spectra_horizons_hours else False
                aggregators_all[f"t{h}"].append(
                    OneStepAggregator(
                        use_snapshot_aggregator=use_snapshot_aggregator and h_to_hours in snapshot_horizons_hours,
                        name=f"{mask_name}t{h_to_hours}",
                        verbose=verbose and (h == 1),
                        record_metrics=self.hparams.log_metrics,
                        record_normed=record_normed,
                        record_rmse=record_rmse,
                        record_abs_values=record_abs_values,
                        snapshot_kwargs=snapshot_kwargs,
                        # Preprocess the snapshots to flip latitudes and bring lats before lons for proper plotting
                        record_spectra=record_spectra,
                        spectra_kwargs=spectra_kwargs,
                        **aggr_kwargs,
                    )
                )

        # Make it into list aggregators, so that it is easy to call the record_batch method
        for k, v in aggregators_all.items():
            name = f"t{int(k[1:]) * hourly_res}" if k.startswith("t") else None
            aggregators_all[k] = ListAggregator(v, verbose=False, name=name)

        if save_to_path is not None:
            # Specify ++module.save_predictions_filename="xarray" to save the predictions in xarray format
            aggregators_all["save_to_disk"] = SaveToDiskAggregator(
                final_dims_of_data=["longitude", "latitude"],
                is_ensemble=is_ensemble,
                coords=aggr_kwargs["coords"],
                concat_dim_name="lead_time",
                batch_dim_name="datetime",
                save_to_path=save_to_path,
            )

        return aggregators_all


class ERA5DataModule2D(ERA5DataModuleBase):
    def __init__(
        self,
        *args,
        input_vars: Sequence[str] = (
            "mean_sea_level_pressure",
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
            "2m_temperature",
            "geopotential_500",
            "temperature_850",
        ),
        output_vars: Sequence[str] = (
            "mean_sea_level_pressure",
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
            "2m_temperature",
            "geopotential_500",
            "temperature_850",
        ),
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.all_vars = list(set(input_vars) | set(output_vars))
        self.normalizer = get_normalizer(
            self._normalizer_files["mean"], self._normalizer_files["std"], names=self.all_vars, is_2d_flattened=True
        )
        channel_axis = -3
        # ====== Don't do the following! Using set will change the order of the variables!!!!! ======
        # input_only_vars = set(input_vars) - set(output_vars)
        # in_vars_without_input_only = set(input_vars) - input_only_vars
        # ======================================================================================
        input_only_vars = [vari for vari in input_vars if vari not in output_vars]
        in_vars_without_input_only = [vari for vari in input_vars if vari not in input_only_vars]
        if len(input_only_vars) > 0:
            log.info(f"Input-only variables: {input_only_vars}")  # will be inputted with "dynamical_condition: key
        self.in_packer = Packer(in_vars_without_input_only, axis=channel_axis)
        self.in_only_packer = Packer(input_only_vars, axis=channel_axis) if len(input_only_vars) > 0 else None
        if self.spatial_crop_outputs is not None:
            channel_axis_unpack_outputs = channel_axis  # + 1  # lat and lon get flattened
        else:
            channel_axis_unpack_outputs = channel_axis
        self.out_packer = Packer(output_vars, axis_pack=channel_axis, axis_unpack=channel_axis_unpack_outputs)

    def _get_split_dataset(self, ds, split: str, dataloader_idx: int = 0, **kwargs) -> ERA5Dataset2D:
        dset = ERA5Dataset2D(
            dataset=ds,
            text_dataset=self.text_data,
            split=split,
            horizon=self.get_horizon(split, dataloader_idx),
            input_vars=self.hparams.input_vars,
            output_vars=self.hparams.output_vars,
            normalizer=self.normalizer,
            in_only_packer=self.in_only_packer,
            return_future_date_for_training=self.hparams.return_future_date_for_training,
            **kwargs,
        )
        return dset


class ERA5DatasetBase(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset: xr.Dataset,
        text_dataset: dict,
        split: str,
        horizon: int,
        static_fields: Sequence[str],
        window: int = 1,
        hourly_resolution: int = 1,
        possible_initial_times: Optional[Sequence[str]] = None,
        spatial_crop_outputs: Optional[Dict[str, slice]] = None,
        output_mask_area: Optional[str] = None,
        spatial_crop_during_training: bool = False,
        loss_latitude_weighting: bool = False,
        max_num_samples: Optional[int] = None,
        subsample: int = 1,
        lat_lon_format: Tuple[str, str] = ("longitude", "latitude"),
        use_dask: bool = False,
        text_skip_missing_dates: bool = False,
        return_future_date_for_training: bool = False,
    ):
        self.dataset = dataset
        self.max_num_samples = max_num_samples
        self.possible_initial_times = (
            [int(h) for h in possible_initial_times] if possible_initial_times is not None else None
        )
        all_times = self.dataset.time.values[:-horizon]  # keep only times for which we can predict horizon hours ahead
        ds_idxs = np.arange(len(all_times), dtype=int)
        if self.possible_initial_times is not None:
            all_hours = all_times.astype("datetime64[h]").astype(int) % 24
            # Create a mask for hours we want to keep as possible initial times
            valid_hours = np.isin(all_hours, self.possible_initial_times)
            ds_idxs = ds_idxs[valid_hours]

        if max_num_samples is not None:
            ds_idxs = subsample_preselected_indices(ds_idxs, max_num_samples)
            # Used to be: ds_idxs = ds_idxs[: max_num_samples * subsample : subsample]
        else:
            ds_idxs = ds_idxs[::subsample]

        self.ds_idxs = ds_idxs
        self.length = len(ds_idxs)
        self.subsample = subsample
        self.dataset_id = split
        self.horizon = horizon
        self.window = window
        self.loss_latitude_weighting = loss_latitude_weighting
        self.use_dask = use_dask
        self.text_skip_missing_dates = text_skip_missing_dates
        self.return_future_date_for_training = return_future_date_for_training
        # Get whether the dataset is formatted as lat/lon or lon/lat

        if self.length < 0:
            raise ValueError(
                f"Invalid length: {self.length} for split: {split}; len(self.dataset.time)={len(self.dataset.time)}, horizon: {horizon}, max_num_samples: {max_num_samples}"
            )

        # Create static conditions
        if static_fields is not None and len(static_fields) > 0:
            static_conditions = []
            for i, static_field_name in enumerate(static_fields):
                if static_field_name in self.dataset.keys():
                    try:
                        static_field = self.dataset[static_field_name].compute().values
                    except AttributeError as e:
                        raise AttributeError(
                            f"Error when loading {static_field_name=}, {self.dataset[static_field_name]=}"
                        ) from e

                    assert np.all(np.isfinite(static_field)), f"Found NaNs in static_field: {static_field_name}"
                    static_conditions.append(static_field)
                elif static_field_name == "lat_lon_embeddings":
                    # Create lat/lon embeddings for each grid point
                    # Create lat x lon meshgrid
                    lats, lons = np.meshgrid(self.dataset.latitude, self.dataset.longitude)
                    # xx, yy = 10, 88
                    # a1 = lats[xx, yy], lons[xx, yy]
                    # a2 = self.dataset.isel(latitude=yy, longitude=xx)
                    # a2 = a2.latitude.values, a2.longitude.values
                    # assert np.allclose(a1, a2), f"lats[yy, xx], lons[yy, xx]: {a1}, a2: {a2}"
                    x = np.cos(lats) * np.cos(lons)
                    y = np.cos(lats) * np.sin(lons)
                    z = np.sin(lats)
                    # Check that no NaNs are present
                    assert np.all(np.isfinite(x)), f"Found NaNs in x: {x}"
                    assert np.all(np.isfinite(y)), f"Found NaNs in y: {y}"
                    assert np.all(np.isfinite(z)), f"Found NaNs in z: {z}"
                    if lat_lon_format == ("latitude", "longitude"):
                        x, y, z = x.T, y.T, z.T
                    # log.info(f"lat_lon_format: {lat_lon_format}, x.shape: {x.shape}, y.shape: {y.shape}, z.shape: {z.shape},"
                    #       f" last_static_field: {static_conditions[-1].shape}, dims: {self.dataset.dims}")
                    static_conditions += [x, y, z]
                else:
                    raise ValueError(f"Invalid static_field: {static_field_name}")

            # Stack static conditions along channel dimension
            num_static_fields = len(static_conditions)
            static_conditions = np.stack(static_conditions, axis=0)
            # Standardize static conditions across field dimension (i.e. use different mean/std for each static field)
            mean_sc = static_conditions.mean(axis=(-2, -1), keepdims=True)
            std_sc = static_conditions.std(axis=(-2, -1), keepdims=True)
            assert len(mean_sc) == len(std_sc) == num_static_fields
            static_conditions = (static_conditions - mean_sc) / std_sc
            self.static_conditions = torch.from_numpy(static_conditions).float()
        else:
            self.static_conditions = None
        self._area_weights = get_lat_weights(self.dataset)
        self._area_weights_tensor = torch.as_tensor(self.area_weights.values, dtype=torch.float32).repeat(
            self.dataset.longitude.size, 1
        )
        if lat_lon_format == ("latitude", "longitude"):
            self._area_weights_tensor = self._area_weights_tensor.T  # transpose to (lat, lon)

        if spatial_crop_outputs is not None:
            # Compute output mask for spatial cropping of tensors
            for k, v in spatial_crop_outputs.items():
                assert k in ["latitude", "longitude"], f"Invalid spatial_crop_outputs key: {k}"
            # Get points within the spatial crop slices (e.g. latitude=slice(10, 20), longitude=slice(20, 30))
            lat_slice = spatial_crop_outputs.get("latitude", slice(None))
            lon_slice = spatial_crop_outputs.get("longitude", slice(None))
            mask = (
                (lat_slice.start <= self.dataset.latitude)
                & (self.dataset.latitude <= lat_slice.stop)
                & (lon_slice.start <= self.dataset.longitude)
                & (self.dataset.longitude <= lon_slice.stop)
            )

            if output_mask_area == "land":
                mask = mask & (self.dataset.land_sea_mask > 0.5)
            elif output_mask_area == "sea":
                mask = mask & (self.dataset.land_sea_mask < 0.5)
            elif output_mask_area is not None:
                raise ValueError(f"Invalid output_mask_area: {output_mask_area}")
            mask = mask.transpose(*lat_lon_format)
            self.mask = torch.from_numpy(mask.values)  # .nonzero(as_tuple=True)
            if split in ["train", "fit"] and spatial_crop_during_training is True:
                # Adjust area weights to fit the mask
                self.return_mask = self.mask
                self._area_weights_tensor = self._area_weights_tensor[self.mask]
            elif split in ["train", "fit"] and spatial_crop_during_training not in [False, True, None]:
                # Upweight the area weights for the masked area, but don't use the mask for cropping
                log.info(f"Upweighting area weights for the masked area by {spatial_crop_during_training}")
                self._area_weights_tensor[self.mask] *= spatial_crop_during_training
                self.return_mask = None
            else:
                # For othes (eval) splits, mask is only used inside aggregators (with special prefix(es))
                self.return_mask = None
                # log.info(f"Output mask won't be returned for split: ``{split}``")
            # self._area_weights_tensor_masked = self._area_weights_tensor[self.mask]
        else:
            assert output_mask_area is None, "output_mask_area is only supported when spatial_crop_outputs is not None"
            self.mask = self.return_mask = self._area_weights_tensor_masked = None
        self.text_dataset = text_dataset
        if self.text_dataset is not None:
            self._text_dim = len(next(iter(self.text_dataset.values())))

    @property
    def area_weights(self):
        return self._area_weights

    @property
    def area_weights_tensor(self):
        return self._area_weights_tensor

    @property
    def loss_weights_tensor(self) -> Optional[torch.Tensor]:
        weights = None
        if self.loss_latitude_weighting:
            weights = self.area_weights_tensor
        return weights

    def __len__(self):
        return self.length


class ERA5Dataset2D(ERA5DatasetBase):
    def __init__(
        self,
        *args,
        input_vars: Sequence[str],
        output_vars: Sequence[str],
        normalizer,
        in_only_packer: Packer = None,
        preprocess_to_tensor: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.input_vars = input_vars
        self.output_vars = output_vars
        self.normalizer = copy.copy(normalizer)
        self.normalizer.to("cpu")
        self.preprocess_to_tensor = preprocess_to_tensor
        # Need to flatten the 3D variables to 2D (by stacking the pressure levels)
        self.all_vars = set(input_vars) | set(output_vars)
        self.input_only_vars = set(input_vars) - set(output_vars)
        self.in_only_packer = in_only_packer
        self.skip_idxs = set()
        self.possible_2d_vars = [
            "mean_sea_level_pressure",
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
            "2m_temperature",
            "mean_top_downward_short_wave_radiation_flux",
        ]
        possible_3d_vars = [
            "geopotential",
            "specific_humidity",
            "temperature",
            "u_component_of_wind",
            "v_component_of_wind",
            # "z", "q", "t", "u", "v",
        ]
        self.vars2d = [v for v in self.all_vars if v in self.possible_2d_vars]
        vars_not_2d = [v for v in self.all_vars if v not in self.possible_2d_vars]
        self.var3d_to_levels = defaultdict(list)
        for v in vars_not_2d:
            var_name = "_".join(v.split("_")[:-1])
            if var_name in possible_3d_vars:
                p_level = int(v.split("_")[-1])
                self.var3d_to_levels[var_name].append(p_level)
            else:
                raise ValueError(f"Invalid variable: {v}")
        # Find all unique levels and filter them out
        # and sort the levels, such that higher levels come first (using the integer value)
        self.all_levels = set()
        for v in self.var3d_to_levels:
            self.var3d_to_levels[v] = sorted(self.var3d_to_levels[v], reverse=True)
            self.all_levels.update(self.var3d_to_levels[v])
        self.all_levels = sorted(self.all_levels)

        self.all_vars_stem = set(self.var3d_to_levels.keys()) | set(self.vars2d)
        ds = self.dataset[self.all_vars_stem]
        ds = ds.sel(level=self.all_levels)
        # ds = ds.sel(level=self.all_levels).load() # makes faster but can get killed

        input_dims = dict(  # input_dims for the ML model, or the __getitem__ method below
            time=self.window + self.horizon,
            latitude=len(ds.latitude),
            longitude=len(ds.longitude),
        )
        self.input_dims = input_dims
        if "level" in ds.dims and len(ds.level) >= 1:
            input_dims["level"] = len(ds.coords["level"])
        self.bgen = xbatcher.BatchGenerator(
            ds,
            input_dims=input_dims,
            preload_batch=False,
            input_overlap={"time": self.window + self.horizon - 1},
            # batch_dims={"time": 1},
        )

    def __getitem__(self, idx):
        if idx in self.skip_idxs:
            return self.__getitem__(idx + 1)
        idx_actual = int(self.ds_idxs[idx])
        # static conditions are time-independent variables such as land_sea_mask, altitude, etc.
        arrays = dict(static_condition=self.static_conditions) if self.static_conditions is not None else dict()
        dynamics = {}
        # Output-only mask for training and evaluating on spatially cropped outputs
        if self.return_mask is not None:
            arrays["predictions_mask"] = self.return_mask

        try:
            batch = self.bgen[idx_actual]  # .load()
        except OSError as e:
            new_idx = idx + 1
            log.warning(f"OSError: {e}. Trying to load a different batch {idx}->{new_idx}.")
            return self[new_idx]
        if self.use_dask:
            batch = dask.compute(batch)[0].load()
        else:
            batch = batch.load()
        # You can access the time of the batch with batch.coords['time'], which is a DataArray of datetime64
        # To select the start time of the batch, you can use batch.coords['time'].values[0]
        batch_start_time = batch.coords["time"].values[0]  # e.g. 2020-01-01T00:00:00
        # Tensorfy all 2D variables
        for vr in self.vars2d:
            dynamics[vr] = torch.from_numpy(batch[vr].values)
        # Tensorfy all 3D variables
        for vr, levels in self.var3d_to_levels.items():
            var_ds = batch[vr]
            for level in levels:
                dynamics[f"{vr}_{level}"] = torch.from_numpy(var_ds.sel(level=level).values)

        if len(self.input_only_vars) > 0:
            # Return as separate key, "dynamical_condition"
            dynamical_condition = {vr: dynamics.pop(vr) for vr in self.input_only_vars}
            arrays["dynamical_condition"] = self.in_only_packer.pack(self.normalizer.normalize(dynamical_condition))

        arrays["dynamics"] = dynamics
        if self.dataset_id not in ["train", "fit"]:
            arrays["metadata"] = dict(datetime=float(batch_start_time.astype("datetime64[s]").astype("int64")))

        return arrays


# Epoch 0:   1%|█▎                       | 202/14973 [02:19<2:49:39,  1.45it/s, v_num=hkpo, raw_loss_step=0.141, loss_step=0.136]
# Epoch 0:   3%|██▌                      | 383/14973 [07:11<4:33:51,  0.89it/s, v_num=hkpo, raw_loss_step=0.139, loss_step=0.135]
#
# With 16 workers:
# Epoch 0:   0%|▏                                                                                                   | 25/14973 [00:30<5:04:05,  0.82it/s, v_num=1nad, raw_loss_step=0.140, loss_step=0.135]
# Epoch 0:   1%|▊                                                                                                  | 117/14973 [01:29<3:08:46,  1.31it/s, v_num=1nad, raw_loss_step=0.139, loss_step=0.134]
# Epoch 0:   1%|█▎                                                                                                 | 202/14973 [02:21<2:52:37,  1.43it/s, v_num=1nad, raw_loss_step=0.141, loss_step=0.136]
# Epoch 0:   2%|██▏                                                                                                | 326/14973 [03:38<2:43:49,  1.49it/s, v_num=1nad, raw_loss_step=0.140, loss_step=0.135]
# Epoch 0:   3%|██▉                                                                                                | 444/14973 [04:51<2:38:55,  1.52it/s, v_num=1nad, raw_loss_step=0.139, loss_step=0.134]
# Epoch 0:   3%|███▏                                                                                               | 482/14973 [05:47<2:54:21,  1.39it/s, v_num=1nad, raw_loss_step=0.143, loss_step=0.138]
# Epoch 0:   3%|███▎                                                                                               | 510/14973 [06:48<3:13:13,  1.25it/s, v_num=1nad, raw_loss_step=0.144, loss_step=0.139]
# Epoch 0:   4%|███▋                                                                                               | 565/14973 [09:04<3:51:27,  1.04it/s, v_num=1nad, raw_loss_step=0.140, loss_step=0.135]
# Epoch 0:   4%|███▊                                                                                               | 572/14973 [09:08<3:50:17,  1.04it/s, v_num=1nad, raw_loss_step=0.139, loss_step=0.134]
# After setting ulimit -n 8192:
# Epoch 0:   1%|▌                                                                                                   | 76/14973 [01:02<3:24:35,  1.21it/s, v_num=6jay, raw_loss_step=0.141, loss_step=0.136]
# Epoch 0:   2%|█▌                                                                                                 | 242/14973 [02:45<2:47:25,  1.47it/s, v_num=6jay, raw_loss_step=0.140, loss_step=0.135]
# Epoch 0:   2%|██                                                                                                 | 314/14973 [03:30<2:43:28,  1.49it/s, v_num=6jay, raw_loss_step=0.140, loss_step=0.135]
# Epoch 0:   3%|███                                                                                                | 459/14973 [04:59<2:38:03,  1.53it/s, v_num=6jay, raw_loss_step=0.140, loss_step=0.135]
# Epoch 0:   4%|███▌                                                                                               | 547/14973 [05:57<2:37:18,  1.53it/s, v_num=6jay, raw_loss_step=0.140, loss_step=0.135]
# Epoch 0:   4%|████▎                                                                                              | 658/14973 [07:06<2:34:37,  1.54it/s, v_num=6jay, raw_loss_step=0.141, loss_step=0.136]
# Epoch 0:   5%|████▋                                                                                              | 701/14973 [08:05<2:44:39,  1.44it/s, v_num=6jay, raw_loss_step=0.138, loss_step=0.133]
# Epoch 0:   5%|████▉                                                                                              | 743/14973 [09:37<3:04:28,  1.29it/s, v_num=6jay, raw_loss_step=0.138, loss_step=0.133]
# Epoch 0:   6%|█████▌                                                                                             | 841/14973 [12:25<3:28:45,  1.13it/s, v_num=6jay, raw_loss_step=0.138, loss_step=0.132]
# Epoch 0:   6%|██████                                                                                             | 912/14973 [14:26<3:42:32,  1.05it/s, v_num=6jay, raw_loss_step=0.141, loss_step=0.136]
# After removing consolidate=True and torch.from_numpy:
# Epoch 0:   1%|▌                                                                                                   | 85/14973 [01:09<3:23:28,  1.22it/s, v_num=xre9, raw_loss_step=0.138, loss_step=0.133]
# Epoch 0:   1%|█▏                                                                                                 | 171/14973 [02:02<2:57:08,  1.39it/s, v_num=xre9, raw_loss_step=0.142, loss_step=0.138]
# Epoch 0:   2%|██▏                                                                                                | 333/14973 [03:43<2:43:37,  1.49it/s, v_num=xre9, raw_loss_step=0.142, loss_step=0.137]
# Epoch 0:   3%|███▎                                                                                               | 503/14973 [05:31<2:39:10,  1.52it/s, v_num=xre9, raw_loss_step=0.139, loss_step=0.134]
# Epoch 0:   4%|████                                                                                               | 623/14973 [06:45<2:35:50,  1.53it/s, v_num=xre9, raw_loss_step=0.142, loss_step=0.137]
# Epoch 0:   6%|█████▊                                                                                             | 870/14973 [09:18<2:31:00,  1.56it/s, v_num=xre9, raw_loss_step=0.139, loss_step=0.134]
# Epoch 0:   7%|███████                                                                                           | 1081/14973 [11:32<2:28:21,  1.56it/s, v_num=xre9, raw_loss_step=0.134, loss_step=0.129]
# Epoch 0:   9%|████████▊                                                                                         | 1344/14973 [16:42<2:49:23,  1.34it/s, v_num=xre9, raw_loss_step=0.134, loss_step=0.128]
