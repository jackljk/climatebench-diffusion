from __future__ import annotations

import os.path
from datetime import timedelta
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import cftime
import numpy as np
import torch
import xarray as xr
from torch import Tensor
from torch.utils.data import Dataset

from src.datamodules.climatebench.climatebench_original import ClimateBenchDataModule
from src.evaluation.aggregators.main import OneStepAggregator
from src.evaluation.aggregators.save_data import SaveToDiskAggregator
from src.evaluation.aggregators.time_mean import TimeMeanAggregator
from src.evaluation.metrics_wb import get_lat_weights
from src.utilities.climatebench_datamodule_utils import (
    get_mean_std_of_variables,
    get_rsdt,
    get_enso,
    handle_ensemble,
    monthlyInterpolator,
    normalize_data,
    standardize_output_xr,
    yearlyInterpolator,
    get_statistics,
)
from src.utilities.normalization import StandardNormalizer
from src.utilities.utils import get_logger, to_torch_and_device


log = get_logger(__name__)

statistics = {
    "tas_mean": {"weighted": 289.83469702468193, "unweighted": 280.41241455078125},
    "tas_std": {"weighted": 15.114547332402394, "unweighted": 21.20634651184082},
    "pr_mean": {"weighted": 3.5318931751025825e-05, "unweighted": 2.9361413908191025e-05},
    "pr_std": {"weighted": 8.351846484899611e-05, "unweighted": 7.215367804747075e-05},
    "log_1_pr_mean": {"weighted": 3.531428776456019e-05, "unweighted": 2.935781958512962e-05},
    "log_1_pr_std": {"weighted": 8.348880039616833e-05, "unweighted": 7.21298492862843e-05},
    # Try these:
    "log_1e-8_pr_mean": {"weighted": -12.47100560750103, "unweighted": -12.584796905517578},
    "log_1e-8_pr_std": {"weighted": 2.8926291719896695, "unweighted": 2.76674485206604},
    "log_mm_day_1_pr_mean": {"weighted": 0.8233942575117554, "unweighted": 0.7447913885116577},
    "log_mm_day_1_pr_std": {"weighted": 0.9123542090854786, "unweighted": 0.851499617099762},
    "log_mm_day_001_pr_mean": {"weighted": -0.7521765418647202, "unweighted": -0.8860650062561035},
    "log_mm_day_001_pr_std": {"weighted": 2.2917330972570222, "unweighted": 2.208401918411255},
}


class ClimateBenchDailyDataModule(ClimateBenchDataModule):
    def __init__(
        self,
        data_dir: str = "~/data/climate-analytics-lab-shared/ClimateBench/daily/data",
        simulations: Sequence[str] = ("ssp126", "ssp370", "ssp585"),
        sim_validation: str = "ssp370",
        validation_ensemble: str = "first",  # only used when mean_over_ensemble is 'stack'
        validation_size: int = 45,
        normalize_vars: Sequence[str] = ("CO2", "CH4", "SO2", "BC"),
        additional_vars: Sequence[str] = None,
        simulations_raw: Sequence[str] = None,
        simulations_anom_type: str = "piControl",
        normalization_type: str = None,  # "standard",
        precip_transform: str = None,  # how to transform precipitation data
        window: int = 10,  # == slider
        output_vars: Sequence[str] | str = "tas",
        mean_over_ensemble: bool = True,
        num_ensemble_members: int | str = "all",
        scale_inputs: str = None,
        DEBUG_dataset_size: int = None,
        comprehensive_validation=None,
        **kwargs,
    ):
        super().__init__(
            data_dir,
            simulations,
            sim_validation,
            validation_size,
            normalize_vars,
            window,
            output_vars,
            mean_over_ensemble,
            **kwargs,
        )
        # For special experiments
        self.experiments = {"G6Solar": "outputs_daily_CESM2_WACCM_G6solar.nc"}
        self.is_experiment = (
            False if sim_validation not in self.experiments else True
        )  # Experiment name contained in sim_validation

        self.hparams.additional_vars = additional_vars if additional_vars is not None else []
        self.TEST_SIM = "ssp245" if not self.is_experiment else sim_validation  # Use TEST sim to store experiment name
        self._sigma_data = None

        if isinstance(self.hparams.output_vars, str):
            self.hparams.output_vars = [self.hparams.output_vars]

        if self.hparams.mean_over_ensemble == "stack":
            self.hparams.sim_validation = {
                "X": self.hparams.sim_validation,
                "Y": self.hparams.sim_validation + "_ensemble_0",
            }
        else:
            self.hparams.sim_validation = {"X": self.hparams.sim_validation, "Y": self.hparams.sim_validation}

        # DEBUGGING MODE
        if self.hparams.DEBUG_dataset_size is not None:
            self.hparams.validation_size = self.hparams.DEBUG_dataset_size - window
        if self.hparams.debug_mode is True:
            self.hparams.DEBUG_dataset_size = 8
            self.hparams.validation_size = 4
            self.hparams.simulations = ["ssp126", sim_validation]

    @property
    def test_set_names(self) -> Sequence[str]:
        # TODO Might need to change this for the daily data
        return [self.TEST_SIM]

    @property
    def sigma_data(self) -> float:
        # Return standard deviation of the training targets
        return self._sigma_data

    def preprocess_xarray_datasets(self, simulations: Sequence[str], mean_over_ensemble: bool = False):
        X_train, Y_train = dict(), dict()
        for i, simu in enumerate(simulations):
            # Read in the data
            input_name = os.path.join(self.hparams.data_dir, "inputs_" + simu + ".nc")
            simulations_raw = self.hparams.simulations_raw
            if self.is_experiment and self.TEST_SIM == "G6Solar":
                # Handle Special case for G6Solar experiment, only need to handle output as the input should already be ssp585
                input_name = os.path.join(self.hparams.data_dir, "inputs_ssp585.nc")
                output_name = os.path.join(self.hparams.data_dir, self.experiments[self.TEST_SIM])
                is_raw = True
                log.info(f"Loading raw data for {self.TEST_SIM} experiment")
            elif simulations_raw is not None and (simulations_raw == "all" or simu in simulations_raw):
                # If the raw data is used for different normalizations
                output_name = os.path.join(self.hparams.data_dir, "outputs_" + simu + "_daily_raw.nc")
                is_raw = True
            else:
                output_name = os.path.join(self.hparams.data_dir, "outputs_" + simu + "_daily.nc")
                is_raw = False

            input_xr = xr.open_dataset(input_name).compute()
            log.info(f"Loading {'raw ' if is_raw else ''}data for {simu}")
            output_xr = xr.open_dataset(output_name)  # .compute()
            if self.hparams.simulations_anom_type not in ["none", None]:
                assert is_raw, "oops"
                output_xr = self._normalize_raw_data(output_xr, name=simu)

            # DEBUGGING MODE
            if self.hparams.DEBUG_dataset_size is not None:
                # Debugging mode
                input_xr = input_xr.isel(time=slice(0, self.hparams.DEBUG_dataset_size)).compute()
                output_xr = output_xr.isel(time=slice(0, self.hparams.DEBUG_dataset_size * 365)).compute()
                log.info(f"DEBUGGING MODE: Using {self.hparams.DEBUG_dataset_size} days of data for {simu}")

            if not self.is_experiment:
                # Handle output data ensemble members
                output_xr = handle_ensemble(output_xr, mean_over_ensemble, simu, self.hparams.num_ensemble_members)

            # Standardize the output xr dataset by dropping the lat_bounds, lon_bounds and nbnd variables and renaming the
            # daily variables to be consistent with the yearly data.
            output_xr = standardize_output_xr(output_xr, simu, self.hparams.output_vars)

            if self.is_experiment:
                X_train[self.TEST_SIM] = input_xr
            else:
                # input has no members
                X_train[simu] = input_xr
            # Add the data to the training set

            if self.is_experiment:
                Y_train[self.TEST_SIM] = output_xr
            else:
                if isinstance(output_xr, dict):
                    Y_train.update(output_xr)
                else:
                    Y_train[simu] = output_xr

        if self.hparams.sim_validation["X"] == "G6Solar":
            log.info("Finished pre-processing data for G6Solar")
        else:
            log.info(f"Finished pre-processing data for {simulations}")
        return X_train, Y_train

    def setup(self, stage: Optional[str] = None):
        """
        Similar to the original setup method, but with the following changes:
        - return dict with X_train, Y_train, X_val, Y_val, X_test, Y_test in xarray format (NOT NumPy) for dataloader
        - Idea is to handle more of the data processing in the dataloader reducing memory usage
        """
        sim_val = self.hparams.sim_validation

        # Check that the test simulation is not in the training set
        assert (
            self.TEST_SIM not in self.hparams.simulations
        ), f"Test simulation {self.TEST_SIM} should not be in the training set"

        if self.var_to_meanstd is None or stage in ["fit", "validate", None]:
            X_train, Y_train = self.preprocess_xarray_datasets(
                self.hparams.simulations, self.hparams.mean_over_ensemble
            )
            X_train, Y_train, X_val, Y_val = self._setup_train_val(X_train, Y_train)

        if self.var_to_meanstd is None:
            # Compute mean and std of variables in training set
            self.var_to_meanstd = get_mean_std_of_variables(X_train, self.hparams.normalize_vars)
            #  Compute standard deviation of the training targets, required for EDM.
            #  To improve:
            #  1a. Compute the standard deviation of the training targets for each output variable, use them separately in EDM (for each channel)
            #  1b. To use multiple output_vars, a first step could simply set sigma_data to the avg of the std of the output_vars
            #  2. Compute it over all training simulations, not one.
            var_to_transform_name = dict()
            if "standard" in self.hparams.normalization_type:
                data_mean_act, data_std_act = self.get_mean_std(sim_val, Y_train, var_to_transform_name)
                log.info(f"Standardizing data with mean: {data_mean_act}, std: {data_std_act}")
                self.normalizer_targets = StandardNormalizer(
                    means=data_mean_act,
                    stds=data_std_act,
                    var_to_transform_name=var_to_transform_name,
                )
                self._sigma_data = 1.0
            else:
                assert (
                    self.hparams.normalization_type is None
                ), f"{self.hparams.normalization_type} not implemented yet"
                assert len(self.output_vars) == 1, f"{self.output_vars} not implemented yet for multiple output_vars"
                self._sigma_data = Y_train[sim_val["Y"]].std()[self.output_vars[0]].item()
                log.info(f"sigma_data: {self._sigma_data}")

        # Normalize data
        if stage in ["fit", "validate", None]:
            set_train, set_val = self._setup_normalize_train_val(X_train, Y_train, X_val, Y_val)
        else:
            set_train = set_val = None

        if stage in ["test", "predict", None]:
            X_test, Y_test = self.preprocess_xarray_datasets([self.TEST_SIM], self.hparams.mean_over_ensemble)
            set_test = self._setup_test(X_test, Y_test)
        else:
            set_test = None
            
        # Get rsdt data
        rsdt = None
        if "rsdt" in self.hparams.additional_vars:
            if self.is_experiment:
                rsdt = get_rsdt(self.hparams.data_dir, self.hparams.simulations, solar_experiment=self.TEST_SIM)
            else:
                rsdt = get_rsdt(self.hparams.data_dir, self.hparams.simulations)

        # Get enso data
        enso = None
        if "enso" in self.hparams.additional_vars:
            # Include TEST_SIM in simulations for ENSO loading
            all_sims = list(self.hparams.simulations) + [self.TEST_SIM]
            enso = get_enso(self.hparams.data_dir, all_sims)

        ds_splits = {
            "train": set_train,
            "val": set_val,
            "test": set_test,
            "predict": set_test,
        }
        ds_vars = {
            "rsdt": rsdt,
            "enso": enso,
        }

        # Set up the tensor datasets and saves to self._data_{split}
        self._setup_tensor_datasets(ds_splits, ds_vars)

        # Print sizes of the datasets (how many examples)
        self.print_data_sizes(stage)

    def get_mean_std(self, sim_val, Y_train, var_to_transform_name):
        if ('tas' not in self.output_vars) or ('pr' not in self.output_vars) and self.hparams.normalization_type != "standard_new":
            log.info("Computing mean and std of the output variables")
            data_mean = Y_train[sim_val["Y"]].mean()
            data_std = Y_train[sim_val["Y"]].std()
            data_mean_act, data_std_act = dict(), dict()
            for ovar in self.output_vars:
                data_mean_act[ovar] = torch.tensor(data_mean[self.ovar_to_var_id[ovar]].item())
                data_std_act[ovar] = torch.tensor(data_std[self.ovar_to_var_id[ovar]].item())
        else:
            data_mean_act, data_std_act = get_statistics(
                self.output_vars, self.hparams.normalization_type, self.hparams.precip_transform, var_to_transform_name
            )
        return data_mean_act,data_std_act

    def _setup_train_val(self, X_train, Y_train):
        sim_val_X = self.hparams.sim_validation["X"]
        sim_val_Y = self.hparams.sim_validation["Y"]

        # Handle the val inputs set
        if self.hparams.comprehensive_validation.val_years is not None:
            val_time_slice = slice(
                self.hparams.comprehensive_validation.val_years[0], self.hparams.comprehensive_validation.val_years[1]
            )
            log.info(f"Only using the time period {val_time_slice} for validation")
            X_val = X_train[sim_val_X].sel(time=val_time_slice)
            val_end_year_day = self._get_validation_end_date(X_val.time[-1].item())
        else:
            # Start year with validation_size years=45 will be 2056
            X_val = X_train[sim_val_X].isel(time=slice(-self.hparams.validation_size, None))
            val_end_year_day = None

        val_start_year_day = self._get_validation_start_date(X_val.time[0].item())

        # Now remove the validation set from the training set inputs
        X_train[sim_val_X] = X_train[sim_val_X].isel(time=slice(None, -self.hparams.validation_size))

        Y_val = Y_train[sim_val_Y].sel(time=slice(val_start_year_day, val_end_year_day))

        # Remove the validation set from the training set targets
        Y_train[sim_val_Y] = Y_train[sim_val_Y].sel(time=slice(None, val_start_year_day - timedelta(days=1)))

        # Log the validation set
        if self.hparams.mean_over_ensemble == "stack":
            log.info(f"Validating on {sim_val_Y} from {X_val.time[0].item()} to {X_val.time[-1].item()}")
        else:
            log.info(f"Validating on {sim_val_X} from {Y_val.time[0].item()} to {Y_val.time[-1].item()}")

        return X_train, Y_train, X_val, Y_val

    def _setup_normalize_train_val(self, X_train, Y_train, X_val, Y_val):
        """
        Setup Helper for normalizing the training and validation set
        """
        sim_val_X = self.hparams.sim_validation["X"]
        sim_val_Y = self.hparams.sim_validation["Y"]
        X_train = {k: normalize_data(x, self.var_to_meanstd) for k, x in X_train.items()}
        X_val = {sim_val_X: normalize_data(X_val, self.var_to_meanstd)}

        # Squeeze nbnd dim
        Y_train = self._drop_nbnd_dim(Y_train)
        Y_val = {sim_val_Y: self._drop_nbnd_dim(Y_val)}

        set_train = {"inputs": X_train, "targets": Y_train}
        set_val = {"inputs": X_val, "targets": Y_val}
        return set_train, set_val

    def _setup_test(self, X_test, Y_test):
        """
        Setup Helper for test set (Does both getting and normalizing the test set)
        """
        if self.hparams.comprehensive_validation.test_years is not None:
            test_time_slice = slice(
                self.hparams.comprehensive_validation.test_years[0],
                self.hparams.comprehensive_validation.test_years[1],
            )
            log.info(f"Only using the time period {test_time_slice} for testing")
            new_X_test, new_Y_test = dict(), dict()
            for k, ds in X_test.items():
                new_X_test[k] = ds.sel(time=test_time_slice)
                test_start_date = self._get_validation_start_date(new_X_test[k].time[0].item())
                test_end_date = self._get_validation_end_date(new_X_test[k].time[-1].item())
                new_Y_test[k] = Y_test[k].sel(time=slice(test_start_date, test_end_date))
            X_test, Y_test = new_X_test, new_Y_test

        # Normalize the test set
        X_test = {k: normalize_data(x, self.var_to_meanstd) for k, x in X_test.items()}
        # Squeeze nbnd dim
        Y_test = self._drop_nbnd_dim(Y_test)
        set_test = {"inputs": X_test, "targets": Y_test}
        return set_test

    def _setup_tensor_datasets(self, ds_splits, ds_vars):
        """
        Helper to set up the tensor datasets for the given splits
        """
        for split, xarrays in ds_splits.items():
            if xarrays is None:
                continue
            data_kwargs = dict(
                dataset_id=split,
                window=self.hparams.window,
                mean_over_mems=self.hparams.mean_over_ensemble,
                output_var=self.output_vars,
                comprehensive_validation=self.hparams.comprehensive_validation,
                additional_vars=ds_vars,
            )
            # Create the pytorch tensor dataset and set to self._data_{split}
            if isinstance(xarrays, (list, tuple)):
                setattr(self, f"_data_{split}", [DailyTensorDataset(t, **data_kwargs) for t in xarrays])
            else:
                setattr(self, f"_data_{split}", DailyTensorDataset(xarrays, **data_kwargs))

    def _normalize_raw_data(self, xarray_data: xr.Dataset, name: str) -> xr.Dataset:
        """
        Helper function to subtract the piControl data from the training data
        """
        log.info(f"Subtracting {self.hparams.simulations_anom_type} data from the training data ``{name}``")
        if self.hparams.simulations_anom_type == "piControl":
            piControl = xr.open_dataset(
                os.path.join(self.hparams.data_dir, "CESM2_piControl_r1i1p1f1_climatology_daily.nc")
            ).squeeze()

            xarray_data = xarray_data.assign_coords(dayofyear=xarray_data["time"].dt.dayofyear)
            piControl_expanded = piControl.sel(dayofyear=xarray_data["dayofyear"])
            return xarray_data - piControl_expanded
        elif self.hparams.simulations_anom_type == "minmax":
            raise NotImplementedError
        else:
            raise ValueError(f"Invalid value for simulations_anom_type: {self.hparams.simulations_anom_type}")

    def _drop_nbnd_dim(self, outputs: Dict[str, xr.Dataset]) -> Dict[str, xr.Dataset]:
        """
        Squeeze the nbnd dimension from the output datasets.
        """
        for k, v in outputs.items():
            if "nbnd" in v.dims:
                outputs[k] = v.drop("nbnd")
                log.info(f"dropping nbnd dimension from the {k} output datasets")
            # else:
            #     log.info(f"There is no nbnd dimension in the {k} output datasets, retaining the original dataset")

        return outputs

    def _get_start_date(self, xarray_dict: Dict[str, xr.Dataset]) -> Dict[str, str]:
        """
        Get the start date of the xarray datasets.
        (NOTE) The start date is equal throughout all variables in the dataset.
        """
        # get first k, v pair from the dictionary
        xarray = next(iter(xarray_dict.values()))
        return str(xarray.time.values[0])

    def _get_validation_start_date(self, start_year: int) -> cftime.DatetimeNoLeap:
        return cftime.DatetimeNoLeap(start_year, 1, 1)

    def _get_validation_end_date(self, end_year: int) -> cftime.DatetimeNoLeap:
        return cftime.DatetimeNoLeap(end_year, 12, 31)

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
        # Use area weights for the aggregators to compute weighted metrics
        split_ds = getattr(self, f"_data_{split}")
        # get the coords from the dataset
        first_key = next(iter(split_ds.ds_outputs))
        coords = {
            "latitude": split_ds.ds_outputs[first_key].latitude,
            "longitude": split_ds.ds_outputs[first_key].longitude,
        }

        if split == "val" and isinstance(split_ds, list):
            split_ds = split_ds[0]  # just need it for the area weights

        area_weights = to_torch_and_device(split_ds.area_weights_tensor, device)
        aggr_kwargs = dict(area_weights=area_weights, is_ensemble=is_ensemble, coords=coords)
        aggregators = dict()

        aggregator = OneStepAggregator(
            record_rmse=True,
            use_snapshot_aggregator=True,
            record_normed=self.hparams.normalization_type is not None,
            record_abs_values=True,  # will record mean and std of the absolute values of preds and targets
            # snapshots_preprocess_fn=lambda x: np.flip(x, axis=-2),  # flip the latitudes for better visualization
            temporal_kwargs=self.hparams.comprehensive_validation,
            **aggr_kwargs,
        )
        aggregators[""] = aggregator

        if self.hparams.comprehensive_validation.run:
            # Logs metrics on the mean over time
            aggregators["time_mean"] = TimeMeanAggregator(
                **aggr_kwargs, log_images=True, mean_over_batch_dim=True, name="time_mean"
            )
            if self.hparams.comprehensive_validation.save_to_disk:
                save_to_disk_aggregator = SaveToDiskAggregator(
                    is_ensemble=is_ensemble,
                    final_dims_of_data=["latitude", "longitude"],
                    save_to_path=save_to_path,
                    max_ensemble_members=None,  # save all ensemble members
                    batch_dim_name="datetime",
                    coords=coords,
                    var_names=self.hparams.output_vars,
                )
                aggregators["save_to_disk"] = save_to_disk_aggregator

        return aggregators


class DailyTensorDataset(Dataset[Dict[str, Tensor]]):
    r"""Dataset wrapping tensors.

    Modified to include the more preprocessing steps in the __getitem__ method (reducing memory usage)

    Args:
        *self.xarrays (Dict[str, xr.DataArray]): xarray data arrays
            - keys: 'inputs', 'targets'
    """

    tensors: Dict[str, Tensor]

    def __init__(
        self,
        tensors: Dict[str, Tensor] | Dict[str, xr.DataArray],
        mean_over_mems: bool,
        window: int,
        dataset_id: str = "",
        interpolation_type: str = "middle",
        output_var: str = "tas",
        comprehensive_validation: Dict = None,
        additional_vars: Dict[str, xr.DataArray] = None,
    ):
        if comprehensive_validation is not None and not comprehensive_validation.run:
            comprehensive_validation = None  # to avoid unnecessary computation, just don't use it
        self.output_var = output_var
        self.ovar_to_var_id = {
            "tas": "tas",
            "dtr": "diurnal_temperature_range",
            "pr": "pr",
            "pr90": "pr90",  # pr90 is the 90th percentile of daily precipitation
        }
        output_var_ids = [self.ovar_to_var_id[ovar] for ovar in output_var]  # to remove unnecessary output variables
        # Bot dses below may have as keys: ssps126, ssps370, ssps585, historical, etc.
        self.ds_inputs = tensors["inputs"]
        self.ds_outputs = {k: v[output_var_ids] for k, v in tensors["targets"].items()}
        ensemble_size = next(iter(self.ds_outputs.values())).sizes.get("member_id", 1)
        for ds in self.ds_outputs.values():
            assert ds.sizes.get("member_id", 1) == ensemble_size, f"Ensemble size mismatch {ensemble_size=}, {ds=}"
        if ensemble_size > 1 and dataset_id not in ["train", "fit"]:
            ensemble_size = 1
            log.info(f"Ensemble size is {ensemble_size} for {dataset_id=}, setting it to 1")
            for k, v in self.ds_outputs.items():
                self.ds_outputs[k] = v.isel(member_id=0)
        self.ensemble_size = ensemble_size
        self._parse_additional_vars(additional_vars)
        # Set area weights
        any_output_ds = list(self.ds_outputs.values())[0]
        self._area_weights = get_lat_weights(any_output_ds)
        self._area_weights_tensor = (
            torch.as_tensor(self.area_weights.values, dtype=torch.float32).repeat(any_output_ds.longitude.size, 1).T
        )  # (lat, lon)

        # Get the List of ssps
        ds_ssps = {
            "inputs": list(self.ds_inputs.keys()),
            "targets": list(self.ds_outputs.keys()),
        }

        # Getting the sizes of the datasets
        if dataset_id == "val" and comprehensive_validation is None:
            # get only the validation size as the number of years
            self.ssp_sizes = {ssp: self.ds_inputs[ssp].sizes["time"] for ssp in ds_ssps["inputs"]}
            if mean_over_mems == "stack":
                # go through ssp_sizes and change the key to be equal to the ds_ssps['targets'] to handle ensem 'stack'
                self.ssp_sizes = {ds_ssps["targets"][i]: size for i, size in enumerate(self.ssp_sizes.values())}

            self.dataset_size = sum(self.ssp_sizes.values())
        else:
            self.ssp_sizes = {ssp: self.ds_outputs[ssp].sizes["time"] for ssp in ds_ssps["targets"]}
            self.dataset_size = sum(self.ssp_sizes.values()) * ensemble_size

        # Get the cutoffs for the datasets for indexing
        self.ssp_cutoffs = (
            {
                ssp: sum([self.ssp_sizes[ssp_] * ensemble_size for ssp_ in ds_ssps["targets"][:i]])
                for i, ssp in enumerate(ds_ssps["targets"])
            }
            if len(ds_ssps["targets"]) > 1
            else {ds_ssps["targets"][0]: 0}
        )
        # print(f"{self.ssp_sizes=}, {self.ssp_cutoffs=}")

        # assert the size of the total dataset is equal to the max idx
        assert (
            self.dataset_size
            == max(self.ssp_cutoffs.values()) + self.ssp_sizes[ds_ssps["targets"][-1]] * ensemble_size
        ), "Size mismatch between datasets"

        # Class Variables
        self.dataset_id = dataset_id
        self.comprehensive_validation = comprehensive_validation
        self.start_datetimes = {
            ssp: self.ds_outputs[ssp].time.values[0] for ssp in ds_ssps["targets"]
        }  # cftime.DatetimeNoLeap values
        self.end_datetimes = {ssp: self.ds_outputs[ssp].time.values[-1] for ssp in ds_ssps["targets"]}

        self.interpolation_type = interpolation_type
        self.mean_over_mems = mean_over_mems
        self.window = window
        # Handle ensemble members if mean_over_mems is 'stack'
        self.handle_ensemble = lambda ssp: ssp.split("_")[0] if self.mean_over_mems == "stack" else ssp

    def __getitem__(self, index) -> Dict[str, Union[Tensor, Any]]:
        # Using the index value find which ssp that index belongs too
        ssp = [ssp for ssp, cutoff in self.ssp_cutoffs.items() if index >= cutoff][-1]
        inputs_ssp = self.ds_inputs[self.handle_ensemble(ssp)]  # handle ensemble naming for input

        if self.dataset_id == "val" and self.comprehensive_validation is None:
            # Sample a random day from the validation set
            outputs, ssp_index_datetime = self._sample_validation(ssp, index)
        else:
            ssp_index = index - self.ssp_cutoffs[ssp]  #  Infer local SSP idx
            if self.ensemble_size > 1:
                # Infer timestamp and member_id from the index
                assert self.mean_over_mems == "all", f"{self.mean_over_mems=}"
                ssp_index, member_id = ssp_index % self.ssp_sizes[ssp], ssp_index // self.ssp_sizes[ssp]

            outputs, ssp_index_datetime = self._get_outputs(ssp, ssp_index)
        if self.ensemble_size > 1:
            # E.g. if ssp has length 10, for first 10 indices, mem=0 is selected, for next 10, mem=1 is selected, etc.
            outputs = outputs.isel(member_id=member_id)
            # print(f"{outputs=}")

        # Interpolate the input data to daily resolution of X (Function handles the edge cases at first and last year)
        #   ssp var needed for output ds with ensem so not using handle_ensemble
        inputs = self._handle_interpolation_yearly(ssp, inputs_ssp, ssp_index_datetime)
        coordinate_mapping = {
            "latitude": inputs.latitude,
            "longitude": inputs.longitude,
        }
        # additional vars are inputs with no ensemble handling so using handle_ensemble
        additional_vars = self._handle_additional_vars(
            self.additional_vars, self.handle_ensemble(ssp), ssp_index_datetime, coordinate_mapping
        )
        if self.mean_over_mems or "member" not in inputs.dims:
            assert "member" not in inputs.dims and "member" not in outputs.dims
            transpose_dims = ["variable", "latitude", "longitude"]
        else:
            transpose_dims = ["member", "variable", "latitude", "longitude"]

            # Add the additional vars to the inputs
        for var_name, var_data in additional_vars.items():
            inputs = inputs.assign(
                {var_name: (("latitude", "longitude"), var_data[var_name].transpose("latitude", "longitude").data)}
            )

        inputs_tensor = inputs.to_array().transpose(*transpose_dims).data.astype(np.float32)  # (C_in, H, W)
        # Shouldn't be necessary: outputs = outputs.transpose(*[d for d in transpose_dims if d != "variable"])
        # outputs = outputs.to_array().data.astype(np.float32)  # (C_out, H, W)
        outputs = {ovar: outputs[ovar].data.astype(np.float32) for ovar in self.output_var}

        # Remove member handling as instead of meaning, (TODO Implementing to get more data by appending to the end treating them as different/new samples)
        # pad the input data to make it shape (1, variable, lat, lon)
        inputs_tensor = np.expand_dims(inputs_tensor, axis=0)

        # bring into shape (variable, lat, lon)
        # assert Y_np.shape[0] == len(self.output_var)
        assert inputs_tensor.shape == (1, len(inputs.data_vars), *list(inputs.dims.values())[::-1])
        batch_element = {
            "inputs": inputs_tensor,
            "targets": outputs,
            "metadata": {  # Add metadata to dset
                "ssp": ssp,
                "datetime": ssp_index_datetime.strftime("%Y-%m-%d"),  # convert to string for pytorch
            },
        }
        return batch_element

    def _handle_additional_vars(
        self, vars: list[str], ssp: str, ssp_index_datetime: cftime.datetime, coordinate_mapping: Dict[str, xr.DataArray]
    ) -> Dict[str, xr.DataArray]:
        """
        Function to handle any additional variables that can be used for experimentation in `__getitem__`

        Vars that can be added:
            - rsdt: Incoming solar radiation at the top of the atmosphere
            - enso: ENSO (NINO3.4) index - scalar value broadcast across spatial dimensions

        return:
        """
        interpolated_vars = {}
        for var in vars:
            if var == "rsdt" and self.rsdt is not None:
                # rsdt disregards ssp type only historical vs ssp + experiments "G6Solar"
                if ssp == "G6Solar":
                    rsdt_xr = self.rsdt["G6Solar"]
                elif ssp == "historical":
                    rsdt_xr = self.rsdt["historical"]
                else:
                    try:
                        rsdt_xr = self.rsdt['ssp']
                    except KeyError:
                        raise KeyError(f"rsdt data not available for {ssp}")
                    except Exception as e:
                        raise Exception(f"Error accessing rsdt data for {ssp}: {e}")
                rsdt_ds = self._handle_interpolation_monthly(rsdt_xr, var, ssp, ssp_index_datetime, coordinate_mapping)
                interpolated_vars[var] = rsdt_ds
            elif var == "enso" and self.enso is not None:
                # ENSO data is keyed by specific SSP (not generic "ssp" like rsdt)
                if ssp == "historical":
                    enso_xr = self.enso["historical"]
                else:
                    # Try to get the specific SSP, fall back to ssp245 (test set default)
                    try:
                        enso_xr = self.enso[ssp]
                    except KeyError:
                        # Fallback: use ssp245 if available, otherwise raise error
                        if "ssp245" in self.enso:
                            enso_xr = self.enso["ssp245"]
                        else:
                            raise KeyError(f"ENSO data not available for {ssp}")
                enso_ds = self._handle_interpolation_monthly_scalar(
                    enso_xr, var, ssp_index_datetime, coordinate_mapping
                )
                interpolated_vars[var] = enso_ds
            else:
                raise NotImplementedError(f"Additional variable {var} not implemented yet")

        return interpolated_vars

    def _parse_additional_vars(self, additional_vars: Dict[str, xr.DataArray]):
        """
        Helper function to handle any addition variables that can be used for experimentation

        Vars that can be added:
            - rsdt: Incoming solar radiation at the top of the atmosphere
            - enso: ENSO (NINO3.4) index - scalar value broadcast across spatial dimensions

        Args:
            - additional_vars: Dict[str, xr.DataArray]
        """
        self.additional_vars = list(additional_vars.keys())
        self.rsdt = None
        self.enso = None
        if additional_vars is not None:
            for var_name, var_data in additional_vars.items():
                if var_name == "rsdt":
                    self.rsdt = var_data
                elif var_name == "enso":
                    self.enso = var_data

    def _get_outputs(self, ssp: str, ssp_index: int) -> Tuple[xr.Dataset, cftime.datetime]:
        """
        Helper function to get the outputs for a given ssp and index

        - ssp: the ssp for the given index
        - index: index of the dataset during __getitem__ method
        - outputs_ssp: the outputs for the given ssp

        return
        - outputs: the outputs for the given index
        - ssp_index_datetime: the datetime for the given index
        """
        # Get the index of the ssp
        # ssp_index = index - self.ssp_cutoffs[ssp]
        # Get the outputs for that index
        outputs = self.ds_outputs[ssp].isel(time=ssp_index)
        return outputs, outputs.time.item()

    def _sample_validation(self, ssp: str, index: int):
        """
        Subsamples a random day from a given year in the validation set (Validation set should just be x years in size)
        NOTE: This is to ensure that the validation set is not biased towards the start or end of the year

        - ssp: the ssp for the given index
        - index: index of the year in the validation set
            (Index 0 is the first year in the validation set)
        - output_ssp: the outputs for the given ssp

        return
        - outputs: the outputs for the given index
        - ssp_index_datetime: the datetime for the given index
        """
        index_year = self.start_datetimes[ssp].year + index

        # Get the all values from that year in output_ssp
        year_values = self.ds_outputs[ssp].sel(time=str(index_year))

        # Get a random day from that year
        random_day = np.random.randint(0, year_values.sizes["time"])

        # Get the values for that random day
        outputs = year_values.isel(time=random_day)

        return outputs, outputs.time.item()

    def _handle_interpolation_monthly(self, input, var, ssp, ssp_index_datetime, coordinate_mapping):
        """
        Handles interpolation of input data in monthly resolution to daily resolution for edge cases (May not be needed)

        Vars that can be interpolated:
            - rsdt: Incoming solar radiation at the top of the atmosphere
            add more as needed...
        """
        FACTOR = 2 # To match the input data lat and lon with the rsdt data
        if var == "rsdt":
            # Monthly scale so get year and month from the datetime
            rsdt_interpolated = monthlyInterpolator(input, ssp_index_datetime)

            # coarsen rsdt lat and lon to match with the input data
            rsdt_interpolated = rsdt_interpolated.coarsen(x=FACTOR, y=FACTOR, boundary='trim').mean()
            # Handle the lat lng values not being the same
            rsdt_interpolated = rsdt_interpolated.rename({"x": "longitude", "y": "latitude"}).assign_coords(
                longitude=coordinate_mapping['longitude'], latitude=coordinate_mapping['latitude']
            )
            return rsdt_interpolated

        return None

    def _handle_interpolation_monthly_scalar(
        self, input_xr: xr.Dataset, var: str, ssp_index_datetime: cftime.datetime, coordinate_mapping: Dict[str, xr.DataArray]
    ) -> xr.Dataset:
        """
        Handles interpolation of scalar monthly data (like ENSO) to daily resolution and broadcasts
        across spatial dimensions.

        Args:
            - input_xr: xarray.Dataset with scalar values (no lat/lon dimensions)
            - var: variable name (e.g., 'enso')
            - ssp_index_datetime: target datetime for interpolation
            - coordinate_mapping: dict with 'latitude' and 'longitude' coordinates to broadcast to

        Returns:
            - xr.Dataset with the interpolated value broadcast across lat/lon dimensions
        """
        # Get the variable name in the dataset (NINO34 for ENSO)
        var_name_mapping = {"enso": "NINO34"}
        data_var = var_name_mapping.get(var, var.upper())
        
        # Use monthlyInterpolator to get the interpolated scalar value
        interpolated = monthlyInterpolator(input_xr, ssp_index_datetime)
        
        # Extract the scalar value
        if data_var in interpolated.data_vars:
            scalar_value = float(interpolated[data_var].values)
        else:
            scalar_value = float(interpolated.values)
        
        # Broadcast the scalar value across the spatial dimensions
        lat_coords = coordinate_mapping['latitude']
        lon_coords = coordinate_mapping['longitude']
        
        # Create a DataArray with the scalar value broadcast across lat/lon
        broadcast_data = np.full((len(lat_coords), len(lon_coords)), scalar_value, dtype=np.float32)
        
        result = xr.Dataset(
            {var: (["latitude", "longitude"], broadcast_data)},
            coords={"latitude": lat_coords, "longitude": lon_coords}
        )
        
        return result

    def _handle_interpolation_yearly(self, ssp, inputs_ssp, ssp_index_datetime):
        """
        Handles the interpolation of input data in yearly resolution to daily resolution
        - To handle the edge cases of the first and last year
        """

        if ssp_index_datetime.year == self.start_datetimes[ssp].year:
            DATE = cftime.DatetimeNoLeap(self.start_datetimes[ssp].year, 7, 2)
            if ssp_index_datetime <= DATE:
                # handle the edge case of the first year by interpolating with the same year
                inputs = yearlyInterpolator(inputs_ssp, ssp_index_datetime, no_prev_year=True)
            else:
                # Else interpolate with the next year as normal
                inputs = yearlyInterpolator(inputs_ssp, ssp_index_datetime)
        elif ssp_index_datetime.year == self.end_datetimes[ssp].year:
            DATE = cftime.DatetimeNoLeap(self.end_datetimes[ssp].year, 7, 2)
            if ssp_index_datetime > DATE:
                # Handle the edge case of the last year by interpolating with the same year
                inputs = yearlyInterpolator(inputs_ssp, ssp_index_datetime, no_next_year=True)
            else:
                # Else interpolate with the previous year as normal
                inputs = yearlyInterpolator(inputs_ssp, ssp_index_datetime)
        else:
            inputs = yearlyInterpolator(inputs_ssp, ssp_index_datetime)

        return inputs

    @property
    def area_weights(self):
        return self._area_weights

    @property
    def area_weights_tensor(self):
        return self._area_weights_tensor

    @property
    def loss_weights_tensor(self) -> Optional[torch.Tensor]:
        return self.area_weights_tensor

    def __len__(self):
        return self.dataset_size
