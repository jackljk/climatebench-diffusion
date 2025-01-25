from __future__ import annotations

import os.path
from datetime import timedelta
from typing import Dict, Optional, Sequence

import numpy as np
import torch
import xarray as xr
from einops import rearrange
from torch import Tensor
from torch.utils.data import Dataset

from src.datamodules.climatebench.climatebench_original import ClimateBenchDataModule
from src.evaluation.aggregators.main import OneStepAggregator


from src.utilities.climatebench_datamodule_utils import (
    yearlyInterpolator,
    monthlyInterpolator,
    standardize_output_xr,
    handle_ensemble,
    get_mean_std_of_variables,
    normalize_data,
    get_rsdt,
)
from src.utilities.utils import get_logger
import cftime

log = get_logger(__name__)


class ClimateBenchDailyDataModule(ClimateBenchDataModule):
    def __init__(
        self,
        data_dir: str = "~/data/climate-analytics-lab-shared/ClimateBench/daily/data",
        simulations: Sequence[str] = ("ssp126", "ssp370", "ssp585"),
        sim_validation: str = "ssp370",
        validation_ensemble: str = "first", # only used when mean_over_ensemble is 'stack'
        validation_size: int = 45,
        normalize_vars: Sequence[str] = ("CO2", "CH4", "SO2", "BC"),
        additional_vars: Sequence[str] = None,
        simulations_raw: Sequence[str] = None,
        simulations_anom_type: str = "piControl",
        window: int = 10,  # == slider
        output_vars: Sequence[str] | str = "tas",
        mean_over_ensemble: bool = True,
        num_ensemble_members: int|str = "all",
        scale_inputs: str = None,  
        DEBUG_dataset_size: int = None,
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
        self.TEST_SIM = "ssp245"
        self._sigma_data = None
        
        if isinstance(self.hparams.output_vars, str):
            self.hparams.output_vars = [self.hparams.output_vars]
            
        if self.hparams.mean_over_ensemble == "stack":
            self.hparams.sim_validation = {
                'X': self.hparams.sim_validation,
                'Y': self.hparams.sim_validation + "_ensemble_0"
            }
        else:
            self.hparams.sim_validation = {
                'X': self.hparams.sim_validation,
                'Y': self.hparams.sim_validation
            }
            
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
        if self._sigma_data is None:
            raise NotImplementedError("FIX")
        return self._sigma_data

    def preprocess_xarray_datasets(self, simulations: Sequence[str], mean_over_ensemble: bool = False):
        X_train, Y_train = dict(), dict()
        for i, simu in enumerate(simulations):
            # Read in the data
            input_name = os.path.join(self.hparams.data_dir, "inputs_" + simu + ".nc")
            if self.hparams.simulations_raw and simu in self.hparams.simulations_raw:
                # If the raw data is used for different normalizations
                output_name = os.path.join(self.hparams.data_dir, "outputs_" + simu + "_daily_raw.nc")
                log.info(f"Using raw data for {simu}")
            else:
                output_name = os.path.join(self.hparams.data_dir, "outputs_" + simu + "_daily.nc")

            input_xr = xr.open_dataset(input_name).compute()
            if self.hparams.simulations_raw:
                log.info(f"Loading and Normalizing raw data for {simu}")
                output_xr = xr.open_dataset(output_name)
                output_xr = self._normalize_raw_data(output_xr)
                log.info(f"Finished normalizing data for {simu}")
            else:
                log.info(f"Loading data pre-normalized data for {simu}")
                output_xr = xr.open_dataset(output_name).compute()

            # DEBUGGING MODE
            if self.hparams.DEBUG_dataset_size is not None:
                # Debugging mode
                input_xr = input_xr.isel(time=slice(0, self.hparams.DEBUG_dataset_size)).compute()
                output_xr = output_xr.isel(time=slice(0, self.hparams.DEBUG_dataset_size * 365)).compute()

            # Handle output data ensemble members
            output_xr = handle_ensemble(output_xr, mean_over_ensemble, simu, self.hparams.num_ensemble_members)

            # Standardize the output xr dataset by dropping the lat_bounds, lon_bounds and nbnd variables and renaming the
            # daily variables to be consistent with the yearly data.
            output_xr = standardize_output_xr(output_xr, simu, self.hparams.output_vars)

            # input has no members
            X_train[simu] = input_xr
            # Add the data to the training set
            if type(output_xr) == dict:
                Y_train.update(output_xr)
            else:
                Y_train[simu] = output_xr

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
            if len(self.output_vars) == 1:
                #  Compute standard deviation of the training targets, required for EDM.
                #  To improve:
                #  1a. Compute the standard deviation of the training targets for each output variable, use them separately in EDM (for each channel)
                #  1b. To use multiple output_vars, a first step could simply set sigma_data to the avg of the std of the output_vars
                #  2. Compute it over all training simulations, not one.
                log.info("Computing sigma_data")
                self._sigma_data = Y_train[sim_val['Y']].std()[self.output_vars[0]].item()
                log.info(f"sigma_data: {self._sigma_data}")
            else:
                self._sigma_data = lambda x: NotImplementedError
            log.info(f"Computed mean and std of variables.")

        # Normalize data
        if stage in ["fit", "validate", None]:
            set_train, set_val = self._setup_normalize_train_val(X_train, Y_train, X_val, Y_val)
        else:
            set_train = set_val = None

        if stage in ["test", "predict", None]:
            X_test, Y_test = self.preprocess_xarray_datasets([self.TEST_SIM])

            set_test = self._setup_test(X_test, Y_test)
        else:
            set_test = None

        # Get rsdt data
        rsdt = (
            get_rsdt(self.hparams.data_dir, self.hparams.simulations)
            if "rsdt" in self.hparams.additional_vars
            else None
        )

        ds_splits = {
            "train": set_train,
            "val": set_val,
            "test": set_test,
            "predict": set_test,
        }
        ds_vars = {
            "rsdt": rsdt,
        }

        # Set up the tensor datasets and saves to self._data_{split}
        self._setup_tensor_datasets(ds_splits, ds_vars)

        # Print sizes of the datasets (how many examples)
        self.print_data_sizes(stage)

    def _setup_train_val(self, X_train, Y_train):
        sim_val_X = self.hparams.sim_validation['X']
        sim_val_Y = self.hparams.sim_validation['Y']

        # Handle the val inputs set
        X_val = X_train[sim_val_X].isel(time=slice(-self.hparams.validation_size, None))
        X_train[sim_val_X] = X_train[sim_val_X].isel(time=slice(None, -self.hparams.validation_size))
        
        sim_val_start_year = X_val.time[0].item()
        
        
        Y_val = Y_train[sim_val_Y].sel(time=slice(self._get_validation_start_date(sim_val_start_year), None))

        # Remove the validation set from the training set
        Y_train[sim_val_Y] = Y_train[sim_val_Y].sel(
            time=slice(
                None,
                self._get_validation_start_date(sim_val_start_year) - timedelta(days=1),
            )
        )

        # Log the validation set
        log.info(f"Validating on {sim_val_Y} from {X_val.time[0].item()} to {X_val.time[-1].item()}") if self.hparams.mean_over_ensemble == 'stack' else log.info(f"Validating on {sim_val_X} from {Y_val.time[0].item()} to {Y_val.time[-1].item()}")

        return X_train, Y_train, X_val, Y_val

    def _setup_normalize_train_val(self, X_train, Y_train, X_val, Y_val):
        """
        Setup Helper for normalizing the training and validation set
        """
        sim_val_X = self.hparams.sim_validation['X']
        sim_val_Y = self.hparams.sim_validation['Y']
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
            print(f"Setting up {split} dataset")
            if xarrays is None:
                continue
            data_kwargs = dict(
                dataset_id=split,
                window=self.hparams.window,
                mean_over_mems=self.hparams.mean_over_ensemble,
                output_var=self.output_vars,
                additional_vars=ds_vars,
            )
            # Create the pytorch tensor dataset and set to to self._data_{split}
            if isinstance(xarrays, (list, tuple)):
                setattr(self, f"_data_{split}", [DailyTensorDataset(t, **data_kwargs) for t in xarrays])
            else:
                setattr(self, f"_data_{split}", DailyTensorDataset(xarrays, **data_kwargs))

    def _normalize_raw_data(self, xarray_data: xr.Dataset):
        """
        Helper function to subtract the piControl data from the training data
        """
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
            else:
                log.info(f"There is no nbnd dimension in the {k} output datasets, retaining the original dataset")
                
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
        aggregator = OneStepAggregator(
            record_rmse=True,
            use_snapshot_aggregator=True,
            record_normed=False,
            record_abs_values=True,  # will record mean and std of the absolute values of preds and targets
            snapshots_preprocess_fn=lambda x: np.flip(x, axis=-2),  # flip the latitudes for better visualization
            **aggr_kwargs,
        )
        aggregators = {"": aggregator}
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
        subsample_val: int = 1,
        additional_vars: Dict[str, xr.DataArray] = None,
    ):
        self.ds_inputs = tensors["inputs"]  # Has ssps126, ssps370, ssps585, historical
        self.ds_outputs = tensors["targets"]  # Has ssps126, ssps370, ssps585, historical
        self._parse_additional_vars(additional_vars)
        # Get the List of ssps
        ds_ssps = {
            "inputs": list(self.ds_inputs.keys()),
            "targets": list(self.ds_outputs.keys()),
        }

        # Gettting the sizes of the datasets
        if dataset_id == "val":
            # get only the validation size as the number of years
            self.ssp_sizes = {ssp: self.ds_inputs[ssp].sizes["time"] for ssp in ds_ssps['inputs']}
            if mean_over_mems == 'stack':
                # go through ssp_sizes and change the key to be equal to the ds_ssps['targets'] to handle ensem 'stack'
                self.ssp_sizes = {ds_ssps['targets'][i]: size for i, size in enumerate(self.ssp_sizes.values())}
                
            self.dataset_size = sum(self.ssp_sizes.values())
        else:
            self.ssp_sizes = {ssp: self.ds_outputs[ssp].sizes["time"] for ssp in ds_ssps['targets']}
            self.dataset_size = sum(self.ssp_sizes.values())

        # Get the cutoffs for the datasets for indexing
        self.ssp_cutoffs = (
            {ssp: sum([self.ssp_sizes[ssp_] for ssp_ in ds_ssps['targets'][:i]]) for i, ssp in enumerate(ds_ssps['targets'])}
            if len(ds_ssps['targets']) > 1
            else {ds_ssps['targets'][0]: 0}
        )

        # assert the size of the total dataset is equal to the max idx
        assert (
            self.dataset_size == max(self.ssp_cutoffs.values()) + self.ssp_sizes[ds_ssps['targets'][-1]]
        ), "Size mismatch between datasets"

        # Class Variables
        self.dataset_id = dataset_id
        self.start_datetimes = {
            ssp: self.ds_outputs[ssp].time.values[0] for ssp in ds_ssps['targets']
        }  # cftime.DatetimeNoLeap values
        self.end_datetimes = {ssp: self.ds_outputs[ssp].time.values[-1] for ssp in ds_ssps['targets']}

        self.interpolation_type = interpolation_type
        self.mean_over_mems = mean_over_mems  
        self.window = window
        self.output_var = output_var
        self.ovar_to_var_id = {
            "tas": "tas",
            "dtr": "diurnal_temperature_range",
            "pr": "pr",
            "pr90": "pr90",  # pr90 is the 90th percentile of daily precipitation
        }
        # Handle ensemble members if mean_over_mems is 'stack'
        self.handle_ensemble = lambda ssp: ssp.split('_')[0] if self.mean_over_mems == 'stack' else ssp

    def __getitem__(self, index) -> Dict[str, Tensor]:
        # Using the index value find which ssp that index belongs too
        ssp = [ssp for ssp, cutoff in self.ssp_cutoffs.items() if index >= cutoff][-1]

        
        inputs_ssp = self.ds_inputs[self.handle_ensemble(ssp)] # handle ensemble naming for input
        outputs_ssp = self.ds_outputs[ssp]

        if self.dataset_id == "val":
            outputs, ssp_index_datetime = self._sample_validation(ssp, index, outputs_ssp)
        else:
            outputs, ssp_index_datetime = self._get_outputs(ssp, index, outputs_ssp)

        # Interpolate the input data to daily resolution of X (Function handles the edge cases at first and last year)
        inputs = self._handle_interpolation_yearly(ssp, inputs_ssp, ssp_index_datetime) # ssp var needed for output ds with ensem so not using handle_ensemble
        additional_vars = self._handle_additional_vars(self.additional_vars, self.handle_ensemble(ssp), ssp_index_datetime) # additional vars are inputs with no ensemble handling so using handle_ensemble
        dset = self._reshape_raw_data(inputs, outputs, additional_vars)

        
        # Add metadata to dset
        dset["metadata"] = {
            "ssp": ssp,
            "datetime": ssp_index_datetime.strftime("%Y-%m-%d"), # convert to string for pytorch 
        }
        
        return dset

    def _handle_additional_vars(
        self, vars: list[str], ssp: str, ssp_index_datetime: cftime.datetime
    ) -> Dict[str, xr.DataArray]:
        """
        Function to handle any additional variables that can be used for experimentation in `__getitem__`

        Vars that can be added:
            - rsdt: Incoming solar radiation at the top of the atmosphere

        return:
        """
        interpolated_vars = {}
        for var in vars:
            if var == "rsdt" and self.rsdt is not None:
                # rsdt disregards ssp type only historical vs ssp
                rsdt_xr = self.rsdt["historical"] if ssp == "historical" else self.rsdt["ssp"]
                rsdt_ds = self._handle_interpolation_monthly(rsdt_xr, var, ssp, ssp_index_datetime)
                interpolated_vars[var] = rsdt_ds

        return interpolated_vars

    def _parse_additional_vars(self, additional_vars: Dict[str, xr.DataArray]):
        """
        Helper function to handle any addition variables that can be used for experimentation

        Vars that can be added:
            - rsdt: Incoming solar radiation at the top of the atmosphere

        Args:
            - additional_vars: Dict[str, xr.DataArray]
        """
        self.additional_vars = list(additional_vars.keys())
        if additional_vars is not None:
            for var_name, var_data in additional_vars.items():
                # handle for rsdt
                self.rsdt = var_data if var_name == "rsdt" else None

    def _get_outputs(self, ssp: str, index: int, outputs_ssp: xr.Dataset):
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
        ssp_index = index - self.ssp_cutoffs[ssp]
        # Get the outputs for that index
        outputs = outputs_ssp.isel(time=ssp_index)
        return outputs, outputs.time.item()

    def _sample_validation(self, ssp: str, index: int, output_ssp: xr.Dataset):
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
        year_values = output_ssp.sel(time=str(index_year))

        # Get a random day from that year
        random_day = np.random.randint(0, year_values.sizes["time"])

        # Get the values for that random day
        outputs = year_values.isel(time=random_day)

        return outputs, outputs.time.item()

    def _handle_interpolation_monthly(self, input, var, ssp, ssp_index_datetime):
        """
        Handles interpolation of input data in monthly resolution to daily resolution for edge cases (May not be needed)

        Vars that can be interpolated:
            - rsdt: Incoming solar radiation at the top of the atmosphere
            add more as needed...
        """
        if var == "rsdt":
            # Monthly scale so get year and month from the datetime
            rsdt_interpolated = monthlyInterpolator(input, ssp_index_datetime)

            # match rsdt lat and lon with the input data
            rsdt_interpolated = rsdt_interpolated.sel(
                y=self.ds_inputs[ssp]["longitude"], x=self.ds_inputs[ssp]["latitude"], method="nearest"
            )
            return rsdt_interpolated

        return None

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

    def _reshape_raw_data(
        self, inputs: Dict[str, xr.Dataset], outputs: Dict[str, xr.Dataset], additional_vars: Dict[str, xr.Dataset]
    ) -> Dict[str, Tensor]:
        if self.mean_over_mems or "member" not in inputs.dims:
            assert "member" not in inputs.dims and "member" not in outputs.dims
            transpose_dims = ["variable", "latitude", "longitude"]
        else:
            transpose_dims = ["member", "variable", "latitude", "longitude"]

        outputs = outputs.transpose(*[d for d in transpose_dims if d != "variable"])

        # Add the additional vars to the inputs
        for var_name, var_data in additional_vars.items():
            inputs = inputs.assign(
                {var_name: (("latitude", "longitude"), var_data[var_name].transpose("latitude", "longitude").data)}
            )

        X_np = inputs.to_array().transpose(*transpose_dims).data
        Y_np = np.stack(
            [outputs[self.ovar_to_var_id[ovar]].data for ovar in self.output_var], axis=-1
        )  # TODO: figure out how to handle self.output_var

        # No historical data for right now TODO add when we get the historical data for daily resolution
        # Remove window since not handling
        # Remove member handling as instead of meaning, (TODO Implementing to get more data by appending to the end treating them as different/new samples)

        # Reshape Y to (variable, lat, lon)
        Y_np = rearrange(Y_np, "lat lon var -> var lat lon")

        # pad the input data to make it shape (1, variable, lat, lon)
        X_np = np.expand_dims(X_np, axis=0)

        # Cast to float32
        X_np = X_np.astype(np.float32)
        Y_np = Y_np.astype(np.float32)

        # bring into shape (variable, lat, lon)
        assert Y_np.shape[0] == len(self.output_var)
        assert X_np.shape == (1, len(inputs.data_vars), *list(inputs.dims.values())[::-1])
        return {"inputs": X_np, "targets": Y_np}

    def __len__(self):
        return self.dataset_size
