from __future__ import annotations

from datetime import datetime, timedelta
from os.path import join
from typing import Dict, Optional, Sequence

import cftime
import numpy as np
import torch
import xarray as xr
from einops import rearrange
from torch import Tensor
from torch.utils.data import Dataset

from src.datamodules.climatebench.climatebench_original import ClimateBenchDataModule
from src.datamodules.climatebench.daily_helpers import timeInterpolateMiddle
from src.evaluation.one_step.main import OneStepAggregator
from src.utilities.utils import get_files, get_logger


log = get_logger(__name__)


def get_rsdt(
    data_path: str,
    simulations: Sequence[str],
) -> xr.Dataset:
    """
    Load the raw data from the given path and return it as a dictionary of xarray datasets.

    Avaliable rsdt simulations are:
    - 'CESM2-rsdt-Amon-gn-piControl.nc',
    - 'CESM2-rsdt-Amon-gn-ssp126.nc',
    - 'CESM2-rsdt-Amon-gn-historical.nc'

    Args:
    - data_path: Path to the directory containing the data

    Returns:
    - rsdt: Dictionary containing the input data for each simulation
    """
    rsdt = dict()
    rsdt_paths = get_files(data_path, "Amon")
    if "historical" in simulations:
        # get the file with historical
        rsdt_path = [path for path in rsdt_paths if "historical" in path][0]
        log.info(f"Loading historical rsdt data from {rsdt_path}")
        rsdt["historical"] = xr.open_dataset(data_path + f"/{rsdt_path}").compute()

    # For now don't load piControl
    rsdt_path = [path for path in rsdt_paths if "ssp126" in path][0]
    log.info(f"Loading rsdt data from {rsdt_path}")
    rsdt["ssp"] = xr.open_dataset(data_path + f"/{rsdt_path}").compute()

    # Squeeze the nbnd & member_id dimension from the output datasets
    for k, v in rsdt.items():
        if "nbnd" in v.dims:
            rsdt[k] = v.drop_dims("nbnd")
            print(f"dropping nbnd dimension from the rsdt{k} variable datasets")
        if "member_id" in v.dims:
            rsdt[k] = v.drop_vars("member_id")
            print(f"dropping member_id dimension from the rsdt{k} variable datasets")

    return rsdt


def get_raw_data(
    data_path: str,
    simulations: Sequence[str],
    stage: str,
    mean_over_ensemble: bool = False,
    Debug_dataset_size: int = None,
    scale_inputs: str = None,
) -> tuple(Dict[str, xr.Dataset], Dict[str, xr.Dataset]):  # type: ignore
    """
    Load the raw data from the given path and return it as a dictionary of xarray datasets.

    Avaliable daily output simulations are:
    - 'output_ssp126_daily'
    - 'output_ssp245_daily'
    - 'output_ssp370_daily'
    - 'output_ssp585_daily'
    - 'outputs_historical_daily_raw'

    Avaliable Input simulations are yearly:
    - 'input_hist-GHG'
    - 'inputs_abrupt-4xCO2'
    - 'inputs_1pctCO2'
    - 'inputs_hist-aer'
    - 'inputs_historical'
    - 'inputs_ssp126'
    - 'inputs_ssp245'
    - 'inputs_ssp370'
    - 'inputs_ssp370-lowNTCF
    - 'inputs_ssp585'

    Args:
    - data_path: Path to the directory containing the data
    - simulations: List of simulations to load
    - stage: The stage of the data to load. Either 'train' or 'validation'
    - mean_over_ensemble: If True, the output data will be averaged over the ensemble dimension
    - Debug_dataset_size: If not None, the size of the dataset to load
    - scale_inputs: If not None, the type of scaling to apply to the input data (downscale or upscale)

    Returns:
    - X_train: Dictionary containing the input data for each simulation
    - Y_train: Dictionary containing the output data for each simulation
    """
    X_train, Y_train = dict(), dict()
    for i, simu in enumerate(simulations):
        input_name = "inputs_" + simu + ".nc"
        output_name = "outputs_" + simu + "_daily.nc"
        if Debug_dataset_size is not None:
            input_xr = (
                xr.open_dataset(join(data_path, input_name))
                .sel(time=slice("2015", str(2015 + Debug_dataset_size)))
                .compute()
            )
            output_xr = (
                xr.open_dataset(join(data_path, output_name))
                .sel(time=slice("2015", str(2015 + Debug_dataset_size)))
                .compute()
            )
        else:
            log.info(f"Loading data from {data_path}")
            input_xr = xr.open_dataset(join(data_path, input_name)).compute()
            log.info(f"Loaded input data from {input_name}")
            # output_xr = xr.open_dataset(join(data_path, output_name)).compute()
            output_xr = xr.open_dataset(join(data_path, output_name), chunks={"member_id": -1}).compute()
            log.info(f"Loaded output data from {output_name}")

        # assert not mean_over_ensemble, "don't wanna explore using mean_over_ensemble."
        if mean_over_ensemble:
            log.info(f"Average over ensemble for {simu}. Ds.dims: {output_xr.dims}")
            output_xr = output_xr.mean(dim="member_id")
            log.info(f"Finished averaging over ensemble for {simu}")
            # transpose_dims = ["time", "latitude", "longitude"]
        else:
            # drop each member where there's a nan in the output
            print("Dataset data_vars: ", output_xr.data_vars)
            # check_nan_ds = output_xr.tas
            # nan_mems = check_nan_ds.isnull().any(dim=["time", "latitude", "longitude"])
            # print("Dropping members with nan in output: ", nan_mems.values)
            # copy input_xr for each ensemble member
            input_xr = xr.concat([input_xr] * output_xr.sizes["member_id"], dim="member_id")
            # transpose_dims = ["time", "member", "latitude", "longitude"]
            output_xr = output_xr.rename({"member_id": "member"})
            input_xr = input_xr.rename({"member_id": "member"})

        # Drop lat_bound, lng_bounds and nbnd to make the data tranpose simpler
        output_xr = output_xr.drop(["lon_bounds", "lat_bounds", "nbnd"])

        # If the input data has lon and lat as dimensions, rename them to longitude and latitude
        if "lon" in input_xr.dims:
            input_xr = input_xr.rename({"lon": "longitude", "lat": "latitude"})

        # Rename the daily variables to be consistent with the yearly data
        if "y" or "x" in output_xr.dims:
            output_xr = output_xr.rename({"y": "latitude", "x": "longitude"})
        # Convert pr and pr90 to mm/day and rename lon and lat to longitude and latitude
        log.info(f"Converting pr and pr90 to mm/day for {simu}")
        output_xr["pr"] *= 86400  # less efficient: output_xr.assign({"pr": output_xr.pr * 86400})  # no pr90
        log.info(f"Finished converting pr and pr90 to mm/day for {simu}")

        # ! Note: Commented out for now
        # output_xr = output_xr.transpose(*transpose_dims)
        X_train[simu] = input_xr
        Y_train[simu] = output_xr

        # Match in/output Spatial resolution
        if scale_inputs is not None:
            log.info(f"Matching inputs res. to output res. for {simu}")
            X_train[simu], Y_train[simu] = _scaleInterpolateLinear(
                X_train[simu], Y_train[simu], scale_type=scale_inputs
            )

    log.info(f"Finished pre-processing data for {simulations}")
    return X_train, Y_train


def _scaleInterpolateLinear(X, Y, scale_type=None):
    """
    Interpolate daily climate data using linear interpolation.

    X: Yearly resolution input data
    Y: Daily resolution output data
    """
    if scale_type is None:
        return X, Y
    elif scale_type == "downscale":  # Increases the resolution of the input data
        lat = Y.latitude
        lon = Y.longitude
        X_upscaled = X.interp(latitude=lat, longitude=lon, method="linear")
        # Make sure all negative values are set to 0 (I don't know why this happens when all the data is positive, but it could be due to the sheer number of 0s, which could cause some values to become negative even thought that does not make sense)
        X_upscaled = X_upscaled.where(X_upscaled >= 0, 0)
        return X_upscaled, Y
    elif scale_type == "upscale":  # Decreases the resolution of the output data
        # Define the coarser grid
        coarse_lat = X["latitude"]
        coarse_lon = X["longitude"]

        # Calculate the downscaling factors
        lat_factor = len(Y["latitude"]) // len(X["latitude"])
        lon_factor = len(Y["longitude"]) // len(X["longitude"])

        # Downscale df2 by averaging over blocks of the size of downscaling factors
        Y_downscaled = Y.coarsen(latitude=lat_factor, longitude=lon_factor, boundary="trim").mean()

        # Align coordinates
        Y_downscaled = Y_downscaled.assign_coords(y=coarse_lat, x=coarse_lon)
        return X, Y_downscaled
    else:
        raise ValueError(f"Invalid scale type: {scale_type}")


# !! Identical as might have to change to fit to the daily data
def get_mean_std_of_variables(
    training_set: Dict[str, xr.Dataset], variables: Sequence[str] | None
) -> Dict[str, Dict[str, float]]:
    #                  simulations: Sequence[str] = ('ssp126', 'ssp370', 'ssp585', 'hist-GHG', 'hist-aer'),
    #                  skip_hist = ds_key in ['ssp126', 'ssp370']
    if variables is None:
        # use all variables in the dataset
        variables = list(training_set.values())[0].data_vars.keys()
    var_to_meanstd = {}
    for var in variables:
        array = np.concatenate([training_set[k][var].data.reshape(-1) for k in training_set.keys()], axis=0)
        var_to_meanstd[var] = {"mean": array.mean(), "std": array.std()}
    return var_to_meanstd


def normalize_data(data: xr.Dataset, var_to_meanstd: Dict[str, Dict[str, float]]) -> xr.Dataset:
    #         var_dims = train_xr[var].dims
    #         train_xr=train_xr.assign({var: (var_dims, normalize(train_xr[var].data, var, meanstd_inputs))})
    for var, meanstd in var_to_meanstd.items():
        data[var] = (data[var] - meanstd["mean"]) / meanstd["std"]
    return data


def denormalize_data(data: xr.Dataset, var_to_meanstd: Dict[str, Dict[str, float]]) -> xr.Dataset:
    for var, meanstd in var_to_meanstd.items():
        data[var] = data[var] * meanstd["std"] + meanstd["mean"]
    return data


class ClimateBenchDailyDataModule(ClimateBenchDataModule):
    def __init__(
        self,
        data_dir: str = "~/data/climate-analytics-lab-shared/ClimateBench/daily/data",
        simulations: Sequence[str] = ("ssp126", "ssp370", "ssp585"),
        sim_validation: str = "ssp370",
        validation_size: int = 45,
        normalize_vars: Sequence[str] = ("CO2", "CH4", "SO2", "BC"),
        window: int = 10,  # == slider
        output_vars: Sequence[str] | str = "tas",
        mean_over_ensemble: bool = True,
        scale_inputs: str = None,  # New parameter
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
        if self.hparams.DEBUG_dataset_size is not None:
            self.hparams.validation_size = self.hparams.DEBUG_dataset_size - window

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

    def setup(self, stage: Optional[str] = None):
        # Compute mean and std of variables
        get_raw_data_kwargs = dict(
            Debug_dataset_size=self.hparams.DEBUG_dataset_size, scale_inputs=self.hparams.scale_inputs
        )

        assert (
            self.TEST_SIM not in self.hparams.simulations
        ), f"Test simulation {self.TEST_SIM} should not be in the training set"
        if self.var_to_meanstd is None or stage in ["fit", "validate", None]:
            X_train, Y_train = get_raw_data(
                self.hparams.data_dir,
                self.hparams.simulations,
                stage="fit",
                mean_over_ensemble=self.hparams.mean_over_ensemble,
                **get_raw_data_kwargs,
            )

            # split validation set from training set
            sim_val = self.hparams.sim_validation
            X_val = X_train[sim_val].isel(time=slice(-self.hparams.validation_size, None))
            sim_val_start_year = X_val.time[0].item()
            Y_val = Y_train[sim_val].sel(time=slice(self._get_validation_start_date(sim_val_start_year), None))
            val_start_date = self._get_start_date(Y_val)  # Get the start date of the validation set

            X_train[sim_val] = X_train[sim_val].isel(time=slice(None, -self.hparams.validation_size))
            Y_train[sim_val] = Y_train[sim_val].sel(
                time=slice(None, self._get_validation_start_date(sim_val_start_year) - timedelta(days=1))
            )
            train_start_dates = self._get_start_date(Y_train)  # Get the start date of the training set

            log.info(f"Validating on {sim_val} from {X_val.time[0].item()} to {X_val.time[-1].item()}")

        if self.var_to_meanstd is None:
            # Compute mean and std of variables in training set
            self.var_to_meanstd = get_mean_std_of_variables(X_train, self.hparams.normalize_vars)
            if len(self.output_vars) == 1:
                #  Compute standard deviation of the training targets, required for EDM.
                #  To improve:
                #  1a. Compute the standard deviation of the training targets for each output variable, use them separately in EDM (for each channel)
                #  1b. To use multiple output_vars, a first step could simply set sigma_data to the avg of the std of the output_vars
                #  2. Compute it over all training simulations, not one.
                self._sigma_data = Y_train[sim_val].std()[self.output_vars[0]].item()
                print(f"sigma_data: {self._sigma_data}")
            else:
                self._sigma_data = lambda x: NotImplementedError
            log.info("Computed mean and std of variables.")

        # Normalize data
        if stage in ["fit", "validate", None]:
            X_train = {k: normalize_data(x, self.var_to_meanstd) for k, x in X_train.items()}
            X_val = normalize_data(X_val, self.var_to_meanstd)
            # Squeeze nbnd dim
            Y_train = self._drop_nbnd_dim(Y_train)
            Y_val = self._drop_nbnd_dim(Y_val)
            log.info("Normalized data.")
            np_dset_train = self._reshape_raw_data(X_train, Y_train)
            log.info("Reshaped data train.")
            np_dset_val = self._reshape_raw_data({"val": X_val}, {"val": Y_val})
            log.info("Reshaped data val.")
        else:
            np_dset_train = np_dset_val = None
        if stage in ["test", "predict", None]:
            X_test, Y_test = get_raw_data(
                self.hparams.data_dir, [self.TEST_SIM], stage="test", mean_over_ensemble=False, **get_raw_data_kwargs
            )
            test_start_dates = self._get_start_date(Y_test)

            # TODO: Note that the test set is normalized before extrapolating to daily resolution
            X_test = {k: normalize_data(x, self.var_to_meanstd) for k, x in X_test.items()}
            Y_test = self._drop_nbnd_dim(Y_test)
            np_dset_test = self._reshape_raw_data(X_test, Y_test)
            np_dset_test = [np_dset_test] + [
                self._reshape_raw_data(
                    {k: v.sel(time=slice("2080", "2100")) for k, v in X_test.items()},
                    {k: v.sel(time=slice("2080", "2100")) for k, v in Y_test.items()},
                )
            ]
        else:
            np_dset_test = None
            test_start_dates = None

        ds_splits = {"train": np_dset_train, "val": np_dset_val, "test": np_dset_test, "predict": np_dset_test}
        ds_starts = {
            "train": train_start_dates,
            "val": val_start_date,
            "test": test_start_dates,
            "predict": test_start_dates,
        }

        for split, numpy_tensors in ds_splits.items():
            split_start_dates = ds_starts[split]
            log.info(f"Setting up {split} dataset")
            if numpy_tensors is None:
                continue
            if isinstance(numpy_tensors, (list, tuple)):
                setattr(
                    self,
                    f"_data_{split}",
                    [DailyTensorDataset(t, split_start_dates, dataset_id=split) for t in numpy_tensors],
                )
            else:
                # Create the pytorch tensor dataset
                tensor_ds = DailyTensorDataset(numpy_tensors, split_start_dates, dataset_id=split)
                # Save the tensor dataset to self._data_{split}
                setattr(self, f"_data_{split}", tensor_ds)

        # Print sizes of the datasets (how many examples)
        self.print_data_sizes(stage)

    def _drop_nbnd_dim(self, outputs: Dict[str, xr.Dataset]) -> Dict[str, xr.Dataset]:
        """
        Squeeze the nbnd dimension from the output datasets.
        """
        for k, v in outputs.items():
            if "nbnd" in v.dims:
                outputs[k] = v.drop("nbnd")
                print(f"dropping nbnd dimension from the {k} output datasets")
            # else:  # this is a no-op
            #     outputs[k] = v
            #     print(f"No nbnd dim in {k}")

        return outputs

    def _reshape_raw_data(self, inputs: Dict[str, xr.Dataset], outputs: Dict[str, xr.Dataset]):
        X, Y = [], []
        w = self.hparams.window
        mean_over_mems = self.hparams.mean_over_ensemble
        for i, (ds_key, input_xr) in enumerate(inputs.items()):
            if mean_over_mems or "member" not in input_xr.dims:
                print(input_xr.dims)
                print(outputs[ds_key].dims)
                assert "member" not in input_xr.dims and "member" not in outputs[ds_key].dims
                transpose_dims = ["time", "variable", "latitude", "longitude"]
            else:
                transpose_dims = ["time", "member", "variable", "latitude", "longitude"]
            output_xr = outputs[ds_key].transpose(*[d for d in transpose_dims if d != "variable"])
            X_np = input_xr.to_array().transpose(*transpose_dims).data
            Y_np = np.stack([output_xr[self.ovar_to_var_id[ovar]].data for ovar in self.output_vars], axis=-1)
            time_length = X_np.shape[0]

            # No historical data for right now TODO add when we get the historical data for daily resolution
            index_range = range(0, time_length - w + 1)
            X += [np.array([X_np[i : i + w] for i in index_range])]
            Y += [np.array([Y_np[i + w - 1] for i in index_range])]

        if mean_over_mems or "member" not in input_xr.dims:
            Y = [rearrange(y, "time lat lon var -> time var lat lon") for y in Y]
        else:
            # Allow for different number of members in different simulations
            X = [rearrange(x, "time window member var lat lon -> (time member) window var lat lon") for x in X]
            Y = [rearrange(y, "time member lat lon var -> (time member) var lat lon") for y in Y]

        # Concatenate all sequences across time (or time x ensemble) dimension
        X = np.concatenate(X, axis=0)
        Y = np.concatenate(Y, axis=0)
        print("X shape: ", X.shape, "Y shape: ", Y.shape, "window: ", w)
        # bring into shape (time, variable, lat, lon)
        assert X.shape[1] == w and X.shape[2] == len(self.hparams.normalize_vars)
        assert Y.shape[1] == len(self.output_vars)
        return {"inputs": X, "targets": Y}

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

    Each sample will be retrieved by indexing tensors along the first dimension.
    Will interpolate the input data to daily resolution on runtime during __getitem__

    Args:
        *tensors (Tensor): tensors that have the same size of the first dimension.
            - Shape --> [time, variable, lat, lon]
        *start_datetime (str): start datetime of the dataset
        *interpolation_type (str): type of interpolation to use
    """

    tensors: Dict[str, Tensor]

    def __init__(
        self,
        tensors: Dict[str, Tensor] | Dict[str, np.ndarray],
        start_datetime: str,
        dataset_id: str = "",
        interpolation_type: str = "middle",
    ):
        tensors = {
            key: torch.from_numpy(tensor.copy()).float() if isinstance(tensor, np.ndarray) else tensor
            for key, tensor in tensors.items()
        }
        any_tensor = next(iter(tensors.values()))
        self.dataset_size = any_tensor.size(0)
        assert all(self.dataset_size == tensor.size(0) for tensor in tensors.values()), "Size mismatch between tensors"
        self.tensors = tensors
        self.dataset_id = dataset_id
        self.start_datetime = datetime.strptime(start_datetime, "%Y-%m-%d %H:%M:%S")
        self.interpolation_type = interpolation_type

    def __getitem__(self, index) -> Dict[str, Tensor]:
        # tensor shape (time, variable, lat, lon)
        tensor_date = self.start_datetime + timedelta(days=index - 1)
        yearly_reso_idx = tensor_date.year - self.start_datetime.year
        # Interpolate the input data to daily resolution of X
        inputs = timeInterpolateMiddle(self.tensors["inputs"], tensor_date, yearly_reso_idx)
        outputs = self.tensors["targets"][index]

        return {"inputs": inputs, "targets": outputs}

    def __len__(self):
        return self.dataset_size


if __name__ == "__main__":
    dm = ClimateBenchDailyDataModule(
        window=1,
        output_vars=["tas", "pr"],
        mean_over_ensemble=False,
        simulations=["ssp126", "ssp370", "ssp585"],
    )
    dm.setup()
    print(dm._data_train[0]["inputs"].shape, dm._data_train[0]["targets"].shape)
    print("Dataset sizes:", len(dm._data_train), len(dm._data_val), len(dm._data_test))
