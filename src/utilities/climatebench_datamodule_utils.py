# General imports/constants
from typing import Dict, Optional, Sequence, Union

import numpy as np
import xarray as xr

from src.utilities.utils import get_files, get_logger


log = get_logger(__name__)

# imports/constants DM


# imports/constants ds
import cftime


NO_DAYS_IN_YEAR = 365
NO_DAYS_IN_MONTH = 30


# Datamodule Helpers
def standardize_output_xr(
    output_xr: Union[xr.Dataset, dict], simulation: str, output_vars: Union[Sequence[str], str]
) -> Union[xr.Dataset, dict]:
    """
    Standardize the output xr dataset by dropping the lat_bounds, lon_bounds and nbnd variables and renaming the
    daily variables to be consistent with the yearly data.

    Args:
        - output_xr: xarray.Dataset | dict
        - simulation: str (Current simulation being preprocessed)
        - output_vars: Sequence[str] | str

    Returns:
        - output_xr: xarray.Dataset | dict (Depending on the input type to handle Ensemble data)
    """
    if isinstance(output_xr, dict):
        # Drop lat_bound, lng_bounds and nbnd to make the data tranpose simpler
        vars_to_drop = [
            var for var in ["lat_bounds", "lon_bounds", "nbnd"] if var in output_xr[list(output_xr.keys())[0]].coords
        ]
        for key in output_xr.keys():
            output_xr[key] = output_xr[key].drop(vars_to_drop)

            # Rename the daily variables to be consistent with the yearly data
            if "y" or "x" in output_xr[key].dims:
                output_xr[key] = output_xr[key].rename({"y": "latitude", "x": "longitude"})
            # Convert pr to mm/day and rename lon and lat to longitude and latitude
            if "pr" in output_vars:
                log.info(f"Converting pr and pr90 to mm/day for {simulation}")
                output_xr[key]["pr"] *= 86400
                log.info(f"Finished converting pr and pr90 to mm/day for {simulation}")
    else:
        # Drop lat_bound, lng_bounds and nbnd to make the data tranpose simpler
        vars_to_drop = [
            var for var in ["lat_bounds", "lon_bounds", "nbnd"] if var in output_xr.coords
        ]  # Check if the variable exists
        output_xr = output_xr.drop(vars_to_drop)

        # Rename the daily variables to be consistent with the yearly data
        if "y" or "x" in output_xr.dims:
            output_xr = output_xr.rename({"y": "latitude", "x": "longitude"})

        if "pr" in output_vars:
            # Convert pr to mm/day and rename lon and lat to longitude and latitude
            log.info(f"Converting pr and pr90 to mm/day for {simulation}")
            output_xr["pr"] *= 86400  # less efficient: output_xr.assign({"pr": output_xr.pr * 86400})  # no pr90
            log.info(f"Finished converting pr and pr90 to mm/day for {simulation}")

    return output_xr


def handle_ensemble(
    output_xr: xr.Dataset, mean_over_ensemble: Optional[Union[str, bool]], simulation: str, num_ensemble: Union[int ,str]
) -> Union[xr.Dataset, dict]:
    """
    Handle ensemble data by either averaging over the ensemble, selecting the first ensemble member or stacking the
    ensemble members.

    Args:
        - output_xr: xarray.Dataset
        - mean_over_ensemble: bool | str
        - simulation: str

    Returns:
        - output_xr: xarray.Dataset | dict (Depending on the input type to handle Ensemble data)
    """
    if mean_over_ensemble is True:
        log.info(f"Average over ensemble for {simulation}. Ds.dims: {output_xr.dims}")
        output_xr = output_xr.mean(dim="member_id")
        log.info(f"Finished averaging over ensemble for {simulation}")
    elif mean_over_ensemble == "first":
        log.info(f"Selecting first ensemble member for {simulation}")
        output_xr = output_xr.isel(member_id=0)
    elif mean_over_ensemble == "stack":
        log.info(f"Stacking ensembles for {simulation}")
        # make dict of {simulation_ensem: xr}
        if num_ensemble == "all":
            output_xr = {
                f"{simulation}_ensemble_{i}": output_xr.isel(member_id=i) for i in range(output_xr.sizes["member_id"])
            }
        else:
            if num_ensemble > output_xr.sizes["member_id"]:
                raise ValueError(f"num_ensemble {num_ensemble} is greater than the number of ensemble members")
            output_xr = {f"{simulation}_ensemble_{i}": output_xr.isel(member_id=i) for i in range(num_ensemble)}
    else:
        log.info(f"Ensemble data not handled for {simulation}")
        raise KeyError(f"Option {mean_over_ensemble} not supported for ensemble handling")

    return output_xr


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


# def get_raw_data(
#     data_path: str,
#     simulations: Sequence[str],
#     stage: str,
#     mean_over_ensemble: bool = False,
#     Debug_dataset_size: int = None,
#     scale_inputs: str = None,
# ) -> tuple(Dict[str, xr.Dataset], Dict[str, xr.Dataset]):
#     """
#     Load the raw data from the given path and return it as a dictionary of xarray datasets.

#     Avaliable daily output simulations are:
#     - 'output_ssp126_daily'
#     - 'output_ssp245_daily'
#     - 'output_ssp370_daily'
#     - 'output_ssp585_daily'
#     - 'outputs_historical_daily_raw'

#     Avaliable Input simulations are yearly:
#     - 'input_hist-GHG'
#     - 'inputs_abrupt-4xCO2'
#     - 'inputs_1pctCO2'
#     - 'inputs_hist-aer'
#     - 'inputs_historical'
#     - 'inputs_ssp126'
#     - 'inputs_ssp245'
#     - 'inputs_ssp370'
#     - 'inputs_ssp370-lowNTCF
#     - 'inputs_ssp585'

#     Args:
#     - data_path: Path to the directory containing the data
#     - simulations: List of simulations to load
#     - stage: The stage of the data to load. Either 'train' or 'validation'
#     - mean_over_ensemble: If True, the output data will be averaged over the ensemble dimension
#     - Debug_dataset_size: If not None, the size of the dataset to load
#     - scale_inputs: If not None, the type of scaling to apply to the input data (downscale or upscale)

#     Returns:
#     - X_train: Dictionary containing the input data for each simulation
#     - Y_train: Dictionary containing the output data for each simulation
#     """
#     X_train, Y_train = dict(), dict()
#     for i, simu in enumerate(simulations):
#         input_name = "inputs_" + simu + ".nc"
#         output_name = "outputs_" + simu + "_daily.nc"
#         if Debug_dataset_size is not None:
#             input_xr = (
#                 xr.open_dataset(join(data_path, input_name))
#                 .sel(time=slice("2015", str(2015 + Debug_dataset_size)))
#                 .compute()
#             )
#             output_xr = (
#                 xr.open_dataset(join(data_path, output_name))
#                 .sel(time=slice("2015", str(2015 + Debug_dataset_size)))
#                 .compute()
#             )
#         else:
#             log.info(f"Loading data from {data_path}")
#             input_xr = xr.open_dataset(join(data_path, input_name)).compute()
#             log.info(f"Loaded input data from {input_name}")
#             # output_xr = xr.open_dataset(join(data_path, output_name)).compute()
#             output_xr = xr.open_dataset(join(data_path, output_name), chunks={"member_id": -1}).compute()
#             log.info(f"Loaded output data from {output_name}")

#         # assert not mean_over_ensemble, "don't wanna explore using mean_over_ensemble."
#         if mean_over_ensemble:
#             log.info(f"Average over ensemble for {simu}. Ds.dims: {output_xr.dims}")
#             output_xr = output_xr.mean(dim="member_id")
#             log.info(f"Finished averaging over ensemble for {simu}")
#             # transpose_dims = ["time", "latitude", "longitude"]
#         else:
#             # drop each member where there's a nan in the output
#             print("Dataset data_vars: ", output_xr.data_vars)
#             # check_nan_ds = output_xr.tas
#             # nan_mems = check_nan_ds.isnull().any(dim=["time", "latitude", "longitude"])
#             # print("Dropping members with nan in output: ", nan_mems.values)
#             # copy input_xr for each ensemble member
#             input_xr = xr.concat([input_xr] * output_xr.sizes["member_id"], dim="member_id")
#             # transpose_dims = ["time", "member", "latitude", "longitude"]
#             output_xr = output_xr.rename({"member_id": "member"})
#             input_xr = input_xr.rename({"member_id": "member"})

#         # Drop lat_bound, lng_bounds and nbnd to make the data tranpose simpler
#         output_xr = output_xr.drop(["lon_bounds", "lat_bounds", "nbnd"])

#         # If the input data has lon and lat as dimensions, rename them to longitude and latitude
#         if "lon" in input_xr.dims:
#             input_xr = input_xr.rename({"lon": "longitude", "lat": "latitude"})

#         # Rename the daily variables to be consistent with the yearly data
#         if "y" or "x" in output_xr.dims:
#             output_xr = output_xr.rename({"y": "latitude", "x": "longitude"})
#         # Convert pr and pr90 to mm/day and rename lon and lat to longitude and latitude
#         log.info(f"Converting pr and pr90 to mm/day for {simu}")
#         output_xr["pr"] *= 86400  # less efficient: output_xr.assign({"pr": output_xr.pr * 86400})  # no pr90
#         log.info(f"Finished converting pr and pr90 to mm/day for {simu}")

#         # ! Note: Commented out for now
#         # output_xr = output_xr.transpose(*transpose_dims)
#         X_train[simu] = input_xr
#         Y_train[simu] = output_xr

#         # Match in/output Spatial resolution
#         if scale_inputs is not None:
#             log.info(f"Matching inputs res. to output res. for {simu}")
#             X_train[simu], Y_train[simu] = _scaleInterpolateLinear(
#                 X_train[simu], Y_train[simu], scale_type=scale_inputs
#             )

#     log.info(f"Finished pre-processing data for {simulations}")
#     return X_train, Y_train


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
    training_set: Dict[str, xr.Dataset], variables: Sequence[str] = None
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


# Torch Dataset Helpers
def yearlyInterpolator(
    input_xr: xr.Dataset,
    ssp_index_datetime: cftime.datetime,
    no_prev_year: bool = False,
    no_next_year: bool = False,
    type="middle",
) -> xr.Dataset:
    """
    Interpolate yearly climate data to daily resolution by linearly interpolating from the middle of the year.

    # NOTE: Only interpolation is supported for now

    # NOTE: E = July 2nd; ==> (364/365)*E_2009 + (1/365)*E_2010 for July 3rd 2009 formula
        For module climatebench_daily

    Args:
     - input_xr: xarray.Dataset
     - ssp_index_datetime: cftime.DatetimeNoLeap
     # Flags to indicate if there is no previous or next year
        - no_prev_year: bool
        - no_next_year: bool
     - type # TODO (Maybe add different interpolation types i.e. start end)

    Returns:
        - interpolated_values: xarray.Dataset
    """
    DAY = ssp_index_datetime.day
    MONTH = ssp_index_datetime.month
    YEAR = ssp_index_datetime.year
    JULY_2ND = cftime.DatetimeNoLeap(YEAR, 7, 2)
    DAYS_FROM_MIDDLE = (ssp_index_datetime - JULY_2ND).days

    try:
        # Get the values for the current year
        E_curr = input_xr.sel(time=YEAR)
    except Exception:
        # Use the nearest year if the current year is not available
        # Due to the lone 2101 in output, we can just use the 2100 input data
        E_curr = input_xr.sel(time=YEAR, method="nearest")

    # Passed July 2nd
    if MONTH > 7 or (MONTH == 7 and DAY > 2):
        try:
            # Get the values for the next year
            # NOTE: Interpolates with same year if there is no previous year
            E_next = input_xr.sel(time=YEAR + 1) if not no_next_year else input_xr.sel(time=YEAR)
        except Exception:
            # Get the values using the same year
            E_next = input_xr.sel(time=YEAR, method="nearest")  # Need to figure out why YEAR became 2101
        # Calculate the interpolated values
        interpolated_values = (
            NO_DAYS_IN_YEAR - DAYS_FROM_MIDDLE
        ) / NO_DAYS_IN_YEAR * E_curr + DAYS_FROM_MIDDLE / NO_DAYS_IN_YEAR * E_next
    else:
        try:
            # Get the values for the previous year
            # NOTE: Interpolates with same year if there is no next year
            E_prev = input_xr.sel(time=YEAR - 1) if not no_prev_year else input_xr.sel(time=YEAR)
        except Exception:
            # Use the same year for interpolation
            E_prev = input_xr.sel(time=YEAR, method="nearest")

        # Calculate the interpolated values
        interpolated_values = (
            NO_DAYS_IN_YEAR - DAYS_FROM_MIDDLE
        ) / NO_DAYS_IN_YEAR * E_prev + DAYS_FROM_MIDDLE / NO_DAYS_IN_YEAR * E_curr

    return interpolated_values


def monthlyInterpolator(input_xr: xr.Dataset, index_datetime: cftime.datetime) -> xr.Dataset:
    """
    Interpolate yearly climate data to daily resolution by linearly interpolating from the middle of the month.

    Note: For module climatebench_daily_modified/Middle of month set set 15th as the middle of the month

    Args:
        - input_xr: xarray.Dataset
        - index_datetime: cftime.DatetimeNoLeap
        # Flags to indicate if there is no previous or next month
        - no_prev_month: bool
        - no_next_month: bool
    """
    DAY = index_datetime.day
    MONTH = index_datetime.month
    YEAR = index_datetime.year
    MIDDLE_OF_MONTH = cftime.DatetimeNoLeap(YEAR, MONTH, 15)
    DAYS_FROM_MIDDLE = (index_datetime - MIDDLE_OF_MONTH).days

    try:
        # Get the values for the current month
        curr = input_xr.sel(time=f"{YEAR}-{MONTH:02d}")
    except Exception:
        # Use the nearest month if the current month is not available
        # Due to the lone 2101 in output, we can just use the 2100 input data
        try:
            if YEAR == 2101:
                curr = input_xr.sel(time="2100-12")
            else:
                curr = input_xr.sel(time=f"{YEAR-1}-{MONTH:02d}")
        except Exception:
            curr = input_xr.sel(time="2100-12")  # Need to figure out why YEAR became 2101
            print("Error occured during interpolation")
            print(f"Year: {YEAR}, Month: {MONTH}")

    # Interpolate the values
    if DAY > 15:
        try:
            # Calculate next month handling the edge case of December
            next_month_string = f"{YEAR}-{MONTH + 1:02d}" if MONTH < 12 else f"{YEAR + 1}-{1:02d}"
            # Get the values for the next month
            next_month = input_xr.sel(time=next_month_string)
        except Exception:
            # Use the same month for interpolation if the next month is not available (i.e. December 2100)
            next_month = input_xr.sel(time=f"{YEAR}-{MONTH:02d}")

        interpolated_values = ((NO_DAYS_IN_MONTH - DAYS_FROM_MIDDLE) / NO_DAYS_IN_MONTH * curr).squeeze() + (
            DAYS_FROM_MIDDLE / NO_DAYS_IN_MONTH * next_month
        ).squeeze()
    else:
        try:
            # Calculate previous month handling the edge case of January
            prev_month_string = f"{YEAR}-{MONTH - 1:02d}" if MONTH > 1 else f"{YEAR - 1}-{12:02d}"
            # Get the values for the previous month
            prev_month = input_xr.sel(time=prev_month_string)
        except Exception:
            # Use the same month for interpolation if the previous month is not available (i.e. January 2015)
            prev_month = input_xr.sel(time=f"{YEAR}-{MONTH:02d}")

        interpolated_values = ((NO_DAYS_IN_MONTH - DAYS_FROM_MIDDLE) / NO_DAYS_IN_MONTH * prev_month).squeeze() + (
            DAYS_FROM_MIDDLE / NO_DAYS_IN_MONTH * curr
        ).squeeze()

    return interpolated_values
