# General imports/constants
from typing import Dict, Optional, Sequence, Union

import cftime
import numpy as np
import xarray as xr

from src.utilities.utils import get_files, get_logger
from src.utilities.normalization import StandardNormalizer
import torch

log = get_logger(__name__)

# imports/constants DM


# imports/constants ds


NO_DAYS_IN_YEAR = 365
NO_DAYS_IN_MONTH = 30

statistics = {
    "standard_new": {
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
    },
    "standard": {
        "tas_mean": torch.tensor(279.7749),
        "tas_std": torch.tensor(29.7625),
        "pr_mean": torch.tensor(2.8494e-05),
        "pr_std": torch.tensor(5.3200e-05),
    },
}


def get_statistics(
    output_vars: list[str] = ["tas"], normalization_type: str = "standard_new", precip_transform: Optional[str] = None, var_to_transform_name: Dict[str, str] = dict()
):
    if normalization_type == "standard":
        assert precip_transform is None, "Use normalization_type='standard_new' instead."
        # Compute the mean and std of the output variables
        if len(output_vars) == 1 and output_vars[0] == "tas":
            data_mean_act = statistics["standard"]["tas_mean"]
            data_std_act = statistics["standard"]["tas_std"]
        elif output_vars == ["tas", "pr"]:
            data_mean_act = {"tas": statistics["standard"]["tas_mean"], "pr": statistics["standard"]["pr_mean"]}
            data_std_act = {"tas": statistics["standard"]["tas_std"], "pr": statistics["standard"]["pr_std"]}
        else:
            raise ValueError(
                f"Normalization type {normalization_type} not supported for output variables {output_vars}"
            )
    elif normalization_type == "standard_new":
        data_mean_act, data_std_act = dict(), dict()
        for ovar in output_vars:
            if ovar == "pr" and precip_transform is not None:
                pr_transform = precip_transform
                if "only" in pr_transform:
                    # Only transform the precipitation data but do not standardize it
                    log.info(f"Only transforming the precipitation data with {pr_transform}")
                    pr_transform = pr_transform.replace("_only", "")
                    mean_pr, std_pr = 0.0, 1.0
                else:
                    mean_pr = statistics["standard_new"][f"{pr_transform}_pr_mean"]["weighted"]
                    std_pr = statistics["standard_new"][f"{pr_transform}_pr_std"]["weighted"]
                var_to_transform_name[ovar] = pr_transform
            else:
                mean_pr = statistics["standard_new"][f"{ovar}_mean"]["weighted"]
                std_pr = statistics["standard_new"][f"{ovar}_std"]["weighted"]
            data_mean_act[ovar] = torch.tensor(mean_pr)
            data_std_act[ovar] = torch.tensor(std_pr)
    else:
        raise ValueError(f"Invalid value for normalization_type: {self.hparams.normalization_type}")
    return data_mean_act, data_std_act


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
            if any(dim in output_xr[key].dims for dim in ["y", "x", "lat", "lon"]):
                # Create dictionary of dimension names to rename
                rename_dict = {"y": "latitude", "x": "longitude", "lat": "latitude", "lon": "longitude"}

                # Only include the dimensions that actually exist in the dataset
                rename_dict = {k: v for k, v in rename_dict.items() if k in output_xr[key].dims}

                # Apply the renaming if there's anything to rename
                if rename_dict:
                    output_xr[key] = output_xr[key].rename(rename_dict)
            # Convert pr to mm/day and rename lon and lat to longitude and latitude
            if False:  # "pr" in output_vars:
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
        # Rename the daily variables to be consistent with the yearly data
        if any(dim in output_xr.dims for dim in ["y", "x", "lat", "lon"]):
            # Create dictionary of dimension names to rename
            rename_dict = {"y": "latitude", "x": "longitude", "lat": "latitude", "lon": "longitude"}

            # Only include the dimensions that actually exist in the dataset
            rename_dict = {k: v for k, v in rename_dict.items() if k in output_xr.dims}

            # Apply the renaming if there's anything to rename
            if rename_dict:
                output_xr = output_xr.rename(rename_dict)

        if False:  # "pr" in output_vars:
            # Convert pr to mm/day and rename lon and lat to longitude and latitude
            log.info(f"Converting pr and pr90 to mm/day for {simulation}")
            output_xr["pr"] *= 86400  # less efficient: output_xr.assign({"pr": output_xr.pr * 86400})  # no pr90
            log.info(f"Finished converting pr and pr90 to mm/day for {simulation}")

    return output_xr


def handle_ensemble(
    output_xr: xr.Dataset,
    mean_over_ensemble: Optional[Union[str, bool]],
    simulation: str,
    num_ensemble: Union[int, str],
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
    elif mean_over_ensemble == "all":
        log.info(f"Using all {output_xr.sizes['member_id']} ensemble members for {simulation}")
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
    solar_experiment: Optional[str] = None,
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

    if solar_experiment == "G6Solar":
        # get the file with G6Solar
        rsdt[solar_experiment] = xr.open_dataset(data_path + "/rsdt_Amon_CESM2-WACCM_G6solar.nc").compute()

    # always load rsdt ssp126
    rsdt_path = [path for path in rsdt_paths if "ssp126" in path]
    assert len(rsdt_path) == 1, f"Expected 1 file for ssp126, found {rsdt_paths=}"
    rsdt_path = rsdt_path[0]
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

    # Standardize the rsdt data using training data i.e "ssp data"
    rsdt_ssp_flattened = rsdt["ssp"].rsdt.data.reshape(-1)
    rsdt_ssp_mean, rsdt_ssp_std = rsdt_ssp_flattened.mean(), rsdt_ssp_flattened.std()

    for k, v in rsdt.items():
        rsdt[k] = normalize_data(v, {"rsdt": {"mean": rsdt_ssp_mean, "std": rsdt_ssp_std}})

    if solar_experiment:
        # remove the ssp from the dictionary
        rsdt.pop("ssp")
        # rename G6Solar lat lon to x, y
        rsdt[solar_experiment] = rsdt[solar_experiment].rename({"lat": "y", "lon": "x"})

    return rsdt


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
    DAYS_FROM_MIDDLE = (MIDDLE_OF_MONTH - index_datetime).days if DAY < 15 else (index_datetime - MIDDLE_OF_MONTH).days

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

        interpolated_values = ((NO_DAYS_IN_MONTH - DAYS_FROM_MIDDLE) / NO_DAYS_IN_MONTH * curr).squeeze() + (
            DAYS_FROM_MIDDLE / NO_DAYS_IN_MONTH * prev_month
        ).squeeze()

    return interpolated_values
