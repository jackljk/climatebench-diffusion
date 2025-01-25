import logging
import os
from typing import Callable, List, Optional

import netCDF4
import numpy as np
import torch
from torch.utils.data import Dataset


def load_series_data_sequential(idx: int, ds: netCDF4.MFDataset, horizon: int, names: List[str]):
    # flip the lat dimension so that it is increasing
    arrays = {
        n: torch.as_tensor(np.flip(ds.variables[n][idx : idx + horizon + 1, :, :], axis=-2).copy()) for n in names
    }
    return {"dynamics": arrays}


def load_series_data_direct(idx: int, ds: netCDF4.MFDataset, horizon: int, names: List[str]):
    # flip the lat dimension so that it is increasing
    arrays = {
        n: torch.as_tensor(
            np.flip(
                np.stack([ds.variables[n][idx, :, :], ds.variables[n][idx + horizon, :, :]], axis=0), axis=-2
            ).copy()
        )
        for n in names
    }
    return {"data": arrays}


def load_series_data_multistep_randomized(
    idx: int,
    ds: netCDF4.MFDataset,
    horizon: int,
    names: List[str],
    random_timestep: Optional[int] = None,
    is_forcing: bool = False,
):
    random_timestep = random_timestep or np.random.randint(1, horizon + 1)  # in [1, horizon]
    arrays = {
        n: torch.as_tensor(
            np.flip(
                np.stack(
                    [
                        ds.variables[n][idx, :, :],  # first step/initial conditions
                        ds.variables[n][idx + random_timestep, :, :],  # random step
                    ],
                    axis=0,
                ),
                axis=-2,
            ).copy()
        )
        for n in names
    }
    return {"data": arrays, "random_timestep": torch.as_tensor(random_timestep, dtype=torch.long)}


def load_series_data_multistep_interpolation(
    idx: int,
    ds: netCDF4.MFDataset,
    horizon: int,
    names: List[str],
    random_timestep: Optional[int] = None,
    is_forcing: bool = False,
):
    # Note that for interpolation, the condition/forcings willl only be used from the interpolation step
    random_timestep = random_timestep or np.random.randint(1, horizon)  # in [1, horizon - 1]

    def get_time_data(name):
        if False:  # is_forcing:
            return ds.variables[name][idx + random_timestep, :, :]
        else:
            return np.stack(
                [
                    ds.variables[name][idx, :, :],  # first step/initial conditions
                    ds.variables[name][idx + random_timestep, :, :],  # random step
                    ds.variables[name][idx + horizon, :, :],  # last step
                ],
                axis=0,
            )

    arrays = {n: torch.as_tensor(np.flip(get_time_data(n), axis=-2).copy()) for n in names}
    return {"data": arrays, "random_timestep": torch.as_tensor(random_timestep)}


class FV3GFSDataset(Dataset):
    def __init__(
        self,
        path: str,
        in_names: List[str],
        out_names: List[str],
        all_names: List[str],
        forcing_names: List[str],
        horizon: int,
        multistep_strategy: Optional[str] = None,
        n_samples: Optional[int] = None,
        min_idx_shift: int = 0,
        forcing_packer: Optional[Callable] = None,
        forcing_normalizer: Optional[Callable] = None,
        split_id: Optional[str] = None,
    ):
        assert n_samples is None or n_samples > 0, f"Invalid n_samples {n_samples}"
        assert min_idx_shift >= 0, f"Invalid min_idx_shift {min_idx_shift}"
        self.names = all_names
        self.in_names = in_names
        self.out_names = out_names
        self.in_or_out_names = list(set(all_names) - set(forcing_names))
        self.forcing_names = forcing_names if len(forcing_names) > 0 else None
        self.forcing_packer = forcing_packer
        if self.forcing_packer is not None:
            assert self.forcing_packer.axis is not None, f"Forcing packer {self.forcing_packer} must have axis set"
        self.forcing_normalizer = forcing_normalizer
        self.horizon = horizon
        self.n_in_channels = len(self.in_names)
        self.n_out_channels = len(self.out_names)
        self.multistep_strategy = multistep_strategy
        self.path = path
        self.full_path = os.path.join(path, "*.nc")
        self.split_id = split_id

        self._get_files_stats()
        self.min_idx_shift = min_idx_shift  # Used to shift the indices to avoid overlap between val & test
        if n_samples is not None:
            self.n_samples_total = n_samples  # Hardcodes max number of samples

        if multistep_strategy != "sequential":
            assert horizon > 0, f"Invalid horizon {horizon} for multistep strategy {multistep_strategy}"

        if multistep_strategy == "sequential":
            self.load_series_data = load_series_data_sequential
        elif multistep_strategy == "random":
            self.load_series_data = load_series_data_multistep_randomized
        elif multistep_strategy == "interpolation":
            self.load_series_data = load_series_data_multistep_interpolation
        elif multistep_strategy in [None, "direct"]:
            self.load_series_data = load_series_data_direct
        else:
            raise ValueError(f"Unknown multistep strategy {multistep_strategy}")

        if multistep_strategy == "sequential":
            self.main_data_key = "dynamics"
        else:
            self.main_data_key = "data"

        self.shared_kwargs = dict(horizon=horizon, ds=self.ds)
        # print(f'Initialized dataset with {len(self)} samples', horizon, split_id)

    def _get_files_stats(self):
        expected_vars = [
            "DLWRFsfc",
            "DSWRFsfc",
            "DSWRFtoa",
            "LHTFLsfc",
            "PRATEsfc",
            "PRESsfc",
            "SHTFLsfc",
            "ULWRFsfc",
            "ULWRFtoa",
            "USWRFsfc",
            "USWRFtoa",
            "air_temperature_0",
            "air_temperature_1",
            "air_temperature_2",
            "air_temperature_3",
            "air_temperature_4",
            "air_temperature_5",
            "air_temperature_6",
            "air_temperature_7",
            "eastward_wind_0",
            "eastward_wind_1",
            "eastward_wind_2",
            "eastward_wind_3",
            "eastward_wind_4",
            "eastward_wind_5",
            "eastward_wind_6",
            "eastward_wind_7",
            "grid_xt",
            "grid_yt",
            "land_sea_mask",
            "northward_wind_0",
            "northward_wind_1",
            "northward_wind_2",
            "northward_wind_3",
            "northward_wind_4",
            "northward_wind_5",
            "northward_wind_6",
            "northward_wind_7",
            "pressure_thickness_of_atmospheric_layer_0",
            "pressure_thickness_of_atmospheric_layer_1",
            "pressure_thickness_of_atmospheric_layer_2",
            "pressure_thickness_of_atmospheric_layer_3",
            "pressure_thickness_of_atmospheric_layer_4",
            "pressure_thickness_of_atmospheric_layer_5",
            "pressure_thickness_of_atmospheric_layer_6",
            "pressure_thickness_of_atmospheric_layer_7",
            "specific_total_water_0",
            "specific_total_water_1",
            "specific_total_water_2",
            "specific_total_water_3",
            "specific_total_water_4",
            "specific_total_water_5",
            "specific_total_water_6",
            "specific_total_water_7",
            "surface_temperature",
            "tendency_of_total_water_path",
            "tendency_of_total_water_path_due_to_advection",
            "time",
            "total_water_path",
        ]
        logging.info(f"Opening data at {self.full_path}")
        self.ds = netCDF4.MFDataset(self.full_path)
        self.ds.set_auto_mask(False)
        # minus one since don't have an output for the last step
        self.n_samples_total = len(self.ds.variables["time"][:]) - self.horizon
        # provided ERA5 dataloader gets the "wrong" x/y convention (x is lat, y is lon)
        # so we follow that convention here for consistency
        if "grid_xt" in self.ds.variables:
            self.img_shape_x = len(self.ds.variables["grid_yt"][:])
            self.img_shape_y = len(self.ds.variables["grid_xt"][:])
        else:
            self.img_shape_x = len(self.ds.variables["lat"][:])
            self.img_shape_y = len(self.ds.variables["lon"][:])
        logging.info(f"Found {self.n_samples_total} samples.")
        logging.info(f"Image shape is {self.img_shape_x} x {self.img_shape_y}.")
        missing_vars = set(expected_vars) - set(self.ds.variables)
        if len(missing_vars) > 0:
            raise ValueError(f"Missing variables: {missing_vars}")
        elif len(set(self.ds.variables) - set(expected_vars)) > 0:
            logging.warning(f"Found unexpected variables: {set(self.ds.variables) - set(expected_vars)}")
        # logging.info(f"Following variables are available: {list(self.ds.variables)}.")

    def __len__(self):
        return self.n_samples_total

    def __getitem__(self, idx):
        idx = idx + self.min_idx_shift  # Shift indices to avoid overlap between val & test
        data = self.load_series_data(idx=idx, names=self.in_or_out_names, **self.shared_kwargs)
        # data_shape = data[list(data.keys())[0]].shape
        # print(f'Loaded data with shape {data_shape}')
        # data = TensorDict(data, batch_size=data_shape)
        if self.forcing_names is not None:
            if self.multistep_strategy in ["random", "interpolation"]:
                fkwargs = {"random_timestep": data["random_timestep"], "is_forcing": True}
            else:
                fkwargs = {}
            forcings = self.load_series_data(idx=idx, names=self.forcing_names, **self.shared_kwargs, **fkwargs)[
                self.main_data_key
            ]
            forcings = self.forcing_packer.pack(self.forcing_normalizer.normalize(forcings))
            data["condition"] = forcings
        return data
