from __future__ import annotations

from os.path import join
from typing import Dict, Optional, Sequence

import numpy as np
import xarray as xr
from einops import rearrange

from src.datamodules.abstract_datamodule import BaseDataModule
from src.datamodules.torch_datasets import MyTensorDataset
from src.evaluation.aggregators.main import OneStepAggregator
from src.utilities.packer import Packer
from src.utilities.utils import raise_error_if_invalid_value


len_historical = 165


def get_raw_data(
    data_path: str,
    simulations: Sequence[str],
    stage: str,
    mean_over_ensemble: bool = False,
) -> (Dict[str, xr.Dataset], Dict[str, xr.Dataset]):
    X_train, Y_train = dict(), dict()
    for i, simu in enumerate(simulations):
        input_name = "inputs_" + simu + ".nc"
        output_name = "outputs_" + simu + ".nc"

        if "piControl" in simu:
            # remove first 100 years of piControl
            print("Removing first 100 years of piControl", simu)
            output_xr = xr.open_dataset(join(data_path, output_name)).drop(["quantile"]).isel(time=slice(100, None))
            # input xarray is all zeroes for all input vars ('CO2', 'CH4', 'SO2', 'BC')
            lat, lon, time = [output_xr.coords[dim].values for dim in ["lat", "lon", "time"]]
            input_xr = xr.Dataset(
                data_vars={
                    "CO2": (("time",), np.zeros((len(time),))),
                    "CH4": (("time",), np.zeros((len(time),))),
                    "SO2": (("time", "lat", "lon"), np.zeros((len(time), len(lat), len(lon)))),
                    "BC": (("time", "lat", "lon"), np.zeros((len(time), len(lat), len(lon)))),
                },
                coords={"time": time, "lat": lat, "lon": lon},
            )
        elif "ssp" not in simu or stage in ["test", "predict"]:
            # Just load hist data in these cases 'hist-GHG' and 'hist-aer'
            input_xr = xr.open_dataset(join(data_path, input_name))
            output_xr = xr.open_dataset(join(data_path, output_name)).drop(["quantile"])  # 3 members
        else:
            # Concatenate with historical data in the case of scenario 'ssp126', 'ssp370' and 'ssp585'
            print("Concatenating historical and future data for scenario " + simu)
            # load inputs
            input_xr = xr.open_mfdataset(
                [join(data_path, "inputs_historical.nc"), join(data_path, input_name)]
            ).compute()
            # load outputs
            output_xr = (
                xr.concat(
                    [
                        xr.open_dataset(join(data_path, "outputs_historical.nc")),
                        xr.open_dataset(join(data_path, output_name)),
                    ],
                    dim="time",
                )
                .compute()
                .drop(["quantile"])
            )
        output_xr = output_xr.assign({"pr": output_xr.pr * 86400, "pr90": output_xr.pr90 * 86400}).rename(
            {"lon": "longitude", "lat": "latitude"}
        )
        if "lon" in input_xr.dims:
            input_xr = input_xr.rename({"lon": "longitude", "lat": "latitude"})

        if mean_over_ensemble:
            output_xr = output_xr.mean(dim="member")
            transpose_dims = ["time", "latitude", "longitude"]
        else:
            # drop each member where there's a nan in the output
            check_nan_ds = output_xr.diurnal_temperature_range
            if "ssp" in simu:  # drop historical part of ssp runs
                check_nan_ds = check_nan_ds.isel(time=slice(len_historical, None))
            nan_mems = check_nan_ds.isnull().any(dim=["time", "latitude", "longitude"])
            print("Dropping members with nan in output: ", nan_mems.values)
            # copy input_xr for each member
            input_xr = xr.concat([input_xr] * output_xr.sizes["member"], dim="member")
            transpose_dims = ["time", "member", "latitude", "longitude"]

        output_xr = output_xr.transpose(*transpose_dims)
        X_train[simu] = input_xr
        Y_train[simu] = output_xr
    return X_train, Y_train


def get_mean_std_of_variables(
    training_set: Dict[str, xr.Dataset], variables: Sequence[str] | None
) -> Dict[str, Dict[str, float]]:
    #                  simulations: Sequence[str] = ('ssp126', 'ssp370', 'ssp585', 'hist-GHG', 'hist-aer'),
    #                  skip_hist = ds_key in ['ssp126', 'ssp370']
    if variables is None:
        # use all variables in the dataset
        variables = list(training_set.values())[0].data_vars.keys()
    # don't compute stats twice on ssp runs for historical part
    ssps = [k for k in training_set.keys() if "ssp" in k]
    remove_hist_keys = ssps[1:] if len(ssps) > 1 else []
    other_keys = [ssps[0]] + [k for k in training_set.keys() if k not in ssps]
    var_to_meanstd = {}
    for var in variables:
        array = np.concatenate(
            [training_set[k][var].data.reshape(-1) for k in other_keys]
            + [training_set[k][var].isel(time=slice(len_historical, None)).data.reshape(-1) for k in remove_hist_keys],
            axis=0,
        )
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


class ClimateBenchDataModule(BaseDataModule):
    def __init__(
        self,
        data_dir: str = "~/data/climatebench",
        simulations: Sequence[str] = ("ssp126", "ssp370", "ssp585", "hist-GHG", "hist-aer"),
        sim_validation: str = "ssp370",
        validation_size: int = 45,
        normalize_vars: Sequence[str] = ("CO2", "CH4", "SO2", "BC"),
        window: int = 10,  # == slider
        output_vars: Sequence[str] | str = "tas",
        mean_over_ensemble: bool = True,
        **kwargs,
    ):
        self.output_vars = [output_vars] if isinstance(output_vars, str) else output_vars
        super().__init__(data_dir=data_dir, **kwargs)
        self.save_hyperparameters()
        self.var_to_meanstd: Dict[str, Dict[str, float]] | None = None
        self.ovar_to_var_id = {
            "tas": "tas",
            "dtr": "diurnal_temperature_range",
            "pr": "pr",
            "pr90": "pr90",  # pr90 is the 90th percentile of daily precipitation
        }
        channel_axis = -3
        # The out packer takes care of mapping each output dimension to the correct variable
        self.out_packer = Packer(self.output_vars, axis=channel_axis)

    def _check_args(self):
        super()._check_args()
        possible_ovars = ["tas", "diurnal_temperature_range", "pr", "pr90", "dtr"]
        for i, ovar in enumerate(self.output_vars):
            raise_error_if_invalid_value(ovar, possible_ovars, "output_vars[i]")

    @property
    def test_set_names(self) -> Sequence[str]:
        return ["ssp245", "ssp245_2080-2100"]

    def setup(self, stage: Optional[str] = None):
        # Compute mean and std of variables
        if self.var_to_meanstd is None or stage in ["fit", "validate", None]:
            X_train, Y_train = get_raw_data(
                self.hparams.data_dir,
                self.hparams.simulations,
                stage="fit",
                mean_over_ensemble=self.hparams.mean_over_ensemble,
            )
            # split validation set from training set
            sim_val = self.hparams.sim_validation
            X_val = X_train[sim_val].isel(time=slice(-self.hparams.validation_size, None))
            Y_val = Y_train[sim_val].isel(time=slice(-self.hparams.validation_size, None))
            X_train[sim_val] = X_train[sim_val].isel(time=slice(None, -self.hparams.validation_size))
            Y_train[sim_val] = Y_train[sim_val].isel(time=slice(None, -self.hparams.validation_size))
            print(f"Validating on {sim_val} from {X_val.time[0].item()} to {X_val.time[-1].item()}")

        if self.var_to_meanstd is None:
            # Compute mean and std of variables in training set
            self.var_to_meanstd = get_mean_std_of_variables(X_train, self.hparams.normalize_vars)
        # Normalize data
        if stage in ["fit", "validate", None]:
            X_train = {k: normalize_data(x, self.var_to_meanstd) for k, x in X_train.items()}
            X_val = normalize_data(X_val, self.var_to_meanstd)
            np_dset_train = self._reshape_raw_data(X_train, Y_train)
            np_dset_val = self._reshape_raw_data({"val": X_val}, {"val": Y_val})
        else:
            np_dset_train = np_dset_val = None
        if stage in ["test", "predict", None]:
            X_test, Y_test = get_raw_data(self.hparams.data_dir, ["ssp245"], stage="test", mean_over_ensemble=True)
            X_test = {k: normalize_data(x, self.var_to_meanstd) for k, x in X_test.items()}
            np_dset_test = self._reshape_raw_data(X_test, Y_test)
            np_dset_test = [np_dset_test] + [
                self._reshape_raw_data(
                    {k: v.sel(time=slice("2080", "2100")) for k, v in X_test.items()},
                    {k: v.sel(time=slice("2080", "2100")) for k, v in Y_test.items()},
                )
            ]
        else:
            np_dset_test = None

        ds_splits = {"train": np_dset_train, "val": np_dset_val, "test": np_dset_test, "predict": np_dset_test}
        for split, numpy_tensors in ds_splits.items():
            if numpy_tensors is None:
                continue
            if isinstance(numpy_tensors, (list, tuple)):
                setattr(self, f"_data_{split}", [MyTensorDataset(t, dataset_id=split) for t in numpy_tensors])
            else:
                # Create the pytorch tensor dataset
                tensor_ds = MyTensorDataset(numpy_tensors, dataset_id=split)
                # Save the tensor dataset to self._data_{split}
                setattr(self, f"_data_{split}", tensor_ds)

        # Print sizes of the datasets (how many examples)
        self.print_data_sizes(stage)

    def _reshape_raw_data(self, inputs: Dict[str, xr.Dataset], outputs: Dict[str, xr.Dataset]):
        X, Y = [], []
        w = self.hparams.window
        mean_over_mems = self.hparams.mean_over_ensemble
        for i, (ds_key, input_xr) in enumerate(inputs.items()):
            if mean_over_mems or "member" not in input_xr.dims:
                assert "member" not in input_xr.dims and "member" not in outputs[ds_key].dims
                transpose_dims = ["time", "variable", "latitude", "longitude"]
            else:
                transpose_dims = ["time", "member", "variable", "latitude", "longitude"]
            skip_hist = ds_key in ["ssp126", "ssp370"]
            output_xr = outputs[ds_key].transpose(*[d for d in transpose_dims if d != "variable"])
            X_np = input_xr.to_array().transpose(*transpose_dims).data
            print(f"Shape of X_np[{i}]: {X_np.shape} for {ds_key}")
            Y_np = np.stack([output_xr[self.ovar_to_var_id[ovar]].data for ovar in self.output_vars], axis=-1)
            print(f"Shape of Y_np[{i}]: {Y_np.shape} for {ds_key}")
            time_length = X_np.shape[0]
            # If skip_historical, the first sequence created has as last/target element the first scenario data point
            if skip_hist:
                index_range = range(len_historical - w + 1, time_length - w + 1)
            # Else, go through the whole dataset historical + scenario (doesn't matter for 'hist-GHG' and 'hist_aer')
            else:
                index_range = range(0, time_length - w + 1)
            X += [np.array([X_np[i : i + w] for i in index_range])]
            Y += [np.array([Y_np[i + w - 1] for i in index_range])]

        if mean_over_mems or "member" not in input_xr.dims:
            for i, y in enumerate(Y):
                print(f"Shape of Y_train[{i}]: {y.shape}")
                print(y)

            Y = [rearrange(y, "time lat lon var -> time var lat lon") for y in Y]
        else:
            # Allow for different number of members in different simulations
            X = [rearrange(x, "time window member var lat lon -> (time member) window var lat lon") for x in X]
            Y = [rearrange(y, "time member lat lon var -> (time member) var lat lon") for y in Y]

        # Concatenate all sequences across time (or time x ensemble) dimension
        X = np.concatenate(X, axis=0)
        Y = np.concatenate(Y, axis=0)
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
        device=None,
        verbose: bool = True,
        save_to_path: str = None,
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


if __name__ == "__main__":
    dm = ClimateBenchDataModule(
        window=1,
        output_vars=["tas", "pr"],
        mean_over_ensemble=False,
        simulations=["ssp126", "ssp370", "ssp585", "piControl", "hist-GHG", "hist-aer"],
    )
    dm.setup()
    print(dm._data_train[0]["inputs"].shape, dm._data_train[0]["targets"].shape)
    print("Dataset sizes:", len(dm._data_train), len(dm._data_val), len(dm._data_test))
# Dataset sizes: 2088 123 82 (window=5, mean_over_ensemble=False)
# Dataset sizes: 2124 135 86 (window=1, mean_over_ensemble=False)
# Dataset sizes: 696 41 82 (window=5, mean_over_ensemble=True)
# Dataset sizes: 708 45 86 (window=1, mean_over_ensemble=True)
