from typing import Any, Dict, List, Mapping, Optional

import cartopy.crs as ccrs
import cftime
import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr
from torch import Tensor

import wandb
from src.evaluation.metrics import root_mean_squared_error
from src.losses.losses import crps_ensemble
from src.utilities.utils import get_logger


log = get_logger(__name__)

metric_functions = {
    "rmse": root_mean_squared_error,
    "crps": crps_ensemble,
    "rmse_member_avg": root_mean_squared_error,
}


class TemporalMetricsAggregator:
    """
    Aggregator for temporal metrics
    """

    def __init__(
        self,
        is_ensemble: bool,
        area_weights: Optional[Tensor] = None,
        normalization: str = "raw",
        metrics: List[str] = ["rmse", "crps", "rmse_member_avg"],
        temporal_scale: str = "monthly",
        var_names: Optional[List[str]] = None,
        save_data: bool = False,
    ):
        """
        Args:
            metrics: List of metrics to aggregate (e.g. ['rmse', 'mae', 'crps])
            temporal_scale: Temporal scale of the aggregation (e.g. 'monthly', 'yearly')
        """
        self.is_ensemble = is_ensemble
        self.normalization = normalization
        self.temporal_scale = temporal_scale  # 'monthly' or 'yearly'
        self.var_names = var_names
        self.area_weights = None if area_weights is None else area_weights.cpu()
        self.save_data = save_data
        self.metrics = metrics
        

        # Build dictionarys to store the aggregated data (i.e, when record_batch is called we only get a batch of data, so we need to store it)
        self._aggregated_generated_data = {}
        self._aggregated_target_data = {}
        self._aggregated_generated_data_norm = {}
        self._aggregated_target_data_norm = {}
        
        # log parameters
        self._log_dict = {
            "date": {"x_axes": ["date"]},
        }

        # Some plotting parameters
        map_transform = ccrs.PlateCarree()
        self._plot_params = {
            "pr": {"cmap": "BrBG", "add_colorbar": False, "transform": map_transform},
            "tas": {"cmap": "inferno", "add_colorbar": False, "transform": map_transform},
            "crps": {"cmap": "viridis", "transform": map_transform},
            "error": {"cmap": "coolwarm", "add_colorbar": False, "transform": map_transform},
        }

        self.lat = None
        self.lon = None

    @torch.inference_mode()
    def update(
        self,
        target_data: Mapping[str, torch.Tensor],
        gen_data: Mapping[str, torch.Tensor],
        target_data_norm: Mapping[str, torch.Tensor],
        gen_data_norm: Mapping[str, torch.Tensor],
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Record batch of data to be aggregated
        - Data to be generated for all of available years in validation/predict set. (data to be years long)
        """
        # def to_arr(x):
        #     return {k: v.cpu().numpy() for k, v in x.items()} if isinstance(x, dict) else x.cpu().numpy()

        def to_tensor(x):
            return {k: v.cpu() for k, v in x.items()} if isinstance(x, dict) else x.cpu()

        _target_data = to_tensor(target_data)
        _gen_data = to_tensor(gen_data)
        _target_data_norm = to_tensor(target_data_norm)
        _gen_data_norm = to_tensor(gen_data_norm)
        _metadata = metadata if metadata is not None else {}
        batch_size = target_data.batch_size[0]
        self.var_names = list(_target_data.keys()) if self.var_names is None else self.var_names

        # get the size of the data (excluding the batch size)
        target_shape = _target_data[self.var_names[0]].shape[1:]
        gen_shape = _gen_data[self.var_names[0]].shape[0:1] + _gen_data[self.var_names[0]].shape[2:]

        map_datetime_to_temporal_scale = self._get_time_formatting_function()
        for i in range(batch_size):
            # get idx date
            date = map_datetime_to_temporal_scale(_metadata["datetime"][i])

            # handle target data
            target_data_collections = [
                (self._aggregated_target_data, _target_data),
                (self._aggregated_target_data_norm, _target_data_norm),
            ]
            for agg_dict, target in target_data_collections:
                if date not in agg_dict:
                    zero_tensor = torch.zeros(target_shape[0], target_shape[1])
                    agg_dict[date] = {"data": {k: zero_tensor.clone() for k in target.keys()}, "count": 0}
                for k, v in target.items():
                    # loop for each output variable - only relevant when using output variables = ['tas', 'pr']
                    agg_dict[date]["data"][k] += v[i, :, :]
                agg_dict[date]["count"] += 1

            # handle generated data
            gen_data_collections = [
                (self._aggregated_generated_data, _gen_data),
                (self._aggregated_generated_data_norm, _gen_data_norm),
            ]
            for agg_dict, gen in gen_data_collections:
                if date not in agg_dict:
                    zero_tensor = torch.zeros(gen_shape[0], gen_shape[1], gen_shape[2])
                    agg_dict[date] = {"data": {k: zero_tensor.clone() for k in gen.keys()}, "count": 0}
                for k, v in gen.items():
                    # loop for each output variable - only relevant when using output variables = ['tas', 'pr']
                    batch_slice = v[:, i, :, :]
                    agg_dict[date]["data"][k] += batch_slice
                agg_dict[date]["count"] += 1

    @torch.inference_mode()
    def compute(self, label: str = "", epoch: int = None):

        log.info("Getting logs for temporal metrics(extended validation) --- May take a while")
        # asset data to ensure trustworthy evaluation
        self._assert_data()

        # mean the aggregated data
        data_dict = {
            "generated": self._mean_aggregated_data(self._aggregated_generated_data),
            "target": self._mean_aggregated_data(self._aggregated_target_data),
            # "generated_norm": self._mean_aggregated_data(self._aggregated_generated_data_norm),
            # "target_norm": self._mean_aggregated_data(self._aggregated_target_data_norm),
        }
        # calculate metrics for log
        dates = list(data_dict["generated"].keys())
        # infer lat and lon from shape of the data
        shape = data_dict["target"][dates[0]][self.var_names[0]].shape
        self.lat = np.arange(-90, 90, 180 / shape[0])
        self.lon = np.arange(-180, 180, 360 / shape[1])

        logs = self._log_dict.copy()
        image_logs = {}
        ssp = label if label else ""
        label = f"extended_{label}/" if label else ""
        for date in dates:  # loop for each date separating metrics by the date
            # get date from date
            gen = data_dict["generated"][date]
            target = data_dict["target"][date]
            if self.is_ensemble:
                # get ens means
                gen_ens_mean = {k: v.mean(dim=0) for k, v in gen.items()}
            

            ####
            # TODO: add more metrics?
            ####
            logs['date'][date] = self._get_metrics(target, gen, gen_ens_mean, date, self.var_names, self.metrics)
                
            # for var in self.var_names:
            #         # if ens, get ens metrics
            #         logs[f"{label}rmse_member_avg/{date}/{var}"] = np.mean(
            #             [
            #                 metric_functions.root_mean_squared_error(
            #                     predicted=gen[var][i], truth=target[var], weights=self.area_weights
            #                 )
            #                 for i in range(gen[var].shape[0])
            #             ]
            #         )
            #         # logs[f"bias_member_avg/{date}/{var}"] = np.mean(
            #         #     [
            #         #         metrics.time_and_global_mean_bias(predicted=gen[var][i], truth=target[var], weights=self.area_weights)
            #         #         for i in range(gen[var].shape[0])
            #         #     ]
            #         # )
            #     logs[f"{label}rmse/{date}/{var}"] = float(
            #         metric_functions.root_mean_squared_error(
            #             predicted=gen_ens_mean[var], truth=target[var], weights=self.area_weights
            #         )
            #     )

            #     # logs[f"bias/{date}/{var}"] = float(
            #     #     metrics.time_and_global_mean_bias(predicted=gen_ens_mean[var], truth=target[var], weights=self.area_weights)
            #     # )
            #     logs[f"{label}crps/{date}/{var}"] = float(
            #         crps_ensemble(predicted=gen[var], truth=target[var], weights=self.area_weights)
            #     )

            image_logs.update(
                {
                    f"{label}snapshots-{self.temporal_scale}-mean/{key}": val
                    for key, val in self._timeAggSnapshots(
                        target, gen, gen_ens_mean, date, is_ensemble=self.is_ensemble, ssp=ssp
                    ).items()
                }
            )
            if self.is_ensemble:
                image_logs.update(
                    {
                        f"{label}snapshots-crps/{key}": val
                        for key, val in self._crpsSnapshots(target, gen, date, ssp=ssp).items()
                    }
                )
        # {
        #     "date" : {
        #         "x_axes": ["date"],
        #         "2015-01": {
        #             "rmse/tas": 0.1,
        #             "crps/tas": 0.2,
        #             "rmse/pr": 0.1,
        #             "crps/pr": 0.2,
        #             "date": "2015-01"
        #         },
        #         "2015-02": {
        #             "rmse": 0.1,
        #             "crps": 0.2,
        #             "date": "2015-02"
        #         }
        #         # etc...
        #     }
        # }

        return {}, image_logs, logs
    
    def _get_metrics(self, target, pred, pred_ens_mean, date, vars, metrics):
        computed_metrics = {}
        for var in vars:
            for metric in metrics:
                if "member_avg" in metric:
                    label = metric + '/' + var
                    computed_metrics[label] = np.mean(
                            [
                                metric_functions[metric](
                                    predicted=pred[var][i], truth=target[var], weights=self.area_weights
                                )
                                for i in range(pred[var].shape[0])
                            ]
                        )
                elif metric == "crps":
                    computed_metrics[label] = float(
                        metric_functions[metric](predicted=pred[var], truth=target[var], weights=self.area_weights)
                    )
                    continue
                else:
                    computed_metrics[label] = float(
                        metric_functions[metric](
                            predicted=pred_ens_mean[var], truth=target[var], weights=self.area_weights
                        )
                    )
            
        # add date with key 'date' for wandb logging purposes
        computed_metrics['date'] = date
         
        return computed_metrics
            
        

    def _crpsSnapshots(self, target, gen, date, ssp=''):
        """
        For comparing at a Monthly time scale, get a large figure of all 12 months for a given year
        Args:
            data_dict: Dictionary with the aggregated data (e.g. self._aggregated_generated_data)
            year: Year to get the comparison for
        Returns:
            fig: figure with all 12 months for the given year
        """
        snapshots = {}

        def to_xr_dataarray(data):
            return xr.DataArray(data, coords=[("lat", self.lat), ("lon", self.lon)], dims=["lat", "lon"])

        for var in self.var_names:
            # Handle CRPS plot
            fig, axs = plt.subplots(1, 1, figsize=(12, 6), subplot_kw={"projection": ccrs.PlateCarree()})
            fig.suptitle(f"{var} - {date} - CRPS FAIR")

            # Calculate CRPS
            crps = crps_ensemble(predicted=gen[var], truth=target[var], reduction="none")
            # then convert to xr dataarray
            crps = to_xr_dataarray(crps)

            im1 = crps.plot(ax=axs, **self._plot_params["crps"])
            axs.set_title(f"CRPS {var}")

            # Add coastlines
            axs.coastlines()

            snapshots[f"image-crps-fair/{date}/{var}"] = wandb.Image(fig)
        return snapshots

    def _timeAggSnapshots(self, target, gen, gen_ens_mean, date, is_ensemble=True, ssp=''):
        """
        Generate Snapshots of the data aggregated in time
        Args:
            data_dict: Dictionary with the aggregated data (e.g. self._aggregated_generated_data)
        Returns:
            dictionary with the snapshots of the data aggregated in time (keys: Dates, Values: snapshots figs)
        """
        if not is_ensemble:
            raise NotImplementedError("Only implemented for ensemble data")
        
        ssp = ssp + "-" if ssp else ""

        snapshots = {}

        def to_xr_dataarray(data):
            return {k: xr.DataArray(v, coords=[("lat", self.lat), ("lon", self.lon)]) for k, v in data.items()}

        def get_random_ensembles(data):
            # get 2 random ensemble members
            idxs = np.random.choice(data[self.var_names[0]].shape[0], 2, replace=False)
            # get ens member and transform to xr dataarray
            ens_1 = to_xr_dataarray({k: v[idxs[0]] for k, v in data.items()})
            ens_2 = to_xr_dataarray({k: v[idxs[1]] for k, v in data.items()})
            return ens_1, ens_2, idxs

        gen_ens_1, gen_ens_2, ens_idxs = get_random_ensembles(gen)
        target = to_xr_dataarray(target)
        gen = to_xr_dataarray(gen_ens_mean)

        for var in self.var_names:
            # Handle Precipitation unique case
            if var == "pr":
                # get log version to make more visible
                target_log = np.log(target[var] + 1)
                gen_log = np.log(gen[var] + 1)
                gen_ens_1_log = np.log(gen_ens_1[var] + 1)
                gen_ens_2_log = np.log(gen_ens_2[var] + 1)

                fig, axs = plt.subplots(2, 2, figsize=(12, 6), subplot_kw={"projection": ccrs.PlateCarree()})
                fig.suptitle(f"{var}image full field - {var} - log - {date}")

                # First row - target and generated mean
                vmin = min(np.min(target_log), np.min(gen_log), np.min(gen_ens_1_log), np.min(gen_ens_2_log))
                vmax = max(np.max(target_log), np.max(gen_log), np.max(gen_ens_1_log), np.max(gen_ens_2_log))
                im1 = target_log.plot(ax=axs[0, 0], vmin=vmin, vmax=vmax, **self._plot_params[var])
                axs[0, 0].set_title("Target")
                im2 = gen_log.plot(ax=axs[0, 1], vmin=vmin, vmax=vmax, **self._plot_params[var])
                axs[0, 1].set_title("Generated - Mean")

                # Second row - sampled ensembles
                im3 = gen_ens_1_log.plot(ax=axs[1, 0], vmin=vmin, vmax=vmax, **self._plot_params[var])
                axs[1, 0].set_title(f"Genrated - Ensemble {ens_idxs[0]}")
                im4 = gen_ens_2.plot(ax=axs[1, 1], vmin=vmin, vmax=vmax, **self._plot_params[var])
                axs[1, 1].set_title(f"Generated - Ensemble {ens_idxs[1]}")

                # create cbar
                cbar = fig.colorbar(im1, ax=axs.ravel().tolist(), orientation="vertical", pad=0.05)

                # Add coastlines
                for ax in axs:
                    ax.coastlines()

                snapshots[f"image-full-field-log/{date}/{var}"] = wandb.Image(fig)

                # Handle Error plot
                fig, axs = plt.subplots(1, 3, figsize=(12, 6), subplot_kw={"projection": ccrs.PlateCarree()})
                fig.suptitle(f"{var}Error - {var} - log - {date}")

                # Calculate error
                error = gen_log - target_log
                error_gen_ens_1 = gen_ens_1_log - target_log
                error_gen_ens_2 = gen_ens_2_log - target_log
                vmin = min(np.min(error), np.min(error_gen_ens_1), np.min(error_gen_ens_2))
                vmax = max(np.max(error), np.max(error_gen_ens_1), np.max(error_gen_ens_2))
                im1 = error.plot(ax=axs[0], mvin=vmin, vmax=vmax, **self._plot_params["error"])
                axs[0].set_title("Error - Generated Mean")
                im2 = error_gen_ens_1.plot(ax=axs[1], mvin=vmin, vmax=vmax, **self._plot_params["error"])
                axs[1].set_title(f"Error - Generated Ensemble {ens_idxs[0]}")
                im3 = error_gen_ens_2.plot(ax=axs[2], mvin=vmin, vmax=vmax, **self._plot_params["error"])
                axs[2].set_title(f"Error - Generated Ensemble {ens_idxs[1]}")

                # create cbar
                cbar = fig.colorbar(im1, ax=axs.ravel().tolist(), orientation="vertical", pad=0.05)

                # Add coastlines
                for ax in axs:
                    ax.coastlines()

                snapshots[f"image-error-log/{date}/{var}"] = wandb.Image(fig)

            # Handle General plot
            fig, axs = plt.subplots(2, 2, figsize=(12, 6), subplot_kw={"projection": ccrs.PlateCarree()})
            fig.suptitle(f"{var}Full Field {var} - {date}")

            # First row Target and Generated Mean
            vmin = min(np.min(target[var]), np.min(gen[var]), np.min(gen_ens_1[var]), np.min(gen_ens_2[var]))
            vmax = max(np.max(target[var]), np.max(gen[var]), np.max(gen_ens_1[var]), np.max(gen_ens_2[var]))
            im1 = target[var].plot(ax=axs[0, 0], vmin=vmin, vmax=vmax, **self._plot_params[var])
            axs[0, 0].set_title("Target")
            im2 = gen[var].plot(ax=axs[0, 1], vmin=vmin, vmax=vmax, **self._plot_params[var])
            axs[0, 1].set_title("Generated - Mean")

            # Second row - Sampled Ensembles
            im3 = gen_ens_1[var].plot(ax=axs[1, 0], vmin=vmin, vmax=vmax, **self._plot_params[var])
            axs[1, 0].set_title(f"Generated - Ensemble {ens_idxs[0]}")
            im4 = gen_ens_2[var].plot(ax=axs[1, 1], vmin=vmin, vmax=vmax, **self._plot_params[var])
            axs[1, 1].set_title(f"Generated - Ensemble {ens_idxs[1]}")

            # create cbar
            cbar = fig.colorbar(im1, ax=axs.ravel().tolist(), orientation="vertical", pad=0.05)

            # Add coastlines
            for ax in axs.flat:
                ax.coastlines()

            snapshots[f"image-full-field/{date}/{var}"] = wandb.Image(fig)

            # Handle Error plot
            fig, axs = plt.subplots(1, 3, figsize=(16, 4), subplot_kw={"projection": ccrs.PlateCarree()})
            fig.suptitle(f"{var}Error {var} - {date}")

            # Calculate error
            error = gen[var] - target[var]
            error_gen_ens_1 = gen_ens_1[var] - target[var]
            error_gen_ens_2 = gen_ens_2[var] - target[var]
            vmin = min(np.min(error), np.min(error_gen_ens_1), np.min(error_gen_ens_2))
            vmax = max(np.max(error), np.max(error_gen_ens_1), np.max(error_gen_ens_2))
            im1 = error.plot(ax=axs[0], vmin=vmin, vmax=vmax, **self._plot_params["error"])
            axs[0].set_title("Error - Generated Mean")
            im2 = error_gen_ens_1.plot(ax=axs[1], vmin=vmin, vmax=vmax, **self._plot_params["error"])
            axs[1].set_title(f"Error - Generated Ensemble {ens_idxs[0]}")
            im3 = error_gen_ens_2.plot(ax=axs[2], vmin=vmin, vmax=vmax, **self._plot_params["error"])
            axs[2].set_title(f"Error - Generated Ensemble {ens_idxs[1]}")

            # create cbar
            cbar = fig.colorbar(im1, ax=axs.ravel().tolist(), orientation="vertical", pad=0.05)

            # Add coastlines
            for ax in axs.flat:
                ax.coastlines()

            snapshots[f"image-error/{date}/{var}"] = wandb.Image(fig)

        return snapshots

    def _mean_aggregated_data(self, data_dict):
        """
        Mean the aggregated data dictionary

        Args:
            data_dict: Dictionary with the aggregated data (e.g. self._aggregated_generated_data)
        Returns:
            Dictionary with the aggregated data meaned (Keys: Dates, Values: Mean of the data)
        """
        meaned_data = {}
        for date, agg_data in data_dict.items():
            count = agg_data["count"]
            meaned_data[date] = {k: v / count for k, v in agg_data["data"].items()}
        return meaned_data

    def _assert_data(self):
        # assert dates in generated and target data are the same
        assert (
            self._aggregated_generated_data.keys() == self._aggregated_target_data.keys()
        ), "Dates in generated and target data are not the same!"
        # assert that the number of data points for each date corresponds to what we expect (i.e jan - 31 values / feb - 28 values for monthly) // (365 values for yearly)
        month_31 = ["01", "03", "05", "07", "08", "10", "12"]
        month_30 = ["04", "06", "09", "11"]
        month_28 = ["02"]
        if self.temporal_scale == "monthly":
            # check jan and dec have 31 values, where we can infer the year from the keys
            date_keys = list(self._aggregated_generated_data.keys())
            date_counts = [self._aggregated_generated_data[date]["count"] for date in date_keys]
            for i, date in enumerate(date_keys):
                month = date.split("-")[1]
                if month in month_31:
                    assert date_counts[i] == 31, f"Month {month} has {date_counts[i]} values, expected 31"
                elif month in month_30:
                    assert date_counts[i] == 30, f"Month {month} has {date_counts[i]} values, expected 30"
                elif month in month_28:
                    assert date_counts[i] == 28, f"Month {month} has {date_counts[i]} values, expected 28"
                else:
                    raise ValueError(f"Month {month} is not valid!")
        elif self.temporal_scale == "yearly":
            # check that each year has 365 values
            date_keys = list(self._aggregated_generated_data.keys())
            date_counts = [self._aggregated_generated_data[date]["count"] for date in date_keys]
            for i, date in enumerate(date_keys):
                assert date_counts[i] == 365, f"Year {date} has {date_counts[i]} values, expected 365"
        else:
            raise ValueError(f"Temporal scale {self.temporal_scale} is not supported!")

    def _get_time_formatting_function(self):
        """
        Helper function to get the function to format datetime object to the desired temporal scale
            returns - Function
        """

        def getMonthly(datetime_obj):
            "Get YYYY-MM from datetime object"
            # convert string to cftime object
            datetime_obj = cftime.DatetimeNoLeap(*[int(i) for i in datetime_obj.split("-")])
            return datetime_obj.strftime("%Y-%m")

        def getYearly(datetime_obj):
            "Get YYYY from datetime object"
            datetime_obj = cftime.DatetimeNoLeap(*[int(i) for i in datetime_obj.split("-")])
            return datetime_obj.strftime("%Y")

        if self.temporal_scale == "monthly":
            return getMonthly
        elif self.temporal_scale == "yearly":
            return getYearly
        else:
            raise ValueError(f"Temporal scale {self.temporal_scale} is not supported!")
