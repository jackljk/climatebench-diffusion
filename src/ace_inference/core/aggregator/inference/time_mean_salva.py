from typing import Dict, Mapping, Optional

import numpy as np
import torch
import xarray as xr

import wandb
from src.ace_inference.core import metrics
from src.ace_inference.core.data_loading.data_typing import VariableMetadata
from src.losses.losses import crps_ensemble


def get_gen_shape(gen_data: Mapping[str, torch.Tensor]):
    for name in gen_data:
        return gen_data[name].shape


class TimeMeanAggregator:
    """Statistics on the time-mean state.

    This aggregator keeps track of the time-mean state, then computes
    statistics on that time-mean state when logs are retrieved.
    """

    _image_captions = {
        "bias_map": "{name} time-mean bias (generated - target) [{units}]",
        "gen_map": "{name} time-mean generated [{units}]",
    }

    def __init__(
        self,
        area_weights: torch.Tensor,
        is_ensemble: bool = False,
        sigma_coordinates=None,
        metadata: Optional[Mapping[str, VariableMetadata]] = None,
    ):
        self._area_weights = area_weights
        self._target_data: Optional[Dict[str, torch.Tensor]] = None
        self._gen_data: Optional[Dict[str, torch.Tensor]] = None
        self._target_data_norm = None
        self._gen_data_norm = None
        self._n_batches = 0
        self._is_ensemble = is_ensemble

        if metadata is None:
            self._metadata: Mapping[str, VariableMetadata] = {}
        else:
            self._metadata = metadata

    @torch.no_grad()
    def record_batch(
        self,
        target_data: Mapping[str, torch.Tensor],
        gen_data: Mapping[str, torch.Tensor],
    ):
        def add_or_initialize_time_mean(
            maybe_dict: Optional[Dict[str, torch.Tensor]],
            new_data: Mapping[str, torch.Tensor],
        ) -> Mapping[str, torch.Tensor]:
            if maybe_dict is None:
                d: Dict[str, torch.Tensor] = {name: tensor for name, tensor in new_data.items()}
            else:
                d = maybe_dict
                for name, tensor in new_data.items():
                    d[name] += tensor
            return d

        self._target_data = add_or_initialize_time_mean(self._target_data, target_data)
        self._gen_data = add_or_initialize_time_mean(self._gen_data, gen_data)
        self._n_batches += 1

    @torch.no_grad()
    def get_logs(self, label: str):
        """
        Returns logs as can be reported to WandB.

        Args:
            label: Label to prepend to all log keys.
        """
        if self._n_batches == 0:
            raise ValueError("No data recorded.")
        area_weights = self._area_weights
        logs = {}
        # dist = Distributed.get_instance()
        for name in self._gen_data.keys():
            gen = self._gen_data[name] / self._n_batches
            target = self._target_data[name] / self._n_batches
            # gen = dist.reduce_mean(self._gen_data[name] / self._n_batches)
            # target = dist.reduce_mean(self._target_data[name] / self._n_batches)
            if self._is_ensemble:
                gen_ens_mean = gen.mean(dim=0)
                logs[f"rmse_member_avg/{name}"] = np.mean(
                    [
                        metrics.root_mean_squared_error(predicted=gen[i], truth=target, weights=area_weights)
                        .cpu()
                        .numpy()
                        for i in range(gen.shape[0])
                    ]
                )
                logs[f"bias_member_avg/{name}"] = np.mean(
                    [
                        metrics.time_and_global_mean_bias(predicted=gen[i], truth=target, weights=area_weights)
                        .cpu()
                        .numpy()
                        for i in range(gen.shape[0])
                    ]
                )
            else:
                gen_ens_mean = gen

            logs[f"rmse/{name}"] = float(
                metrics.root_mean_squared_error(predicted=gen_ens_mean, truth=target, weights=area_weights)
                .cpu()
                .numpy()
            )

            logs[f"bias/{name}"] = float(
                metrics.time_and_global_mean_bias(predicted=gen_ens_mean, truth=target, weights=area_weights)
                .cpu()
                .numpy()
            )
            logs[f"crps/{name}"] = float(
                crps_ensemble(predictions=gen, observations=target, weights=area_weights).cpu().numpy()
            )
        return {f"{label}/{key}": logs[key] for key in logs}, {}

    def _get_image(self, key: str, name: str, data: torch.Tensor):
        sample_dim = 0
        lat_dim = -2
        data = data.mean(dim=sample_dim).flip(dims=[lat_dim]).cpu()
        caption = self._get_caption(key, name, data)
        return wandb.Image(data, caption=caption)

    def _get_caption(self, key: str, name: str, data: torch.Tensor) -> str:
        if name in self._metadata:
            caption_name = self._metadata[name].long_name
            units = self._metadata[name].units
        else:
            caption_name, units = name, "unknown_units"
        caption = self._image_captions[key].format(name=caption_name, units=units)
        caption += f" vmin={data.min():.4g}, vmax={data.max():.4g}."
        return caption

    @torch.no_grad()
    def get_dataset(self, label: str) -> xr.Dataset:
        logs = self.get_logs(label=label)
        logs = {key.replace("/", "-"): logs[key] for key in logs}
        data_vars = {}
        for key, value in logs.items():
            data_vars[key] = xr.DataArray(value)
        return xr.Dataset(data_vars=data_vars)
