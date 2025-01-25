from typing import Dict, Mapping, Optional, Tuple

import numpy as np
import torch
import xarray as xr

from src.evaluation import metrics
from src.evaluation.aggregators._abstract_aggregator import AbstractAggregator
from src.losses.losses import crps_ensemble
from src.utilities.utils import add


def get_gen_shape(gen_data: Mapping[str, torch.Tensor]):
    for name in gen_data:
        return gen_data[name].shape


class TimeMeanAggregator(AbstractAggregator):
    """Statistics on the time-mean state.

    This aggregator keeps track of the time-mean state, then computes
    statistics on that time-mean state when logs are retrieved.
    """

    _image_captions = {
        "bias_map": "{name} time-mean bias (generated - target)",
        "gen_map": "{name} time-mean generated",
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._target_data: Optional[Dict[str, torch.Tensor]] = None
        self._gen_data: Optional[Dict[str, torch.Tensor]] = None
        self._target_data_norm = None
        self._gen_data_norm = None
        self._n_batches = 0

    @torch.inference_mode()
    def _record_batch(
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
                d = add(maybe_dict, new_data)
            return d

        self._target_data = add_or_initialize_time_mean(self._target_data, target_data)
        self._gen_data = add_or_initialize_time_mean(self._gen_data, gen_data)
        self._n_batches += 1

    @torch.inference_mode()
    def _get_logs(self, **kwargs) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Returns logs as can be reported to WandB.
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
                crps_ensemble(predicted=gen, truth=target, weights=area_weights).cpu().numpy()
            )
        return logs, {}

    @torch.inference_mode()
    def get_dataset(self, **kwargs) -> xr.Dataset:
        logs = self.get_logs(**kwargs)
        logs = {key.replace("/", "-"): logs[key] for key in logs}
        data_vars = {}
        for key, value in logs.items():
            data_vars[key] = xr.DataArray(value)
        return xr.Dataset(data_vars=data_vars)
