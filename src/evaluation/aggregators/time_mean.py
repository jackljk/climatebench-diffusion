from typing import Dict, Mapping, Optional, Tuple

import numpy as np
import torch
import xarray as xr
from src.evaluation.torchmetrics import (
    MeanSquaredError,
    MeanError,
    ContinuousRankedProbabilityScore,
    MeanAbsoluteError,
    SpreadSkillRatio,
)

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
    def _get_logs(self, **kwargs):
        """
        Returns logs as can be reported to WandB.
        """
        if self._n_batches == 0:
            raise ValueError(
                "No data recorded. This aggregator is only called for forecasting tasks. "
                "Did you mistakenly try to use it for a different task?"
            )
        area_weights = self._area_weights
        logs = {}
        # dist = Distributed.get_instance()
        for name in self._gen_data.keys():
            gen = self._gen_data[name] / self._n_batches
            target = self._target_data[name] / self._n_batches
            print(f"{gen.shape=}, {target.shape=}, {self._n_batches=}")
            metric_aggs = {
                f"rmse/{name}": MeanSquaredError(weights=area_weights, squared=False),
                f"bias/{name}": MeanError(weights=area_weights),
            }
            if self._is_ensemble:
                gen_ens_mean = gen.mean(dim=0)
                metric_aggs_ens_only = {
                    f"rmse_member_avg/{name}": MeanSquaredError(weights=area_weights, squared=False),
                    f"bias_member_avg/{name}": MeanError(weights=area_weights),
                }
                for ens_i, ens_mem in enumerate(gen):
                    metric_aggs_ens_only[f"rmse_member_avg/{name}"].update(ens_mem, target)
                for key, metric in metric_aggs_ens_only.items():
                    logs[key] = to_float(metric.compute())

                # Ensemble logs:
                ssr_agg = SpreadSkillRatio(weights=area_weights)
                ssr_agg.update(gen, target)
                logs[f"ssr/{name}"] = to_float(ssr_agg.compute())
                metric_aggs[f"crps/{name}"] = ContinuousRankedProbabilityScore(weights=area_weights)
            else:
                gen_ens_mean = gen
                # Without ensemble, CRPS becomes a mean absolute error
                metric_aggs[f"crps/{name}"] = MeanAbsoluteError(weights=area_weights)

            # Compute metrics
            for key, metric in metric_aggs.items():
                gen_here = gen if metric == "crps" else gen_ens_mean
                metric.update(gen_here, target)
                logs[key] = to_float(metric.compute())  # Should be corrctly synced across all processes

        return logs, {}, {}

    @torch.inference_mode()
    def get_dataset(self, **kwargs) -> xr.Dataset:
        logs = self.compute(**kwargs)
        logs = {key.replace("/", "-"): logs[key] for key in logs}
        data_vars = {}
        for key, value in logs.items():
            data_vars[key] = xr.DataArray(value)
        return xr.Dataset(data_vars=data_vars)


def to_float(tensor: torch.Tensor) -> float:
    return tensor.cpu().numpy().item()
