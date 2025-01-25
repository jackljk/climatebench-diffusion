from collections import defaultdict
from typing import Dict, Mapping, Optional

import torch
import xarray as xr
from torch import nn

from src.ace_inference.core import metrics
from src.ace_inference.core.distributed import Distributed
from src.losses import losses

from ..inference.reduced import compute_metric_on
from .reduced_metrics import AreaWeightedReducedMetric, ReducedMetric


def get_gen_shape(gen_data: Mapping[str, torch.Tensor]):
    for name in gen_data:
        return gen_data[name].shape


class L1Loss:
    def __init__(self, device: torch.device):
        self._total = torch.tensor(0.0, device=device)

    def record(self, target: torch.Tensor, gen: torch.Tensor):
        self._total += nn.functional.l1_loss(
            gen,
            target,
        )

    def get(self) -> torch.Tensor:
        return self._total


class MeanAggregator:
    """
    Aggregator for mean-reduced metrics.

    These are metrics such as means which reduce to a single float for each batch,
    and then can be averaged across batches to get a single float for the
    entire dataset. This is important because the aggregator uses the mean to combine
    metrics across batches and processors.
    """

    def __init__(
        self,
        area_weights: torch.Tensor,
        target_time: int = 1,
        is_ensemble: bool = False,
        dist: Optional[Distributed] = None,
        device: torch.device = torch.device("cpu"),
    ):
        self._area_weights = area_weights
        self._shape_x = None
        self._shape_y = None
        self._n_batches = 0
        self.device = device
        self._loss = torch.tensor(0.0, device=self.device)
        self._variable_metrics: Optional[Dict[str, Dict[str, ReducedMetric]]] = None
        self._target_time = target_time
        self.is_ensemble = is_ensemble
        if dist is None:
            self._dist = Distributed.get_instance()
        else:
            self._dist = dist

    def _get_variable_metrics(self, gen_data: Mapping[str, torch.Tensor]):
        if self._variable_metrics is None:
            self._variable_metrics = defaultdict(dict)

            area_weights = self._area_weights
            for key in gen_data.keys():
                metrics_zipped = [
                    ("weighted_rmse", metrics.root_mean_squared_error),
                    ("weighted_bias", metrics.weighted_mean_bias),
                    ("weighted_grad_mag_percent_diff", metrics.gradient_magnitude_percent_diff),
                    ("weighted_mean_gen", compute_metric_on(source="gen", metric=metrics.weighted_mean)),
                ]
                if self.is_ensemble:
                    metrics_zipped += [
                        ("weighted_crps", losses.crps_ensemble),
                        ("weighted_ssr", metrics.spread_skill_ratio),
                    ]

                for i, (metric_name, metric) in enumerate(metrics_zipped):
                    self._variable_metrics[metric_name][key] = AreaWeightedReducedMetric(
                        area_weights=area_weights,
                        device=self.device,
                        compute_metric=metric,
                    )

        return self._variable_metrics

    @torch.no_grad()
    def record_batch(
        self,
        loss: float,
        target_data: Mapping[str, torch.Tensor],
        gen_data: Mapping[str, torch.Tensor],
        target_data_norm: Mapping[str, torch.Tensor],
        gen_data_norm: Mapping[str, torch.Tensor],
        i_time_start: int = 0,
    ):
        self._loss += loss
        variable_metrics = self._get_variable_metrics(gen_data)
        time_dim = 1
        time_dim_gen = 2 if self.is_ensemble else time_dim
        time_len = gen_data[list(gen_data.keys())[0]].shape[time_dim_gen]
        target_time = self._target_time - i_time_start
        if target_time >= 0 and time_len > target_time:
            for name in gen_data.keys():
                target = target_data[name].select(dim=time_dim, index=target_time)
                gen_full = gen_data[name].select(dim=time_dim_gen, index=target_time)
                if self.is_ensemble:
                    ensemble_mean = gen_full.mean(dim=0)
                else:
                    ensemble_mean = gen_full
                for metric in variable_metrics:
                    kwargs = {}
                    if "ssr" in metric or "crps" in metric:
                        gen = gen_full
                    elif "grad_mag" in metric:
                        gen = gen_full
                        kwargs["is_ensemble_prediction"] = self.is_ensemble
                    else:
                        gen = ensemble_mean

                    variable_metrics[metric][name].record(target=target, gen=gen, **kwargs)
            # only increment n_batches if we actually recorded a batch
            self._n_batches += 1

    @torch.no_grad()
    def get_logs(self, label: str):
        """
        Returns logs as can be reported to WandB.

        Args:
            label: Label to prepend to all log keys.
        """
        if self._variable_metrics is None or self._n_batches == 0:
            raise ValueError("No batches have been recorded.")
        logs = {f"{label}/loss": self._loss / self._n_batches}
        for metric in self._variable_metrics:
            for key in self._variable_metrics[metric]:
                logs[f"{label}/{metric}/{key}"] = self._variable_metrics[metric][key].get() / self._n_batches
        for key in sorted(logs.keys()):
            logs[key] = float(self._dist.reduce_mean(logs[key].detach()).cpu().numpy())
        return logs

    @torch.no_grad()
    def get_dataset(self, label: str) -> xr.Dataset:
        logs = self.get_logs(label=label)
        logs = {key.replace("/", "-"): logs[key] for key in logs}
        data_vars = {}
        for key, value in logs.items():
            data_vars[key] = xr.DataArray(value)
        return xr.Dataset(data_vars=data_vars)
