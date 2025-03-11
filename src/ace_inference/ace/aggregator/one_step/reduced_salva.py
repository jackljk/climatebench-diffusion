from collections import defaultdict
from typing import Dict, Mapping, Optional

import torch
import xarray as xr
import xskillscore as xs
from torch import nn

from src.ace_inference.core import metrics
from src.losses import losses

from ..reduced_metrics import AreaWeightedReducedMetric, ReducedMetric


class AbstractMeanMetric:
    def __init__(self, device: torch.device):
        self._total = torch.tensor(0.0, device=device)

    def get(self) -> torch.Tensor:
        return self._total


class L1Loss(AbstractMeanMetric):
    def record(self, targets: torch.Tensor, preds: torch.Tensor):
        self._total += nn.functional.l1_loss(preds, targets)


class SpreadSkillRatio(AbstractMeanMetric):
    def record(self, targets: torch.Tensor, preds: torch.Tensor):
        rmse = nn.functional.mse_loss(preds.mean(dim=0), targets, reduction="mean").sqrt()
        # calculate spread over ensemble dim
        spread = preds.var(dim=0).mean().sqrt()
        self._total += spread / rmse


class CRPS(AbstractMeanMetric):
    def record(self, targets: torch.Tensor, preds: torch.Tensor):
        preds_da = xr.DataArray(preds.cpu(), dims=["member", "sample", "lat", "lon"])
        targets_da = xr.DataArray(targets.cpu(), dims=["sample", "lat", "lon"])
        crps = xs.crps_ensemble(observations=targets_da, forecasts=preds_da, member_dim="member")
        self._total += torch.from_numpy(crps.values).float()


class MeanAggregator:
    """
    Aggregator for mean-reduced metrics.

    These are metrics such as means which reduce to a single float for each batch,
    and then can be averaged across batches to get a single float for the
    entire dataset. This is important because the aggregator uses the mean to combine
    metrics across batches and processors.
    """

    def __init__(self, area_weights: torch.Tensor, is_ensemble: bool):
        self._area_weights = area_weights
        self._n_batches = 0
        self._variable_metrics: Optional[Dict[str, Dict[str, ReducedMetric]]] = None
        self.is_ensemble = is_ensemble

    def _get_variable_metrics(self, gen_data: Mapping[str, torch.Tensor]):
        if self._variable_metrics is None:
            self._variable_metrics = defaultdict(dict)
            any_gen_data = gen_data[list(gen_data.keys())[0]]
            self.device = any_gen_data.device
            area_weights = self._area_weights.to(self.device)

            for var_name in gen_data.keys():
                self._variable_metrics["l1"][var_name] = L1Loss(device=self.device)
                metrics_zipped = [
                    ("weighted_rmse", metrics.root_mean_squared_error),
                    ("weighted_bias", metrics.weighted_mean_bias),
                    ("weighted_grad_mag_percent_diff", metrics.gradient_magnitude_percent_diff),
                ]
                if self.is_ensemble:
                    self._variable_metrics["ssr"][var_name] = SpreadSkillRatio(device=self.device)
                    metrics_zipped += [("weighted_crps", losses.crps_ensemble)]

                for i, (metric_name, metric) in enumerate(metrics_zipped):
                    self._variable_metrics[metric_name][var_name] = AreaWeightedReducedMetric(
                        area_weights=area_weights, device=self.device, compute_metric=metric
                    )

        return self._variable_metrics

    @torch.no_grad()
    def record_batch(
        self,
        target_data: Mapping[str, torch.Tensor],
        gen_data: Mapping[str, torch.Tensor],
        target_data_norm: Mapping[str, torch.Tensor] = None,
        gen_data_norm: Mapping[str, torch.Tensor] = None,
        inputs_norm: Mapping[str, torch.Tensor] = None,
    ):
        variable_metrics = self._get_variable_metrics(gen_data)
        if self.is_ensemble:
            ensemble_mean = {name: member_preds.mean(dim=0) for name, member_preds in gen_data.items()}
        else:
            ensemble_mean = gen_data

        for name in gen_data.keys():  # e.g. temperature, precipitation, etc
            for metric in variable_metrics:  # e.g. l1, weighted_rmse, etc
                kwargs = {}
                # compute gradf mag differently, and potentially rmse
                if "ssr" in metric or "crps" in metric:
                    pred = gen_data[name]
                elif "grad_mag" in metric:
                    pred = gen_data[name]
                    kwargs["is_ensemble_prediction"] = self.is_ensemble
                else:
                    pred = ensemble_mean[name]

                # time_s = time.time()
                variable_metrics[metric][name].record(targets=target_data[name], preds=pred, **kwargs)
                # time_taken = time.time() - time_s
                # print(f"Time taken for {metric} {name} in s: {time_taken:.5f}")
        self._n_batches += 1

    @torch.no_grad()
    def get_logs(self, label: str = ""):
        """
        Returns logs as can be reported to WandB.

        Args:
            label: Label to prepend to all log keys.
        """
        if self._variable_metrics is None or self._n_batches == 0:
            raise ValueError("No batches have been recorded.")
        logs = {}
        label = label + "/" if label else ""
        for metric in self._variable_metrics:
            for key in self._variable_metrics[metric]:
                logs[f"{label}{metric}/{key}"] = (self._variable_metrics[metric][key].get() / self._n_batches).detach()
        # dist = Distributed.get_instance()
        for key in sorted(logs.keys()):
            logs[key] = float(logs[key].cpu())  # .numpy()
        #     logs[key] = float(dist.reduce_mean(logs[key]).cpu().numpy())
        return logs

    @torch.no_grad()
    def get_dataset(self, label: str) -> xr.Dataset:
        logs = self.get_logs(label=label)
        logs = {key.replace("/", "-"): logs[key] for key in logs}
        data_vars = {}
        for key, value in logs.items():
            data_vars[key] = xr.DataArray(value)
        return xr.Dataset(data_vars=data_vars)
