from typing import Dict, Mapping, Optional

import torch
import xarray as xr

from src.evaluation.aggregators._abstract_aggregator import AbstractAggregator
from src.evaluation.torchmetrics import (
    ContinuousRankedProbabilityScore,
    MeanAbsoluteError,
    MeanError,
    MeanSquaredError,
    SpreadSkillRatio,
)
from src.utilities.utils import add
from src.utilities.plotting import create_wandb_figures


def get_gen_shape(gen_data: Mapping[str, torch.Tensor]):
    for name in gen_data:
        return gen_data[name].shape


class TimeMeanAggregator(AbstractAggregator):
    """Statistics on the time-mean state.

    This aggregator keeps track of the time-mean state, then computes
    statistics on that time-mean state when logs are retrieved.
    """

    def __init__(self, log_images: bool = False, mean_over_batch_dim: bool = False, **kwargs):
        super().__init__(**kwargs)
        self._target_data: Optional[Dict[str, torch.Tensor]] = None
        self._gen_data: Optional[Dict[str, torch.Tensor]] = None
        self._target_data_norm = None
        self._gen_data_norm = None
        self._log_images = log_images
        self.total = 0
        self.mean_over_batch_dim = mean_over_batch_dim

    @torch.inference_mode()
    def _record_batch(
        self,
        target_data: Mapping[str, torch.Tensor],
        gen_data: Mapping[str, torch.Tensor],
        target_data_norm: Mapping[str, torch.Tensor] = None,
        gen_data_norm: Mapping[str, torch.Tensor] = None,
        metadata=None
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
        if self.mean_over_batch_dim:
            self.total += target_data[list(target_data.keys())[0]].shape[0]
            b_dim_gen = 1 if self._is_ensemble else 0
            target_data = {name: tensor.sum(dim=0, keepdim=True) for name, tensor in target_data.items()}
            gen_data = {name: tensor.sum(dim=b_dim_gen, keepdim=True) for name, tensor in gen_data.items()}
            # target: torch.Size([1, 192, 288]) torch.Size([1, 192, 288])
            # gen: torch.Size([5, 1, 192, 288]) torch.Size([5, 1, 192, 288])
        else:
            self.total += 1
        self._target_data = add_or_initialize_time_mean(self._target_data, target_data)
        self._gen_data = add_or_initialize_time_mean(self._gen_data, gen_data)

    @torch.inference_mode()
    def _get_logs(self, label: str = "", **kwargs):
        """
        Returns logs as can be reported to WandB.
        """
        if self.total == 0:
            raise ValueError(
                "No data recorded. This aggregator is only called for forecasting tasks. "
                "Did you mistakenly try to use it for a different task?"
            )
        area_weights = self._area_weights
        logs, log_snapshots = {}, {}
        for name in self._gen_data.keys():
            gen = self._gen_data[name] / self.total
            target = self._target_data[name] / self.total
            device = gen.device
            metric_aggs = {
                f"rmse/{name}": MeanSquaredError(weights=area_weights, squared=False).to(device),
                f"bias/{name}": MeanError(weights=area_weights).to(device),
            }
            if self._is_ensemble:
                gen_ens_mean = gen.mean(dim=0)
                metric_aggs[f"ssr/{name}"] = SpreadSkillRatio(weights=area_weights).to(device)
                metric_aggs[f"crps/{name}"] = ContinuousRankedProbabilityScore(weights=area_weights).to(device)
                #  Log  member-wise metrics
                metric_aggs_ens_only = {
                    f"rmse_member_avg/{name}": MeanSquaredError(weights=area_weights, squared=False),
                    f"bias_member_avg/{name}": MeanError(weights=area_weights),
                }
                for ens_i, ens_mem in enumerate(gen):
                    for key, metric in metric_aggs_ens_only.items():
                        metric.update(ens_mem, target)

                for key, metric in metric_aggs_ens_only.items():
                    logs[key] = to_float(metric.compute())
            else:
                gen_ens_mean = gen
                # Without ensemble, CRPS becomes a mean absolute error
                metric_aggs[f"crps/{name}"] = MeanAbsoluteError(weights=area_weights).to(device)

            # Compute metrics
            for key, metric in metric_aggs.items():
                gen_here = gen if "ssr" in key or "crps" in key else gen_ens_mean
                metric.update(gen_here, target)
                logs[key] = to_float(metric.compute())  # Should be correctly synced across all processes

            # remove datetime from self.coords dict
            if 'datetime' in self.coords:
                self.coords.pop('datetime')
            # fig_shared_label="" since we'll add the time_mean prefix further down (so that it's in the front)
            snapshots_var = create_wandb_figures(target, gen, name, fig_shared_label="", coords=self.coords)
            log_snapshots.update(snapshots_var)

        label = label + "/" if label else ""
        logs = {f"{label}{key}": logs[key] for key in logs}
        log_snapshots = {f"{label}{key}": log_snapshots[key] for key in log_snapshots}
        return logs, log_snapshots, {}

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
