from collections import defaultdict
from typing import Dict, Mapping, Optional

import torch
import xarray as xr
from torch import nn

from src.evaluation import metrics
from src.evaluation.reduced_metrics import AreaWeightedReducedMetric, ReducedMetric
from src.losses import losses


class AbstractMeanMetric:
    def __init__(self, device: torch.device):
        self._total = torch.tensor(0.0, device=device)

    def get(self) -> torch.Tensor:
        return self._total


class L1Loss(AbstractMeanMetric):
    # Note: NOT area weighted
    def record(self, targets: torch.Tensor, preds: torch.Tensor):
        self._total += nn.functional.l1_loss(preds, targets)


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
        is_ensemble: bool,
        record_normed: bool = False,
        record_rmse: bool = True,
        record_abs_values: bool = False,
    ):
        self._area_weights = area_weights
        self._n_batches = 0
        self._variable_metrics: Optional[Dict[str, Dict[str, ReducedMetric]]] = None
        self.is_ensemble = is_ensemble
        self.record_normed = record_normed
        self.record_rmse = record_rmse
        self.record_abs_values = record_abs_values
        if area_weights is None:
            self._area_weights_dims = (-3, -2, -1)  # None  # ( -3, -2, -1))
        elif len(area_weights.shape) == 2:
            self._area_weights_dims = (-2, -1)
        elif len(area_weights.shape) == 1:
            self._area_weights_dims = (-1,)
        else:
            raise ValueError(f"Area weights must be 1D or 2D tensor, got {area_weights.shape}")

    def _get_variable_metrics(self, gen_data: Mapping[str, torch.Tensor]):
        if self._variable_metrics is None:
            self._variable_metrics = defaultdict(dict)
            if torch.is_tensor(gen_data):
                self.device = gen_data.device
                gen_data_keys = [""]
            else:
                self.device = gen_data[list(gen_data.keys())[0]].device  # any key will do
                gen_data_keys = list(gen_data.keys())
            if self._area_weights is not None:
                area_weights = self._area_weights.to(self.device)
            else:
                area_weights = None

            metric_names = ["l1", "rmse", "bias", "grad_mag_percent_diff"]
            if self.is_ensemble:
                metric_names += ["ssr", "crps"]
            if self.record_normed:
                metric_names += [f"{metric}_normed" for metric in metric_names if metric != "l1"]
            for i, var_name in enumerate(gen_data_keys):
                try:
                    self._variable_metrics["l1"][var_name] = L1Loss(device=self.device)
                except KeyError as e:
                    if i > 0:
                        raise e
                    self._variable_metrics = dict()
                    for metric in metric_names:
                        self._variable_metrics[metric] = dict()
                    self._variable_metrics["l1"][var_name] = L1Loss(device=self.device)

                if self.record_rmse:
                    mse_metric = ("rmse", metrics.root_mean_squared_error)
                else:
                    mse_metric = ("mse", metrics.mean_squared_error)
                metrics_zipped = [
                    mse_metric,
                    ("bias", metrics.weighted_mean_bias),
                    ("grad_mag_percent_diff", metrics.gradient_magnitude_percent_diff),
                ]
                if self.record_abs_values:
                    metrics_zipped += [
                        ("mean_gen", metrics.compute_metric_on(source="gen", metric=metrics.weighted_mean)),
                        ("mean_target", metrics.compute_metric_on(source="target", metric=metrics.weighted_mean)),
                        ("std_gen", metrics.compute_metric_on(source="gen", metric=metrics.weighted_std)),
                        ("std_target", metrics.compute_metric_on(source="target", metric=metrics.weighted_std)),
                    ]
                if self.is_ensemble:
                    metrics_zipped += [("crps", losses.crps_ensemble)]
                    metrics_zipped += [("ssr", metrics.spread_skill_ratio)]

                for i, (metric_name, metric) in enumerate(metrics_zipped):
                    self._variable_metrics[metric_name][var_name] = AreaWeightedReducedMetric(
                        area_weights=area_weights,
                        device=self.device,
                        compute_metric=metric,
                        dim=self._area_weights_dims,
                    )

            if self.record_normed:
                for var_name in gen_data_keys:
                    for i, (metric_name, metric) in enumerate(metrics_zipped):
                        self._variable_metrics[f"{metric_name}_normed"][var_name] = AreaWeightedReducedMetric(
                            area_weights=area_weights,
                            device=self.device,
                            compute_metric=metric,
                            dim=self._area_weights_dims,
                        )

        return self._variable_metrics

    @torch.inference_mode()
    def record_batch(
        self,
        target_data: Mapping[str, torch.Tensor],
        gen_data: Mapping[str, torch.Tensor],
        target_data_norm: Mapping[str, torch.Tensor] = None,
        gen_data_norm: Mapping[str, torch.Tensor] = None,
    ):
        variable_metrics = self._get_variable_metrics(gen_data)
        is_tensor = torch.is_tensor(gen_data)
        if is_tensor:  # add dummy key
            gen_data = {"": gen_data}
            target_data = {"": target_data}
            gen_data_norm = {"": gen_data_norm}
            target_data_norm = {"": target_data_norm}

        record_normed_list = [True, False] if self.record_normed else [False]
        for is_normed in record_normed_list:
            if is_normed:
                preds_data = gen_data_norm
                truth_data = target_data_norm
                var_metrics_here = {metric: v for metric, v in variable_metrics.items() if "normed" in metric}
            else:
                preds_data = gen_data
                truth_data = target_data
                var_metrics_here = {metric: v for metric, v in variable_metrics.items() if "normed" not in metric}

            for metric in var_metrics_here.keys():  # e.g. l1, weighted_rmse, etc
                if "grad_mag" in metric:
                    kwargs = {"is_ensemble_prediction": self.is_ensemble}
                else:
                    kwargs = {}

                for var_name, var_preds in preds_data.items():  # e.g. temperature, precipitation, etc
                    if "ssr" in metric or "crps" in metric or "grad_mag" in metric:
                        preds = var_preds
                    else:
                        preds = var_preds.mean(dim=0) if self.is_ensemble else var_preds

                    # time_s = time.time()
                    try:
                        variable_metrics[metric][var_name].record(targets=truth_data[var_name], preds=preds, **kwargs)
                    except AssertionError as e:
                        raise AssertionError(f"Error with {metric=}. {var_name=}, {self.is_ensemble=}") from e
                    # time.time() - time_s
                    # print(f"Time taken for {metric} {name} in s: {time_taken:.5f}")

        self._n_batches += 1

    @torch.inference_mode()
    def get_logs(self, label: str = "", epoch: Optional[int] = None) -> Dict[str, float]:
        """
        Returns logs as can be reported to WandB.

        Args:
            label: Label to prepend to all log keys.
            epoch: Current epoch number.
        """
        if self._variable_metrics is None or self._n_batches == 0:
            raise ValueError(f"No batches have been recorded. n_batches={self._n_batches}")
        logs = {}
        label = label + "/" if label else ""
        for i, metric in enumerate(self._variable_metrics):
            for variable, metric_value in self._variable_metrics[metric].items():
                metric_value = metric_value.get()
                if metric_value is None:
                    raise ValueError(
                        f"{metric=} hasn't been computed for {variable=}. ({label=}, {self._n_batches=}, {i=})"
                    )
                log_key = f"{label}{metric}/{variable}".rstrip("/")
                logs[log_key] = float((metric_value / self._n_batches).detach().item())

        # for key in sorted(logs.keys()):
        # logs[key] = float(logs[key].cpu())  # .numpy()

        return logs

    @torch.inference_mode()
    def get_dataset(self, label: str) -> xr.Dataset:
        logs = self.get_logs(label=label)
        logs = {key.replace("/", "-"): logs[key] for key in logs}
        data_vars = {}
        for key, value in logs.items():
            data_vars[key] = xr.DataArray(value)
        return xr.Dataset(data_vars=data_vars)
