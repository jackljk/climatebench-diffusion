from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Optional, Protocol, Union

import torch
import xarray as xr

from src.ace_inference.core.aggregator.inference.reduced import MeanAggregator
from src.ace_inference.core.aggregator.inference.time_mean import TimeMeanAggregator
from src.ace_inference.core.aggregator.inference.video import VideoAggregator
from src.ace_inference.core.aggregator.inference.zonal_mean import ZonalMeanAggregator
from src.ace_inference.core.aggregator.one_step.reduced import MeanAggregator as OneStepMeanAggregator
from src.ace_inference.core.data_loading.data_typing import SigmaCoordinates, VariableMetadata
from src.ace_inference.core.device import get_device
from src.ace_inference.core.distributed import Distributed
from src.ace_inference.core.wandb import WandB
from src.evaluation.aggregators.snapshot import SnapshotAggregator
from wandb import Table


wandb = WandB.get_instance()


class _Aggregator(Protocol):
    @torch.no_grad()
    def record_batch(
        self,
        loss: float,
        target_data: Mapping[str, torch.Tensor],
        gen_data: Mapping[str, torch.Tensor],
        target_data_norm: Mapping[str, torch.Tensor],
        gen_data_norm: Mapping[str, torch.Tensor],
    ): ...

    @torch.no_grad()
    def get_logs(self, label: str): ...

    @torch.no_grad()
    def get_dataset(self) -> xr.Dataset: ...


class InferenceAggregator:
    """
    Aggregates statistics for inference.

    To use, call `record_batch` on the results of each batch, then call
    `get_logs` to get a dictionary of statistics when you're done.
    """

    def __init__(
        self,
        area_weights: torch.Tensor,
        sigma_coordinates: SigmaCoordinates,
        n_timesteps: int,
        n_ensemble_members: int = 1,
        record_step_20: bool = False,
        log_video: bool = False,
        enable_extended_videos: bool = False,
        log_zonal_mean_images: bool = False,
        dist: Optional[Distributed] = None,
        metadata: Optional[Mapping[str, VariableMetadata]] = None,
        device: torch.device | str = None,
    ):
        """
        Args:
            area_weights: Area weights for each grid cell.
            sigma_coordinates: Data sigma coordinates
            n_timesteps: Number of timesteps of inference that will be run.
            record_step_20: Whether to record the mean of the 20th steps.
            log_video: Whether to log videos of the state evolution.
            enable_extended_videos: Whether to log videos of statistical
                metrics of state evolution
            log_zonal_mean_images: Whether to log zonal-mean images (hovmollers) with a
                time dimension.
            dist: Distributed object to use for metric aggregation.
            metadata: Mapping of variable names their metadata that will
                used in generating logged image captions.
        """
        self._is_ensemble = n_ensemble_members > 1
        device = device if device is not None else get_device()
        kwargs = dict(
            area_weights=area_weights.to(device),
            dist=dist,
            is_ensemble=self._is_ensemble,
            device=device,
        )
        self._aggregators: Dict[str, _Aggregator] = {
            "mean": MeanAggregator(target="denorm", n_timesteps=n_timesteps, **kwargs),
            "mean_norm": MeanAggregator(target="norm", n_timesteps=n_timesteps, **kwargs),
            "time_mean": TimeMeanAggregator(area_weights, dist=dist, metadata=metadata, is_ensemble=self._is_ensemble),
        }
        if record_step_20:
            self._aggregators["mean_step_20"] = OneStepMeanAggregator(target_time=20, **kwargs)
        if log_video:
            self._aggregators["video"] = VideoAggregator(
                n_timesteps=n_timesteps,
                enable_extended_videos=enable_extended_videos,
                dist=dist,
                metadata=metadata,
            )
        if log_zonal_mean_images and not self._is_ensemble:
            self._aggregators["zonal_mean"] = ZonalMeanAggregator(
                n_timesteps=n_timesteps, dist=dist, metadata=metadata
            )
        if n_timesteps is not None:
            potential_timesteps = [20, 500, 1400, 5000, 10_000, 14_000, 24_000, 34_000, 43_000]
            for t in potential_timesteps:
                if n_timesteps >= t:
                    self._aggregators[f"snapshot/t{t}"] = SnapshotAggregator(
                        is_ensemble=self._is_ensemble, target_time=t
                    )

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
        if len(target_data) == 0:
            raise ValueError("No data in target_data")
        if len(gen_data) == 0:
            raise ValueError("No data in gen_data")
        for aggregator in self._aggregators.values():
            try:
                aggregator.record_batch(
                    loss=loss,
                    target_data=target_data,
                    gen_data=gen_data,
                    target_data_norm=target_data_norm,
                    gen_data_norm=gen_data_norm,
                    i_time_start=i_time_start,
                )
            except Exception as e:
                print("---------------------> Error in aggregator", aggregator, i_time_start)
                raise e

    @torch.no_grad()
    def get_logs(self, label: str):
        """
        Returns logs as can be reported to WandB.

        Args:
            label: Label to prepend to all log keys.
        """
        logs = {}
        for name, aggregator in self._aggregators.items():
            try:
                logs.update(aggregator.get_logs(label=name))
            except RuntimeError as e:
                print(f"---------------------> Error in aggregator {name}: {e}")
                pass
        logs = {f"{label}/{key}": val for key, val in logs.items()}
        return logs

    @torch.no_grad()
    def get_inference_logs(self, label: str) -> List[Dict[str, Union[float, int]]]:
        """
        Returns a list of logs to report to WandB.

        This is done because in inference, we use the wandb step
        as the time step, meaning we need to re-organize the logged data
        from tables into a list of dictionaries.
        """
        return to_inference_logs(self.get_logs(label=label))

    @torch.no_grad()
    def get_datasets(self, aggregator_whitelist: Optional[Iterable[str]] = None) -> Dict[str, xr.Dataset]:
        """
        Args:
            aggregator_whitelist: aggregator names to include in the output. If
                None, return all the datasets associated with all aggregators.
        """
        if aggregator_whitelist is None:
            aggregators = self._aggregators.keys()
        else:
            aggregators = aggregator_whitelist
        datasets = dict()
        for name in aggregators:
            if name in self._aggregators.keys():
                datasets[name] = self._aggregators[name].get_dataset()

        return datasets


def to_inference_logs(log: Mapping[str, Union[Table, float, int]]) -> List[Dict[str, Union[float, int]]]:
    # we have a dictionary which contains WandB tables
    # which we will convert to a list of dictionaries, one for each
    # row in the tables. Any scalar values will be reported in the last
    # dictionary.
    n_rows = 0
    for val in log.values():
        if isinstance(val, Table):
            n_rows = max(n_rows, len(val.data))
    logs: List[Dict[str, Union[float, int]]] = []
    for i in range(n_rows):
        logs.append({})
    for key, val in log.items():
        if isinstance(val, Table):
            for i, row in enumerate(val.data):
                for j, col in enumerate(val.columns):
                    key_without_table_name = key[: key.rfind("/")]
                    logs[i][f"{key_without_table_name}/{col}"] = row[j]
        else:
            logs[-1][key] = val
    return logs


def table_to_logs(table: Table) -> List[Dict[str, Union[float, int]]]:
    """
    Converts a WandB table into a list of dictionaries.
    """
    logs = []
    for row in table.data:
        logs.append({table.columns[i]: row[i] for i in range(len(row))})
    return logs
