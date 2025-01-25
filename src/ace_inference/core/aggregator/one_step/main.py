import inspect
from typing import Mapping, Optional, Protocol

import torch

from src.ace_inference.core.data_loading.data_typing import SigmaCoordinates, VariableMetadata

from .reduced_salva import MeanAggregator
from .snapshot import SnapshotAggregator


class _Aggregator(Protocol):
    def get_logs(self, label: str) -> Mapping[str, torch.Tensor]: ...

    def record_batch(
        self,
        loss: float,
        target_data: Mapping[str, torch.Tensor],
        gen_data: Mapping[str, torch.Tensor],
        target_data_norm: Mapping[str, torch.Tensor],
        gen_data_norm: Mapping[str, torch.Tensor],
    ) -> None: ...


class OneStepAggregator:
    """
    Aggregates statistics for the first timestep.

    To use, call `record_batch` on the results of each batch, then call
    `get_logs` to get a dictionary of statistics when you're done.
    """

    def __init__(
        self,
        area_weights: torch.Tensor,
        sigma_coordinates: SigmaCoordinates,
        is_ensemble: bool,
        use_snapshot_aggregator: bool = True,
        metadata: Optional[Mapping[str, VariableMetadata]] = None,
    ):
        self._snapshot = (
            SnapshotAggregator(is_ensemble=is_ensemble, metadata=metadata) if use_snapshot_aggregator else None
        )
        self._mean = MeanAggregator(area_weights=area_weights, is_ensemble=is_ensemble)
        self._aggregators: Mapping[str, _Aggregator] = {
            "snapshot": self._snapshot,
            "mean": self._mean,
            # "derived": DerivedMetricsAggregator(area_weights, sigma_coordinates)
        }

    @torch.no_grad()
    def record_batch(
        self,
        target_data: Mapping[str, torch.Tensor],
        gen_data: Mapping[str, torch.Tensor],
        target_data_norm: Mapping[str, torch.Tensor],
        gen_data_norm: Mapping[str, torch.Tensor],
        inputs_norm: Mapping[str, torch.Tensor] = None,
    ):
        if len(target_data) == 0:
            raise ValueError("No data in target_data")
        if len(gen_data) == 0:
            raise ValueError("No data in gen_data")
        for aggregator in self._aggregators.values():
            if aggregator is None:
                continue
            kwargs = {}
            if "inputs_norm" in inspect.signature(aggregator.record_batch).parameters:
                kwargs["inputs_norm"] = inputs_norm

            aggregator.record_batch(
                target_data=target_data,
                gen_data=gen_data,
                target_data_norm=target_data_norm,
                gen_data_norm=gen_data_norm,
                **kwargs,
            )

    @torch.no_grad()
    def get_logs(self, label: str):
        """
        Returns logs as can be reported to WandB.

        Args:
            label: Label to prepend to all log keys.
        """
        logs = {f"{label}/{key}": val for key, val in self._mean.get_logs(label="").items()}
        if self._snapshot is not None:
            logs_media = self._snapshot.get_logs(label="snapshot")
            logs_media = {f"{label}/{key}": val for key, val in logs_media.items()}
        else:
            logs_media = {}
        for agg_label, agg in self._aggregators.items():
            if agg is None or agg_label in ["mean", "snapshot"]:
                continue
            logs.update({f"{label}/{key}": float(val) for key, val in agg.get_logs(label=agg_label).items()})
        return logs, logs_media
