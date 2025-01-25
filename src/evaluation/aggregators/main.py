from __future__ import annotations

from abc import ABC
from typing import Any, Callable, Dict, Mapping, Protocol, Tuple

import torch

from src.evaluation.aggregators._abstract_aggregator import AbstractAggregator
from src.evaluation.aggregators.snapshot import SnapshotAggregator
from src.evaluation.aggregators.timestepwise import MeanAggregator
from src.utilities.utils import get_logger


log = get_logger(__name__)


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


class ListAggregator(AbstractAggregator, ABC):
    def __init__(
        self,
        aggregators: list[AbstractAggregator],
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert self.name is None, f"ListAggregator {self.name} should not have a name"
        assert self._area_weights is None, f"ListAggregator {self.name} should not have area weights"

        self._aggregators = aggregators
        for i, aggregator in enumerate(self._aggregators):
            assert isinstance(aggregator, AbstractAggregator), f"Aggregator {i} is not an AbstractAggregator"
            assert aggregator.name is not None, f"Aggregator {i}: {aggregator} has no name"

    def record_batch(self, **kwargs) -> None:
        for aggregator in self._aggregators:
            aggregator.record_batch(**kwargs)

    def _record_batch(self, **kwargs) -> None:
        raise NotImplementedError("ListAggregator should not be called directly")

    def _get_logs(self, **kwargs) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        logs_values = {}
        logs_media = {}
        for aggregator in self._aggregators:
            logs_values_i, logs_media_i = aggregator.get_logs(prefix=None, **kwargs)
            logs_values.update(logs_values_i)
            logs_media.update(logs_media_i)
        return logs_values, logs_media


class OneStepAggregator(AbstractAggregator):
    """
    Aggregates statistics for the timestep pairs.

    To use, call `record_batch` on the results of each batch, then call
    `get_logs` to get a dictionary of statistics when you're done.
    """

    def __init__(
        self,
        use_snapshot_aggregator: bool = True,
        record_normed: bool = False,
        record_rmse: bool = True,
        record_abs_values: bool = False,  # logs absolutes mean and std of preds and targets
        snapshot_var_names: list[str] = None,
        every_nth_epoch_snapshot: int = 8,
        snapshots_preprocess_fn: Callable = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if use_snapshot_aggregator:
            self._snapshot = SnapshotAggregator(
                is_ensemble=self._is_ensemble,
                var_names=snapshot_var_names,
                every_nth_epoch=every_nth_epoch_snapshot,
                preprocess_fn=snapshots_preprocess_fn,
            )
        else:
            self._snapshot = None

        self._mean = MeanAggregator(
            area_weights=self._area_weights,
            is_ensemble=self._is_ensemble,
            record_normed=record_normed,
            record_rmse=record_rmse,
            record_abs_values=record_abs_values,
        )
        self._aggregators: Mapping[str, _Aggregator] = {
            "snapshot": self._snapshot,
            "mean": self._mean,
        }

    @torch.inference_mode()
    def _record_batch(
        self,
        target_data: Mapping[str, torch.Tensor],
        gen_data: Mapping[str, torch.Tensor],
        target_data_norm: Mapping[str, torch.Tensor],
        gen_data_norm: Mapping[str, torch.Tensor],
        metadata: Mapping[str, Any],
    ):
        if len(target_data) == 0:
            raise ValueError("No data in target_data")
        if len(gen_data) == 0:
            raise ValueError("No data in gen_data")

        for k, aggregator in self._aggregators.items():
            if aggregator is None:
                continue
            if aggregator == self._snapshot:
                aggregator.record_batch(
                    target_data=target_data,
                    gen_data=gen_data,
                    target_data_norm=target_data_norm,
                    gen_data_norm=gen_data_norm,
                    metadata=metadata,
                )
            else:
                aggregator.record_batch(
                    target_data=target_data,
                    gen_data=gen_data,
                    target_data_norm=target_data_norm,
                    gen_data_norm=gen_data_norm,
                )

    @torch.inference_mode()
    def _get_logs(self, **kwargs) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Returns logs as can be reported to WandB.

        Args:
            label: Label to prepend to all log keys.
        """
        try:
            logs = self._mean.get_logs(label="", **kwargs)
        except ValueError as e:
            raise ValueError(
                f"Aggregator ``{self.name}`` has problems with mean sub-aggregator.\n"
                f"Did you forget to record any batches?"
            ) from e

        if self._snapshot is not None:
            logs_media = self._snapshot.get_logs(**kwargs)
            # logs_media = {f"snapshot/{key}": val for key, val in logs_media.items()}
        else:
            logs_media = {}
        for agg_label, agg in self._aggregators.items():
            if agg is None or agg_label in ["mean", "snapshot"]:
                continue
            logs.update(agg.get_logs(label=agg_label, **kwargs))
            # logs.update({f"{label}/{key}": float(val) for key, val in agg.get_logs(label=agg_label).items()})
        return logs, logs_media
