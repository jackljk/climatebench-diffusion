from __future__ import annotations

from abc import ABC
from typing import Any, Callable, Dict, Mapping, Tuple

import torch

from src.evaluation.aggregators._abstract_aggregator import AbstractAggregator, _Aggregator
from src.evaluation.aggregators.snapshot import SnapshotAggregator
from src.evaluation.aggregators.spectra import SpectraAggregator
from src.evaluation.aggregators.timestepwise import MeanAggregator
from src.utilities.utils import get_logger


log = get_logger(__name__)


class ListAggregator(AbstractAggregator, ABC):
    def __init__(
        self,
        aggregators: list[AbstractAggregator],
        **kwargs,
    ):
        super().__init__(**kwargs)
        agg_names = [agg.name for agg in aggregators]
        # Set to self.name if all aggregators have the same name
        if self.name is None:
            self.name = agg_names[0] if all(name == agg_names[0] for name in agg_names) else None
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

    def _get_logs(self, **kwargs) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        logs_values = {}
        logs_media = {}
        logs_own_xaxis = {}
        for aggregator in self._aggregators:
            logs_values_i, logs_media_i, logs_own_xaxis = aggregator.get_logs(**kwargs)
            logs_values.update(logs_values_i)
            logs_media.update(logs_media_i)
            logs_own_xaxis.update(logs_own_xaxis)
        return logs_values, logs_media, logs_own_xaxis


class OneStepAggregator(AbstractAggregator):
    """
    Aggregates statistics for the timestep pairs.

    To use, call `record_batch` on the results of each batch, then call
    `get_logs` to get a dictionary of statistics when you're done.
    """

    def __init__(
        self,
        record_metrics: bool = True,
        record_normed: bool = False,
        record_rmse: bool = True,
        record_abs_values: bool = False,  # logs absolutes mean and std of preds and targets
        use_snapshot_aggregator: bool = True,
        snapshot_var_names: list[str] = None,
        every_nth_epoch_snapshot: int = 8,
        snapshots_preprocess_fn: Callable = None,
        record_spectra: bool = False,
        spectra_var_names: list[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        snapshot_agg = mean_agg = spectra_agg = None
        if record_metrics:
            mean_agg = MeanAggregator(
                area_weights=self._area_weights,
                is_ensemble=self._is_ensemble,
                record_normed=record_normed,
                record_rmse=record_rmse,
                record_abs_values=record_abs_values,
            )
        if use_snapshot_aggregator:
            snapshot_agg = SnapshotAggregator(
                is_ensemble=self._is_ensemble,
                var_names=snapshot_var_names,
                every_nth_epoch=every_nth_epoch_snapshot,
                preprocess_fn=snapshots_preprocess_fn,
            )

        if record_spectra:
            spectra_agg = SpectraAggregator(
                is_ensemble=self._is_ensemble,
                var_names=spectra_var_names,
                coords=self.coords,
                data_to_log="targets" if record_spectra == "targets" else "preds",
            )

        self._aggregators: Dict[str, _Aggregator] = {
            "snapshot": snapshot_agg,
            "mean": mean_agg,
            "spectra": spectra_agg,
        }

    @torch.inference_mode()
    def _record_batch(
        self,
        target_data: Mapping[str, torch.Tensor],
        gen_data: Mapping[str, torch.Tensor],
        target_data_norm: Mapping[str, torch.Tensor],
        gen_data_norm: Mapping[str, torch.Tensor],
        metadata: Mapping[str, Any] = None,
    ):
        if len(target_data) == 0:
            raise ValueError("No data in target_data")
        if len(gen_data) == 0:
            raise ValueError("No data in gen_data")

        for k, aggregator in self._aggregators.items():
            if aggregator is None:
                continue
            aggregator.record_batch(
                target_data=target_data,
                gen_data=gen_data,
                target_data_norm=target_data_norm,
                gen_data_norm=gen_data_norm,
                metadata=metadata,
            )

    @torch.inference_mode()
    def _get_logs(self, **kwargs) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
        """
        Returns logs as can be reported to WandB.

        Args:
            label: Label to prepend to all log keys.
        """
        logs, logs_media, logs_own_xaxis = {}, {}, {}
        for agg_type, agg in self._aggregators.items():
            if agg is None:
                continue
            try:
                logs_i_all = agg.get_logs(**kwargs)
            except ValueError as e:
                log.error(
                    f"Aggregator ``{self.name}/{agg_type}`` has problems.\n" f"Did you forget to record any batches?"
                )
                raise e
            if not isinstance(logs_i_all, tuple):
                assert logs_i_all is None or len(logs_i_all) == 0, f"Expected one dict, got {logs_i_all=}"
                continue

            logs_i, logs_media_i, logs_own_xaxis_i = logs_i_all
            logs.update(logs_i)
            logs_media.update(logs_media_i)
            logs_own_xaxis.update(logs_own_xaxis_i)

        return logs, logs_media, logs_own_xaxis
