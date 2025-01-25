"""
This file contains code for computing metrics of single variables on batches of data,
and aggregating them into a single metric value. The functions here mainly exist
to turn metric functions that may have different APIs into a common API,
so that they can be iterated over and called in the same way in a loop.
"""

from typing import Literal, Optional, Protocol

import torch

from src.evaluation.metrics import Dimension


class ReducedMetric(Protocol):
    """Used to record a metric value on batches of data (potentially out-of-memory)
    and then get the total metric at the end.
    """

    def record(self, target: torch.Tensor, gen: torch.Tensor):
        """
        Update metric for a batch of data.
        """
        ...

    def get(self) -> torch.Tensor:
        """
        Get the total metric value, not divided by number of recorded batches.
        """
        ...


class AreaWeightedFunction(Protocol):
    """
    A function that computes a metric on the true and predicted values,
    weighted by area.
    """

    def __call__(
        self,
        truth: torch.Tensor,
        predicted: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        dim: Dimension = (),
    ) -> torch.Tensor: ...


class AreaWeightedSingleTargetFunction(Protocol):
    """
    A function that computes a metric on a single value, weighted by area.
    """

    def __call__(
        self,
        tensor: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        dim: Dimension = (),
    ) -> torch.Tensor: ...


def compute_metric_on(
    source: Literal["preds", "targets"], metric: AreaWeightedSingleTargetFunction
) -> AreaWeightedFunction:
    """Turns a single-target metric function
    (computed on only the generated or target data) into a function that takes in
    both the generated and target data as arguments, as required for the APIs
    which call generic metric functions.
    """

    def metric_wrapper(
        truth: torch.Tensor,
        predicted: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        dim: Dimension = (),
    ) -> torch.Tensor:
        if source == "preds":
            return metric(predicted, weights=weights, dim=dim)
        elif source == "targets":
            return metric(truth, weights=weights, dim=dim)

    return metric_wrapper


class AreaWeightedReducedMetric:
    """
    A wrapper around an area-weighted metric function.
    """

    def __init__(
        self,
        area_weights: Optional[torch.Tensor],
        device: torch.device,
        compute_metric: AreaWeightedFunction,
        dim: Dimension = (-2, -1),
    ):
        self._area_weights = area_weights.to(device) if area_weights is not None else None
        self._compute_metric = compute_metric
        self._total = None
        self._device = device
        self._dim = dim

    def record(self, targets: torch.Tensor, preds: torch.Tensor, batch_dim: int = 0, **kwargs):
        """Add a batch of data to the metric.

        Args:
            targets: Target data. Should have shape [batch, time, height, width].
            preds: Generated data. Should have shape [batch, time, height, width].
            batch_dim: The dimension of the batch axis over which to average the metric.
        """
        # dim=(-2, -1) means average over the two spatial dimensions
        # dim=batch_dim works usually too, but some data may have other non-spatial dimensions
        new_value = self._compute_metric(
            truth=targets, predicted=preds, weights=self._area_weights, dim=self._dim, **kwargs
        ).mean(dim=None)
        # assert new_value.dim() == 0, f"Expected scalar value, got {new_value}"
        if self._total is None:
            self._total = torch.zeros_like(new_value, device=targets.device)
        self._total += new_value

    def get(self) -> torch.Tensor:
        """Returns the metric."""
        return self._total
