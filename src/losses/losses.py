from functools import partial
from typing import Dict, Iterable, Optional, Union

import torch
import torch.nn as nn
from torch import Tensor

from src.evaluation.metrics import weighted_mean
from src.utilities.utils import get_logger


log = get_logger(__name__)


class LpLoss(torch.nn.Module):
    def __init__(
        self,
        p=2,
        relative: bool = True,
        weights: Optional[Tensor] = None,
        weighted_dims: Union[int, Iterable[int]] = (),
    ):
        """
        Args:
            p: Lp-norm type. For example, p=1 for L1-norm, p=2 for L2-norm.
            relative: If True, compute the relative Lp-norm, i.e. ||x - y||_p / ||y||_p.
        """
        super(LpLoss, self).__init__()

        if p <= 0:
            raise ValueError("Lp-norm type should be positive")

        self.p = p
        self.loss_func = self.rel if relative else self.abs
        self.weights = weights

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, weights):
        self._weights = weights
        if weights is not None:
            self.mean_func = partial(weighted_mean, weights=weights)
        else:
            self.mean_func = torch.mean

    def rel(self, x, y):
        num_examples = x.size()[0]
        diff_norms = torch.norm(x.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples, -1), self.p, 1)

        # print(diff_norms.shape, y_norms.shape, self.mean_func)
        return self.mean_func(diff_norms / y_norms)

    def abs(self, x, y):
        num_examples = x.size()[0]
        diff_norms = torch.norm(x.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p, 1)
        return self.mean_func(diff_norms)

    def __call__(self, x, y):
        return self.loss_func(x, y)


class CRPSLoss(torch.nn.Module):
    def forward(self, inputs, targets):
        return crps_ensemble(
            truth=targets,
            predicted=inputs,
        )


def crps_ensemble(
    truth: Tensor,  # TRUTH
    predicted: Tensor,  # FORECAST
    weights: Tensor = None,
    dim: Union[int, Iterable[int]] = (),
    reduction="mean",
) -> Tensor:
    """
    .. Author: Salva Rühling Cachay

    pytorch adaptation of https://github.com/TheClimateCorporation/properscoring/blob/master/properscoring/_crps.py#L187
    but implementing the fair, unbiased CRPS as in Zamo & Naveau (2018; https://doi.org/10.1007/s11004-017-9709-7)

    This implementation is based on the identity:
    .. math::
        CRPS(F, x) = E_F|X - x| - 1/2 * E_F|X - X'|
    where X and X' denote independent random variables drawn from the forecast
    distribution F, and E_F denotes the expectation value under F.

    We use the fair, unbiased formulation of the ensemble CRPS, which is particularly important for small ensembles.
    Anecdotically, the unbiased CRPS leads to slightly smaller (i.e. "better") values than the biased version.
    Basically, we use n_members * (n_members - 1) instead of n_members**2 to average over the ensemble spread.
    See Zamo & Naveau (2018; https://doi.org/10.1007/s11004-017-9709-7) for details.

    Alternative implementation: https://github.com/NVIDIA/modulus/pull/577/files
    """
    assert truth.ndim == predicted.ndim - 1, f"{truth.shape=}, {predicted.shape=}"
    assert truth.shape == predicted.shape[1:]  # ensemble ~ first axis
    n_members = predicted.shape[0]
    skill = (predicted - truth).abs().mean(dim=0)
    # insert new axes so forecasts_diff expands with the array broadcasting
    # torch.unsqueeze(predictions, 0) has shape (1, E, ...)
    # torch.unsqueeze(predictions, 1) has shape (E, 1, ...)
    forecasts_diff = torch.unsqueeze(predicted, 0) - torch.unsqueeze(predicted, 1)
    # Forecasts_diff has shape (E, E, ...)
    # Old version: score += - 0.5 * forecasts_diff.abs().mean(dim=(0, 1))
    # Using n_members * (n_members - 1) instead of n_members**2 is the fair, unbiased CRPS. Better for small ensembles.
    spread = forecasts_diff.abs().sum(dim=(0, 1)) / (n_members * (n_members - 1))
    crps = skill - 0.5 * spread
    # score has shape (...)  (same as observations)
    if reduction == "none":
        return crps
    assert reduction == "mean", f"Unknown reduction {reduction}"
    if weights is not None:  # weighted mean
        crps = (crps * weights).sum(dim=dim) / weights.expand(crps.shape).sum(dim=dim)
    else:
        crps = crps.mean(dim=dim)
    return crps


class AbstractWeightedLoss(torch.nn.Module):
    def __init__(
        self,
        weights: Tensor = None,
        reduction: str = "mean",
        learned_var_dim_name_to_idx_and_n_dims: Dict[int, int] = None,
        learn_per_dim: bool = True,
        verbose: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.weights = weights
        self.reduction = reduction
        self.learn_per_dim = learn_per_dim
        learned_var_dim_name_to_idx_and_n_dims = learned_var_dim_name_to_idx_and_n_dims or {}
        # If more than 1, sort by dim index (the first element of value, a tuple)
        # if len(learned_var_dim_name_to_idx_and_n_dims) >= 1:
        # learned_var_dim_name_to_idx_and_n_dims = dict(sorted(learned_var_dim_name_to_idx_and_n_dims.items(), key=lambda x: x[1][0]))
        # dim0_idx = next(iter(learned_var_dim_name_to_idx_and_n_dims.values()))[0]
        # assert all(dim0_idx <= idx for idx, _ in learned_var_dim_name_to_idx_and_n_dims.values()), f"Learned variance dimensions must be sorted by index. Got {learned_var_dim_name_to_idx_and_n_dims=}"
        # log.info(f">>>> {learned_var_dim_name_to_idx_and_n_dims=}")
        self.learned_var_dim_name_to_idx_and_n_dims = learned_var_dim_name_to_idx_and_n_dims
        self.n_logvar_dims = len(learned_var_dim_name_to_idx_and_n_dims)
        # Assert all dim idxs are unique
        dim_idxs = [idx for idx, _ in self.learned_var_dim_name_to_idx_and_n_dims.values()]
        dim_sizes = [n_dims for _, n_dims in self.learned_var_dim_name_to_idx_and_n_dims.values()]
        assert len(dim_idxs) == len(set(dim_idxs)), f"Dimension indices must be unique. {dim_idxs=}"

        if self.n_logvar_dims == 1:
            # Save the dimension idx for the log var
            assert len(dim_idxs) == 1, f"Only one learned variance dimension is supported. {dim_idxs=}"
            self._dim_idxs_from = dim_idxs[0]
            self._dim_idxs_to = 0
        elif self.n_logvar_dims >= 2:
            # Save tuple of the dimension idxs for the log vars
            self._dim_idxs_from = tuple(dim_idxs)
            self._dim_idxs_to = tuple(range(self.n_logvar_dims))

        for dim_name, (dim_idx, n_dims) in self.learned_var_dim_name_to_idx_and_n_dims.items():
            assert n_dims > 0, f"Number of dimensions must be positive. {n_dims=}"
            # assert dim_idx >= 0, f"Dimension index must be non-negative. {dim_idx=}"
            # Register in __init__ to make it part of the model's parameters
            # We implement the variance-weighted loss by simply learning a list of scalars, one per frame
            if learn_per_dim:
                setattr(self, f"{dim_name}_logvar", nn.Parameter(torch.zeros(n_dims, requires_grad=True)))
                if verbose:
                    log.info(f"Using loss with learned ``{dim_name}`` logvar with {n_dims=}, {dim_idx=}.")

        if not learn_per_dim:
            # Learn a scalar for each entry in dim_0 x dim_1 x ... x dim_n
            assert len(dim_sizes) > 0, f"Number of dimensions must be positive. {dim_sizes=}"
            self.logvars = nn.Parameter(torch.zeros(*dim_sizes, requires_grad=True))
            if verbose:
                log.info(f"Using loss function with learned logvar with dimensions {dim_sizes}.")

        self._channel_dim = None
        if "channels" in self.learned_var_dim_name_to_idx_and_n_dims:
            self._channel_dim = self.learned_var_dim_name_to_idx_and_n_dims["channels"][0]
            # assert self._channel_dim == dim0_idx, f"Channel dimension must be the first learned variance dimension. {self._channel_dim=}, {dim0_idx=}"

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, weights):
        self._weights = weights

    @property
    def channels_logvar_vector(self):
        if self._channel_dim is None:
            raise AttributeError("No learned variance dimension named 'channels' found.")
        if self.learn_per_dim:
            return self.channels_logvar
        # Take mean over non-channel dimensions
        return self.logvars.mean(dim=tuple(range(1, self.logvars.ndim)))

    def weigh_loss(self, loss, add_weight=None, multiply_weight=None):
        if self.weights is not None:
            weights = self.weights
        else:
            assert add_weight is not None or multiply_weight is not None, "No weights provided. Please set or provide."
            weights = torch.ones_like(loss)

        # Shapes are e.g.: inputs: (B, C, H, W), targets: (B, C, H, W), weights: (H, W) or (C, H, W)
        # Similarly, inputs, targets and weights can have shapes (B, C, S), (B, C, S), (S) or (C, S),
        # where S is the sequence length
        if add_weight is not None:
            # Don't do += because it may fail when broadcasting weights for different shapes
            weights = weights + add_weight

        if multiply_weight is not None:
            diff_shape = len(multiply_weight.shape) - len(weights.shape)
            if diff_shape != 0:
                # Add singleton dimensions to multiply_weight to match weights (based on diff_shape)
                for _ in range(diff_shape):
                    weights = weights.unsqueeze(0)  # Add batch dimension to weights # todo: do somewhere else

                # if len(weights.shape) < len(multiply_weight.shape):
                #     # Check if dim=1 is channel or time dimension
                #     if self.channel_dim is not None and weights.shape[self.channel_dim] == self.num_channels:
                #         # Add singleton channel dimension to weights at dim=self.channel_dim
                #         weights = weights.unsqueeze(2) #self.time_dim)
                #     # elif self.time_dim is not None and weights.shape[self.time_dim] == self.num_times:
                #         # Add singleton time dimension to weights at dim=self.time_dim
                #         # weights = weights.unsqueeze(self.time_dim)
                # # if self.time_dim is not None and weights.shape[self.time_dim] != self.num_times:
                #     # Add singleton time dimension to weights at dim=self.time_dim
                #     # weights = weights.unsqueeze(self.time_dim)
                self.weights = weights  # Update weights with new singleton dimensions to not repeat this step

            # print(f"{weights.shape=}, {multiply_weight.shape=}")
            # Don't do *= because it may fail when broadcasting weights for different shapes
            try:
                weights = weights * multiply_weight
            except RuntimeError as e:
                raise RuntimeError(f"Failed to compute {weights.shape=} * {multiply_weight.shape=}.") from e

        try:
            loss = weights * loss
        except RuntimeError as e:
            raise RuntimeError(f"Failed to multiply {weights.shape=} by {loss.shape=}.") from e

        if self.n_logvar_dims == 0:
            pass
        else:
            # Bring logvar dimensions to the front. note that the dims are sorted by index (increasing)
            loss = torch.movedim(loss, self._dim_idxs_from, self._dim_idxs_to)
            # Take mean over non-logvar dimensions
            loss = loss.mean(dim=tuple(range(self.n_logvar_dims, loss.ndim)))

            if self.n_logvar_dims == 1:
                # Only one learned variance dimension
                log_var_name = next(iter(self.learned_var_dim_name_to_idx_and_n_dims.keys()))
                log_vars = getattr(self, f"{log_var_name}_logvar")
            elif self.learn_per_dim:
                # TODO: fix this. Seems like it doesn't backpropagate through the log_vars
                log_vars = [
                    getattr(self, f"{log_var_name}_logvar")
                    for log_var_name in self.learned_var_dim_name_to_idx_and_n_dims
                ]
                log_vars = torch.outer(*log_vars)
            else:
                log_vars = self.logvars  # Learned variance for all dimensions (>=2)

            # Important check below.
            #   If specified logvar dims are misaligned (e.g. due to unexpected singleton dimensions),
            #   the loss will be broadcasted incorrectly (e.g. loss of shape (1,) will be broadcasted to log_vars shape)
            #   rather than correctly applying on a per-dimension basis.
            assert loss.shape == log_vars.shape, f"{loss.shape=}, {log_vars.shape=}, {weights.shape=}"
            # Apply the learned variance to the loss
            loss = loss / torch.exp(log_vars) + log_vars

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            raise ValueError(f"Unknown reduction {self.reduction}")

    def forward(self, preds, targets):
        raise NotImplementedError("Subclasses must implement this method.")


class WeightedMSE(AbstractWeightedLoss):
    def forward(self, preds, targets, **kwargs):
        error = self.weigh_loss((preds - targets) ** 2, **kwargs)
        return error


class WeightedMAE(AbstractWeightedLoss):
    def forward(self, preds, targets, **kwargs):
        error = self.weigh_loss((preds - targets).abs(), **kwargs)
        return error


class WeightedCRPS(AbstractWeightedLoss):
    def forward(self, preds, targets):
        error = self.weigh_loss(crps_ensemble(predicted=preds, observations=truth, reduction="none"))
        return error


def get_loss(name, reduction="mean", **kwargs):
    """Returns the loss function with the given name."""
    name = name.lower().strip().replace("-", "_")
    if name in ["l1", "mae", "mean_absolute_error"]:
        loss = nn.L1Loss(reduction=reduction, **kwargs)
    elif name in ["l2", "mse", "mean_squared_error"]:
        loss = nn.MSELoss(reduction=reduction, **kwargs)
    elif name in ["l2_rel"]:
        loss = LpLoss(p=2, relative=True, **kwargs)
    elif name in ["l1_rel"]:
        loss = LpLoss(p=1, relative=True, **kwargs)
    elif name in ["smoothl1", "smooth"]:
        loss = nn.SmoothL1Loss(reduction=reduction, **kwargs)
    elif name in ["wmse", "weighted_mse"]:
        loss = WeightedMSE(**kwargs)
    elif name in ["wmseold2"]:
        from src.losses.lossesold2 import WeightedMSE as WeightedMSEOld2

        loss = WeightedMSEOld2(**kwargs)
    elif name in ["wmae", "weighted_mae"]:
        loss = WeightedMAE(**kwargs)
    elif name in ["wcrps", "weighted_crps"]:
        loss = WeightedCRPS(**kwargs)
    # elif name in ["crps_gaussian"]:
    #     loss = CRPSGaussianLoss(reduction=reduction)
    elif name in ["crps"]:
        assert reduction == "mean", "CRPS loss only supports mean reduction"
        loss = CRPSLoss(reduction=reduction, **kwargs)
    # elif name in ["nll", "negative_log_likelihood"]:
    #     loss = NLLLoss(reduction=reduction)
    else:
        raise ValueError(f"Unknown loss function {name}")
    return loss
