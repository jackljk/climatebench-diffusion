from typing import Iterable, Optional, Union

import numpy as np
import torch
from typing_extensions import TypeAlias


Dimension: TypeAlias = Union[int, Iterable[int]]
Array: TypeAlias = Union[np.ndarray, torch.Tensor]

GRAVITY = 9.80665  # m/s^2


def spherical_area_weights(lats: Array, num_lon: int) -> torch.Tensor:
    """Computes area weights given the latitudes of a regular lat-lon grid.

    Args:
        lats: tensor of shape (num_lat,) with the latitudes of the cell centers.
        num_lon: Number of longitude points.
        device: Device to place the tensor on.

    Returns:
        a torch.tensor of shape (num_lat, num_lon).
    """
    if isinstance(lats, np.ndarray):
        lats = torch.from_numpy(lats)
    weights = torch.cos(torch.deg2rad(lats)).repeat(num_lon, 1).t()
    weights /= weights.sum()
    return weights


def weighted_mean(
    tensor: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    dim: Dimension = (),
    keepdim: bool = False,
) -> torch.Tensor:
    """Computes the weighted mean across the specified list of dimensions.

    Args:
        tensor: torch.Tensor
        weights: Weights to apply to the mean.
        dim: Dimensions to compute the mean over.
        keepdim: Whether the output tensor has `dim` retained or not.

    Returns:
        a tensor of the weighted mean averaged over the specified dimensions `dim`.
    """
    if weights is None:
        return tensor.mean(dim=dim, keepdim=keepdim)
    return (tensor * weights).sum(dim=dim, keepdim=keepdim) / weights.expand(tensor.shape).sum(
        dim=dim, keepdim=keepdim
    )


def weighted_std(
    tensor: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    dim: Dimension = (),
) -> torch.Tensor:
    """Computes the weighted standard deviation across the specified list of dimensions.

    Computed by first computing the weighted variance, then taking the square root.

    weighted_variance = weighted_mean((tensor - weighted_mean(tensor)) ** 2)) ** 0.5

    Args:
        tensor: torch.Tensor
        weights: Weights to apply to the variance.
        dim: Dimensions to compute the standard deviation over.

    Returns:
        a tensor of the weighted standard deviation over the
            specified dimensions `dim`.
    """
    if weights is None:
        weights = torch.tensor(1.0, device=tensor.device)

    mean = weighted_mean(tensor, weights=weights, dim=dim, keepdim=True)
    variance = weighted_mean((tensor - mean) ** 2, weights=weights, dim=dim)
    return torch.sqrt(variance)


def weighted_mean_bias(
    truth: torch.Tensor,
    predicted: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    dim: Dimension = (),
) -> torch.Tensor:
    """Computes the mean bias across the specified list of dimensions assuming
    that the weights are applied to the last dimensions, e.g. the spatial dimensions.

    Args:
        truth: torch.Tensor
        predicted: torch.Tensor
        dim: Dimensions to compute the mean over.
        weights: Weights to apply to the mean.

    Returns:
        a tensor of the mean biases averaged over the specified dimensions `dim`.
    """
    assert truth.shape == predicted.shape, "Truth and predicted should have the same shape."
    bias = predicted - truth
    return weighted_mean(bias, weights=weights, dim=dim)


def root_mean_squared_error(
    truth: torch.Tensor,
    predicted: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    dim: Dimension = (),
) -> torch.Tensor:
    """
    Computes the weighted global RMSE over all variables. Namely, for each variable:

        sqrt((weights * ((xhat - x) ** 2)).mean(dims))

    If you want to compute the RMSE over the time dimension, then pass in
    `truth.mean(time_dim)` and `predicted.mean(time_dim)` and specify `dims=space_dims`.

    Args:
        truth: torch.Tensor whose last dimensions are to be weighted
        predicted: torch.Tensor whose last dimensions are to be weighted
        weights: torch.Tensor to apply to the squared bias.
        dim: Dimensions to average over.

    Returns:
        a tensor of shape (variable,) of weighted RMSEs.
    """
    assert truth.shape == predicted.shape, "Truth and predicted should have the same shape."
    sq_bias = torch.square(predicted - truth)
    return weighted_mean(sq_bias, weights=weights, dim=dim).sqrt()


def ensemble_spread(
    ensemble: torch.Tensor, weights: Optional[torch.Tensor] = None, corr_factor: bool = True, dim: Dimension = ()
) -> torch.Tensor:
    """Square root of the ensemble variance. See Fortuin et al. (2013) for more details."""
    spread = weighted_mean(ensemble.var(dim=0), weights=weights, dim=dim).sqrt()
    if corr_factor:
        n_mems = ensemble.shape[0]
        spread *= ((n_mems + 1) / n_mems) ** 0.5
    return spread


def spread_skill_ratio(
    truth: torch.Tensor,
    predicted: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    dim: Dimension = (),
) -> torch.Tensor:
    assert truth.shape == predicted.shape[1:]  # ensemble ~ first axis
    weighted_rmse = root_mean_squared_error(truth, predicted.mean(dim=0), weights=weights, dim=dim)
    weighted_spread = ensemble_spread(predicted, weights=weights, dim=dim)
    return weighted_spread / weighted_rmse


def weighted_crps(
    truth: torch.Tensor,  # TRUTH
    predicted: torch.Tensor,  # FORECAST
    weights: Optional[torch.Tensor] = None,
    dim: Dimension = (),
    reduction="mean",
    biased: bool = False,
) -> torch.Tensor:
    """
    .. Author: Salva Rühling Cachay

    pytorch version of https://github.com/TheClimateCorporation/properscoring/blob/master/properscoring/_crps.py#L187

    This implementation is based on the identity:
    .. math::
        CRPS(F, x) = E_F|X - x| - 1/2 * E_F|X - X'|
    where X and X' denote independent random variables drawn from the forecast
    distribution F, and E_F denotes the expectation value under F.

    We use the fair, unbiased formulation of the ensemble CRPS, which is better for small ensembles.
    Basically, we use n_members * (n_members - 1) instead of n_members**2 to average over the ensemble spread.
    See Zamo & Naveau (2018; https://doi.org/10.1007/s11004-017-9709-7) for details.

    Alternative implementation: https://github.com/NVIDIA/modulus/pull/577/files
    """
    assert truth.ndim == predicted.ndim - 1, f"observations.shape={truth.shape}, predictions.shape={predicted.shape}"
    assert truth.shape == predicted.shape[1:]  # ensemble ~ first axis
    n_members = predicted.shape[0]
    if n_members == 1:
        return weighted_mean((predicted - truth).abs(), weights=weights, dim=dim)

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


def gradient_magnitude(tensor: torch.Tensor, dim: Dimension = ()) -> torch.Tensor:
    """Compute the magnitude of gradient across the specified dimensions."""
    gradients = torch.gradient(tensor, dim=dim)
    return torch.sqrt(sum([g**2 for g in gradients]))


def weighted_mean_gradient_magnitude(
    tensor: torch.Tensor, weights: Optional[torch.Tensor] = None, dim: Dimension = ()
) -> torch.Tensor:
    """Compute weighted mean of gradient magnitude across the specified dimensions."""
    return weighted_mean(gradient_magnitude(tensor, dim), weights=weights, dim=dim)


def gradient_magnitude_percent_diff(
    truth: torch.Tensor,
    predicted: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    dim: Dimension = (),
    is_ensemble_prediction: bool = False,
) -> torch.Tensor:
    """Compute the percent difference of the weighted mean gradient magnitude across
    the specified dimensions."""
    truth_grad_mag = weighted_mean_gradient_magnitude(truth, weights, dim)
    if is_ensemble_prediction:
        predicted_grad_mag = 0
        for ens_i, pred in enumerate(predicted):
            predicted_grad_mag += weighted_mean_gradient_magnitude(pred, weights, dim)
        predicted_grad_mag /= predicted.shape[0]
    else:
        assert truth.shape == predicted.shape, "Truth and predicted should have the same shape."
        predicted_grad_mag = weighted_mean_gradient_magnitude(predicted, weights, dim)
    return 100 * (predicted_grad_mag - truth_grad_mag) / truth_grad_mag


def rmse_of_time_mean(
    truth: torch.Tensor,
    predicted: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    time_dim: Dimension = 0,
    spatial_dims: Dimension = (-2, -1),
) -> torch.Tensor:
    """Compute the RMSE of the time-average given truth and predicted.

    Args:
        truth: truth tensor
        predicted: predicted tensor
        weights: weights to use for computing spatial RMSE
        time_dim: time dimension
        spatial_dims: spatial dimensions over which RMSE is calculated

    Returns:
        The RMSE between the time-mean of the two input tensors. The time and
            spatial dims are reduced.
    """
    truth_time_mean = truth.mean(dim=time_dim)
    predicted_time_mean = predicted.mean(dim=time_dim)
    ret = root_mean_squared_error(truth_time_mean, predicted_time_mean, weights=weights, dim=spatial_dims)
    return ret


def time_and_global_mean_bias(
    truth: torch.Tensor,
    predicted: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    time_dim: Dimension = 0,
    spatial_dims: Dimension = (-2, -1),
) -> torch.Tensor:
    """Compute the global- and time-mean bias given truth and predicted.

    Args:
        truth: truth tensor
        predicted: predicted tensor
        weights: weights to use for computing the global mean
        time_dim: time dimension
        spatial_dims: spatial dimensions over which global mean is calculated

    Returns:
        The global- and time-mean bias between the predicted and truth tensors. The
            time and spatial dims are reduced.
    """
    truth_time_mean = truth.mean(dim=time_dim)
    predicted_time_mean = predicted.mean(dim=time_dim)
    result = weighted_mean(predicted_time_mean - truth_time_mean, weights=weights, dim=spatial_dims)
    return result


def vertical_integral(
    integrand: torch.Tensor,
    surface_pressure: torch.Tensor,
    sigma_grid_offsets_ak: torch.Tensor,
    sigma_grid_offsets_bk: torch.Tensor,
) -> torch.Tensor:
    """Computes a vertical integral, namely:

    (1 / g) * ∫ x dp

    where
    - g = acceleration due to gravity
    - x = integrad
    - p = pressure level

    Args:
        integrand (lat, lon, vertical_level), (kg/kg)
        surface_pressure: (lat, lon), (Pa)
        sigma_grid_offsets_ak: Sorted sigma grid offsets ak, (vertical_level + 1,)
        sigma_grid_offsets_bk: Sorted sigma grid offsets bk, (vertical_level + 1,)

    Returns:
        Vertical integral of the integrand (lat, lon).
    """
    ak, bk = sigma_grid_offsets_ak, sigma_grid_offsets_bk
    if ak.device != surface_pressure.device:
        ak = ak.to(surface_pressure.device)
    if bk.device != surface_pressure.device:
        bk = bk.to(surface_pressure.device)
    if ak.device != integrand.device or ak.device != surface_pressure.device:
        raise ValueError(
            f"sigma_grid_offsets_ak.device ({ak.device}), "
            f"sigma_grid_offsets_bk.device ({bk.device}), "
            f"integrand.device ({integrand.device}), "
            f"surface_pressure.device ({surface_pressure.device}) must be the same."
        )
    pressure_thickness = ((ak + (surface_pressure.unsqueeze(-1) * bk))).diff(dim=-1)  # Pa
    integral = torch.sum(pressure_thickness * integrand, axis=-1)  # type: ignore
    return 1 / GRAVITY * integral


def surface_pressure_due_to_dry_air(
    specific_total_water: torch.Tensor,
    surface_pressure: torch.Tensor,
    sigma_grid_offsets_ak: torch.Tensor,
    sigma_grid_offsets_bk: torch.Tensor,
) -> torch.Tensor:
    """Computes the dry air (Pa).

    Args:
        specific_total_water (lat, lon, vertical_level), (kg/kg)
        surface_pressure: (lat, lon), (Pa)
        sigma_grid_offsets_ak: Sorted sigma grid offsets ak, (vertical_level + 1,)
        sigma_grid_offsets_bk: Sorted sigma grid offsets bk, (vertical_level + 1,)

    Returns:
        Vertically integrated dry air (lat, lon) (Pa)
    """

    num_levels = len(sigma_grid_offsets_ak) - 1

    if num_levels != len(sigma_grid_offsets_bk) - 1 or num_levels != specific_total_water.shape[-1]:
        raise ValueError(("Number of vertical levels in ak, bk, and specific_total_water must" "be the same."))

    total_water_path = vertical_integral(
        specific_total_water,
        surface_pressure,
        sigma_grid_offsets_ak,
        sigma_grid_offsets_bk,
    )
    dry_air = surface_pressure - GRAVITY * total_water_path
    return dry_air
