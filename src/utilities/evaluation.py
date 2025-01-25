from __future__ import annotations

from collections import defaultdict
from typing import Optional, Tuple

import numpy as np
import torch


def evaluate_ensemble_prediction(
    predictions: np.ndarray,
    targets: np.ndarray,
    ensemble_dim: int = 0,
    also_per_member_metrics: bool = False,
    also_per_member_metrics_correlation: bool = True,
    mean_over_samples: bool = True,
):
    """Evaluate the predictions of an ensemble of models.

    Args:
        predictions (np.ndarray): The predictions of the ensemble, of shape (n_models, n_samples, *)
        targets (np.ndarray): The targets, of shape (n_samples, *)
        ensemble_dim (int, optional): The dimension of the ensemble. Default: 0.
        also_per_member_metrics (bool, optional): If True, also compute the metrics for each model in the ensemble.
        mean_over_samples (bool ): If True, compute the metrics over the samples dimension. Default: True.

    Returns:
        dict: A dictionary containing the evaluation metrics
    """
    assert (
        predictions.shape[1] == targets.shape[ensemble_dim]
    ), f"predictions.shape[1] ({predictions.shape[1]}) != targets.shape[0] ({targets.shape[ensemble_dim]})"
    # shape could be: preds: (10, 730, 3, 60, 60), targets: (730, 3, 60, 60), or (5, 64, 12), (64, 12)
    n_preds, n_samples = predictions.shape[:2]

    # if channel dimension is missing, add it
    if len(predictions.shape) == 3:
        predictions = predictions[:, :, np.newaxis]
    if len(targets.shape) == 2:
        targets = targets[:, np.newaxis]

    # Compute the mean prediction
    mean_preds = predictions.mean(axis=ensemble_dim)
    # first, compute the metrics for the mean prediction
    if mean_over_samples:
        mean_dims = tuple(range(mean_preds.ndim))
        mean_dims_cc = (0, 1)
    else:
        mean_dims = tuple(range(1, mean_preds.ndim))
        mean_dims_cc = 1  # take avg. over after computing the correlation

    mse_ensemble = np.mean((mean_preds - targets) ** 2, axis=mean_dims)  # shape: () or (n_samples,)
    rmse_ensemble = np.sqrt(mse_ensemble)

    mses = {"mse": mse_ensemble}
    if also_per_member_metrics:
        # next, compute the MSE for each model
        diff = predictions - targets  # shape: (n_models, n_samples, *)
        # check that diff[i] is the same as predictions[i] - targets
        assert np.allclose(diff[0], predictions[ensemble_dim] - targets)
        mses["mse_per_mem"] = np.mean(diff**2, axis=tuple(range(1, predictions.ndim)))  # shape: (n_models,)
        mses["mse_per_mem_mean"] = np.mean(mses["mse_per_mem"])

    # second, compute the correlation between the predictions of each model
    corr_ensemble = pearson_correlation_numpy(mean_preds, targets, mean_dims=mean_dims_cc)
    corrs = {"corr": corr_ensemble}
    if also_per_member_metrics_correlation:
        corr_per_mem_sum = 0 if mean_over_samples else np.zeros(n_samples)
        for i in range(n_preds):
            corr = pearson_correlation_numpy(predictions[i], targets, mean_dims=mean_dims_cc)
            corr_per_mem_sum += corr
        corrs["corr_mem_avg"] = corr_per_mem_sum / n_preds

    # third, compute the CRPS for each model
    crps = evaluate_ensemble_crps(predictions, targets, mean_over_samples=mean_over_samples)

    # compute the spread of the ensemble
    spread_skill_ratio = evaluate_ensemble_spread_skill_ratio(
        predictions, targets, skill_metric=rmse_ensemble, mean_dims=mean_dims
    )
    # Compute the spread-skill ratio l1 and l2 distances from the ideal value of 1
    ssr_l1_dist = np.abs(1 - spread_skill_ratio)
    ssr_squared_dist = (1 - spread_skill_ratio) ** 2

    # compute negative log-likelihood
    # np.var(predictions, axis=ensemble_dim)
    # nll = evaluate_ensemble_nll(mean_preds, var_preds, targets, mean_dims=mean_dims)

    to_return = {
        "ssr": spread_skill_ratio,
        "ssr_abs_dist": ssr_l1_dist,
        "ssr_squared_dist": ssr_squared_dist,
        "crps": crps,
        **mses,
        **corrs,
    }
    return to_return


def evaluate_ensemble_crps(
    ensemble_predictions: np.ndarray,
    targets: np.ndarray,
    member_dim: str = "member",
    mean_over_samples: bool = True,
) -> float | np.ndarray:
    return losses.crps_ensemble(
        predictions=torch.from_numpy(ensemble_predictions),
        observations=torch.from_numpy(targets),
    ).item()


def evaluate_ensemble_spread_skill_ratio(
    ensemble_predictions: np.ndarray, targets: np.ndarray, skill_metric: float = None, mean_dims=None
) -> float:
    """
    Compute the spread-skill ratio (SSR) of an ensemble of predictions.
    The SSR is defined as the ratio of the ensemble spread to the ensemble skill.
    The ensemble spread is the standard deviation over the ensemble members.
    Args:
        ensemble_predictions (np.ndarray): The predictions of the ensemble, of shape (n_models, n_samples, *)
        targets (np.ndarray): The targets, of shape (n_samples, *)
        skill_metric (float, optional): The skill metric to use. Defaults to None, in which case the RMSE is used.
        mean_dims (tuple, optional): The dimensions over which to compute the mean. Default: None (all dimensions).
    """
    n_mems = ensemble_predictions.shape[0]
    variance = np.var(ensemble_predictions, axis=0).mean(axis=mean_dims)
    spread = np.sqrt(variance)
    # Add correction factor sqrt((M+1)/M); see https://doi.org/10.1175/JHM-D-14-0008.1), important for small ensemble sizes
    spread *= ((n_mems + 1) / n_mems) ** 0.5

    if skill_metric is None:
        mse = evaluate_ensemble_mse(ensemble_predictions, targets, mean_dims=mean_dims)
        skill_metric = np.sqrt(mse)

    spread_skill_ratio = spread / skill_metric
    return spread_skill_ratio


def evaluate_ensemble_nll(
    mean_predictions: np.ndarray, var_predictions: np.ndarray, targets: np.ndarray, mean_dims=None
) -> float:
    """
    Compute the negative log-likelihood of an ensemble of predictions.
    """
    nll = 0.5 * np.log(2 * np.pi * var_predictions) + (targets - mean_predictions) ** 2 / (2 * var_predictions)
    return nll.mean(axis=mean_dims)


def evaluate_ensemble_mse(ensemble_predictions: np.ndarray, targets: np.ndarray, mean_dims=None) -> float:
    mean_preds = ensemble_predictions.mean(axis=0)
    mse = np.mean((mean_preds - targets) ** 2, axis=mean_dims)
    return mse


def evaluate_ensemble_corr(ensemble_predictions: np.ndarray, targets: np.ndarray) -> float:
    mean_preds = ensemble_predictions.mean(axis=0)
    corr = np.corrcoef(mean_preds.reshape(1, -1), targets.reshape(1, -1), rowvar=False)[0, 1]
    return float(corr)


def evaluate_ensemble_prediction_for_varying_members(predictions: np.ndarray, targets: np.ndarray):
    n_members, n_samples = predictions.shape[:2]
    results = defaultdict(list)
    for n in range(1, n_members + 1):
        results_n = evaluate_ensemble_prediction(predictions[:n], targets)
        # for each result, only keep the values if they are scalars and add them to the list
        for k, v in results_n.items():
            if np.isscalar(v):
                results[k] += [v]
            elif n == n_members:
                results[k] = v
    return results


def pearson_correlation(input: torch.Tensor, target: torch.Tensor, reduce_batch: bool = False):
    """Taken from pdearena/modules/loss.py"""
    B = input.size(0)
    T = input.size(1)
    input = input.reshape(B, T, -1)
    target = target.reshape(B, T, -1)
    input_mean = torch.mean(input, dim=(2), keepdim=True)
    target_mean = torch.mean(target, dim=(2), keepdim=True)
    # Unbiased since we use unbiased estimates in covariance
    input_std = torch.std(input, dim=(2), unbiased=False)
    target_std = torch.std(target, dim=(2), unbiased=False)

    corr = torch.mean((input - input_mean) * (target - target_mean), dim=2) / (input_std * target_std).clamp(
        min=torch.finfo(torch.float32).tiny
    )  # shape (B, T)
    if reduce_batch:
        corr = torch.mean(corr, dim=0)
    return corr


def pearson_correlation_numpy(predictions: np.ndarray, targets: np.ndarray, mean_dims: Optional[Tuple[int]] = None):
    B = predictions.shape[0]
    T = predictions.shape[1]
    predictions = predictions.reshape([B, T, -1])
    targets = targets.reshape([B, T, -1])
    predictions_mean = np.mean(predictions, axis=(2), keepdims=True)
    targets_mean = np.mean(targets, axis=(2), keepdims=True)
    # Unbiased since we use unbiased estimates in covariance
    predictions_std = np.std(predictions, axis=(2), ddof=1)
    targets_std = np.std(targets, axis=(2), ddof=1)

    corr = np.mean((predictions - predictions_mean) * (targets - targets_mean), axis=2) / np.maximum(
        predictions_std * targets_std, np.finfo(np.float32).tiny
    )  # shape (B, T)
    if mean_dims is not None:
        corr = np.mean(corr, axis=mean_dims)
    return corr
