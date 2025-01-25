from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import Any, Dict, Mapping, Optional, Union

import numpy as np
import torch
import xarray as xr

from src.ace_inference.core.aggregator.inference.main import InferenceAggregator
from src.ace_inference.core.aggregator.null import NullAggregator
from src.ace_inference.core.data_loading.data_typing import GriddedData
from src.ace_inference.core.device import get_device
from src.ace_inference.core.optimization import NullOptimization
from src.ace_inference.core.stepper import SingleModuleStepper, SteppedData
from src.ace_inference.inference.data_writer.main import DataWriter, NullDataWriter
from src.ace_inference.inference.derived_variables import (
    compute_derived_quantities,
    # compute_stepped_derived_quantities,
)
from src.utilities.normalization import StandardNormalizer


class WindowStitcher:
    """
    Handles stitching together the windows of data from the inference loop.

    For example, handles passing in windows to data writers which combine
    them together into a continuous series, and handles storing prognostic
    variables from the end of a window to use as the initial condition for
    the next window.
    """

    def __init__(
        self,
        n_forward_steps: int,
        writer: Union[DataWriter, NullDataWriter],
        is_ensemble: bool = False,
    ):
        self.i_time = 0
        self.n_forward_steps = n_forward_steps
        self.writer = writer
        self.is_ensemble = is_ensemble
        # tensors have shape [n_sample, n_lat, n_lon] with no time axis
        self._initial_condition: Optional[Mapping[str, torch.Tensor]] = None

    def append(
        self,
        data: Dict[str, torch.tensor],
        gen_data: Dict[str, torch.tensor],
        batch_times: xr.DataArray,
    ) -> None:
        """
        Appends a time segment of data to the ensemble batch.

        Args:
            data: The reference data for the current time segment, tensors
                should have shape [n_sample, n_time, n_lat, n_lon]
            gen_data: The generated data for the current time segment, tensors
                should have shape [n_sample, n_time, n_lat, n_lon]
            batch_times: Time coordinates for each sample in the batch.
        """
        tensor_shape = next(data.values().__iter__()).shape
        self.writer.append_batch(
            target=data,
            prediction=gen_data,
            start_timestep=self.i_time,
            start_sample=0,
            batch_times=batch_times,
        )
        self.i_time += tensor_shape[1]
        if self.i_time < self.n_forward_steps:  # only store if needed
            # store the end of the time window as
            # initial condition for the next segment.
            self._initial_condition = {key: value[:, -1] for key, value in data.items()}
            self.ensemble_keys = list(gen_data.keys())
            for key, value in gen_data.items():
                self._initial_condition[key] = value[..., -1, :, :].detach().cpu()  # 3rd last dimension is time

            for key, value in self._initial_condition.items():
                self._initial_condition[key] = value.detach().cpu()

    def apply_initial_condition(
        self,
        data: Mapping[str, torch.Tensor],
        ensemble_member: int = None,
    ):
        """
        Applies the last recorded state of the batch as the initial condition for
        the next segment of the timeseries.

        Args:
            data: The data to apply the initial condition to, tensors should have
                shape [n_sample, n_time, n_lat, n_lon] and the first value along
                the time axis will be replaced with the last value from the
                previous segment.
        """
        if self.i_time > self.n_forward_steps:
            raise ValueError(
                "Cannot apply initial condition after "
                "the last segment has been appended, currently at "
                f"time index {self.i_time} "
                f"with {self.n_forward_steps} max forward steps."
            )
        if ensemble_member is not None:
            assert self.is_ensemble, "Cannot apply initial condition for ensemble member > 0 if not ensemble"
        if self.is_ensemble:
            assert ensemble_member is not None, "Must specify ensemble member to apply initial condition for ensemble"

        if self._initial_condition is not None:
            for key, value in data.items():
                ic = self._initial_condition[key].to(value.device)
                if self.is_ensemble and key in self.ensemble_keys:
                    ic = ic[ensemble_member, ...]
                value[:, 0] = ic


def _inference_internal_loop(
    stepped: SteppedData,
    i_time: int,
    aggregator: InferenceAggregator,
    stitcher: WindowStitcher,
    batch_times: xr.DataArray,
):
    """Do operations that need to be done on each time step of the inference loop.

    This function exists to de-duplicate code between run_inference and
    run_data_inference."""

    # for non-initial windows, we want to record only the new data
    # and discard the initial sample of the window
    if i_time > 0:
        stepped = stepped.remove_initial_condition()
        batch_times = batch_times.isel(time=slice(1, None))
        i_time_aggregator = i_time + 1
    else:
        i_time_aggregator = i_time
    # record raw data for the batch, and store the final state
    # for the next segment
    stitcher.append(stepped.target_data, stepped.gen_data, batch_times)
    # record metrics
    aggregator.record_batch(
        loss=float(stepped.metrics["loss"]),
        target_data=stepped.target_data,
        gen_data=stepped.gen_data,
        target_data_norm=stepped.target_data_norm,
        gen_data_norm=stepped.gen_data_norm,
        i_time_start=i_time_aggregator,
    )


def _to_device(data: Mapping[str, torch.Tensor], device: torch.device) -> Dict[str, Any]:
    return {key: value.to(device) for key, value in data.items()}


def run_inference(
    aggregator: InferenceAggregator,
    stepper: SingleModuleStepper,
    data: GriddedData,
    n_forward_steps: int,
    forward_steps_in_memory: int,
    n_ensemble_members: int,
    eval_device: torch.device | str,
    writer: Optional[Union[DataWriter, NullDataWriter]] = None,
) -> Dict[str, float]:
    if writer is None:
        writer = NullDataWriter()
    stitcher = WindowStitcher(n_forward_steps, writer, is_ensemble=n_ensemble_members > 1)

    not_compute_metrics = isinstance(aggregator, NullAggregator)
    with torch.no_grad():
        stepper.module.eval()
        # We have data batches with long windows, where all data for a
        # given batch does not fit into memory at once, so we window it in time
        # and run the model on each window in turn.
        #
        # We process each time window and keep track of the
        # final state. We then use this as the initial condition
        # for the next time window.
        device = get_device()
        logging.info(f"Running inference on {n_forward_steps} steps, with {n_ensemble_members} ensemble members")
        timers: Dict[str, float] = defaultdict(float)
        current_time = time.time()
        for i, window_batch_data in enumerate(data.loader):
            timers["data_loading"] += time.time() - current_time
            current_time = time.time()
            i_time = i * forward_steps_in_memory
            logging.info(
                f"Inference: starting window spanning {i_time}"
                f" to {i_time + forward_steps_in_memory} steps, "
                f"out of total {n_forward_steps}."
            )
            window_data = _to_device(window_batch_data.data, device)

            target_data = compute_derived_quantities(window_data, data.sigma_coordinates)
            metrics, gen_data, gen_data_norm = defaultdict(list), [], []
            for ens_mem in range(n_ensemble_members):
                print(f"Ensemble member {ens_mem}")
                stitcher.apply_initial_condition(
                    window_data, ensemble_member=ens_mem if n_ensemble_members > 1 else None
                )
                stepped = stepper.run_on_batch(
                    window_data,
                    NullOptimization(),
                    n_forward_steps=forward_steps_in_memory,
                )

                if not_compute_metrics:
                    gen_data.append({key: value.detach() for key, value in stepped.gen_data.items()})
                    gen_data_norm.append({key: value.detach() for key, value in stepped.gen_data_norm.items()})
                else:
                    for k, v in stepped.metrics.items():
                        metrics[k].append(float(v.detach().cpu()))
                    gen_data.append({key: value.detach().cpu() for key, value in stepped.gen_data.items()})
                    gen_data_norm.append({key: value.detach().cpu() for key, value in stepped.gen_data_norm.items()})

            if n_ensemble_members == 1:
                stepped = stepped
            else:
                # Stack the ensemble members into a single tensor (first dimension)
                ensemble_dim = 0
                if not_compute_metrics:
                    metrics = None
                else:
                    metrics = {key: np.mean(value) for key, value in metrics.items()}
                gen_data = {
                    key: torch.stack([value[key] for value in gen_data], dim=ensemble_dim)
                    for key in gen_data[0].keys()
                }
                gen_data_norm = {
                    key: torch.stack([value[key] for value in gen_data_norm], dim=ensemble_dim)
                    for key in gen_data_norm[0].keys()
                }
                stepped = SteppedData(
                    metrics=metrics,
                    target_data=stepped.target_data,
                    gen_data=gen_data,
                    target_data_norm=stepped.target_data_norm,
                    gen_data_norm=gen_data_norm,
                )

            stepped.target_data = target_data
            stepped.gen_data = compute_derived_quantities(stepped.gen_data, data.sigma_coordinates)
            stepped.gen_data = _to_device(stepped.gen_data, device)
            stepped.gen_data_norm = _to_device(stepped.gen_data_norm, device)
            stepped.target_data_norm = _to_device(stepped.target_data_norm, device)
            timers["run_on_batch"] += time.time() - current_time
            current_time = time.time()
            _inference_internal_loop(
                stepped,
                i_time,
                aggregator,
                stitcher,
                window_batch_data.times,
            )
            del stepped
            timers["writer_and_aggregator"] += time.time() - current_time
            current_time = time.time()

        for name, duration in timers.items():
            print(f"{name} duration: {duration:.2f}s")
    return timers


def remove_initial_condition(data: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {key: value[:, 1:] for key, value in data.items()}


def run_dataset_inference(
    aggregator: InferenceAggregator,
    normalizer: StandardNormalizer,
    prediction_data: GriddedData,
    target_data: GriddedData,
    n_forward_steps: int,
    forward_steps_in_memory: int,
    writer: Optional[Union[DataWriter, NullDataWriter]] = None,
) -> Dict[str, float]:
    if writer is None:
        writer = NullDataWriter()
    stitcher = WindowStitcher(n_forward_steps, writer)

    device = get_device()
    # We have data batches with long windows, where all data for a
    # given batch does not fit into memory at once, so we window it in time
    # and run the model on each window in turn.
    #
    # We process each time window and keep track of the
    # final state. We then use this as the initial condition
    # for the next time window.
    timers: Dict[str, float] = defaultdict(float)
    current_time = time.time()
    for i, (pred, target) in enumerate(zip(prediction_data.loader, target_data.loader)):
        timers["data_loading"] += time.time() - current_time
        current_time = time.time()
        i_time = i * forward_steps_in_memory
        logging.info(
            f"Inference: starting window spanning {i_time}"
            f" to {i_time + forward_steps_in_memory} steps,"
            f" out of total {n_forward_steps}."
        )
        pred_window_data = _to_device(pred.data, device)
        target_window_data = _to_device(target.data, device)
        stepped = SteppedData(
            {"loss": torch.tensor(float("nan"))},
            pred_window_data,
            target_window_data,
            normalizer.normalize(pred_window_data),
            normalizer.normalize(target_window_data),
        )
        stepped = compute_stepped_derived_quantities(stepped, target_data.sigma_coordinates)
        timers["run_on_batch"] += time.time() - current_time
        current_time = time.time()
        _inference_internal_loop(
            stepped,
            i_time,
            aggregator,
            stitcher,
            target.times,
        )
        timers["writer_and_aggregator"] += time.time() - current_time
        current_time = time.time()
    for name, duration in timers.items():
        logging.info(f"{name} duration: {duration:.2f}s")
    return timers
