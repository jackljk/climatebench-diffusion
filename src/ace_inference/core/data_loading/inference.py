import dataclasses
import logging

import numpy as np
import torch
import xarray as xr

from src.ace_inference.core.data_loading._xarray import XarrayDataset
from src.ace_inference.core.data_loading.data_typing import HorizontalCoordinates, SigmaCoordinates
from src.ace_inference.core.data_loading.params import XarrayDataParams
from src.ace_inference.core.data_loading.requirements import DataRequirements
from src.ace_inference.core.data_loading.utils import BatchData
from src.ace_inference.core.distributed import Distributed


@dataclasses.dataclass
class InferenceInitialConditionIndices:
    """
    Configuration of the indices for initial conditions during inference.
    """

    n_initial_conditions: int
    first: int = 0
    interval: int = 1

    def __post_init__(self):
        if self.interval < 0:
            raise ValueError("interval must be positive")

    def as_indices(self) -> np.ndarray:
        stop = self.n_initial_conditions * self.interval + self.first
        return np.arange(self.first, stop, self.interval)


@dataclasses.dataclass
class InferenceDataLoaderParams:
    """
    Configuration for inference data.

    This is like the `DataLoaderParams` class, but with some additional
    constraints. During inference, we have only one batch, so the number of
    samples directly determines the size of that batch.

    Attributes:
        dataset: Parameters to define the dataset.
        start_indices: Slice indicating the set of indices to consider for initial
            conditions of inference series of data. Values following the initial
            condition will still come from the full dataset.
        num_data_workers: Number of parallel workers to use for data loading.
    """

    dataset: XarrayDataParams
    start_indices: InferenceInitialConditionIndices
    num_data_workers: int = 0

    @property
    def n_samples(self) -> int:
        return self.start_indices.n_initial_conditions


class InferenceDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        params: InferenceDataLoaderParams,
        forward_steps_in_memory: int,
        requirements: DataRequirements,
    ):
        dataset = XarrayDataset(params.dataset, requirements=requirements)
        self._dataset = dataset
        self._sigma_coordinates = dataset.sigma_coordinates
        self._metadata = dataset.metadata
        self._area_weights = dataset.area_weights
        self._horizontal_coordinates = dataset.horizontal_coordinates
        self._forward_steps_in_memory = forward_steps_in_memory
        self._total_steps = requirements.n_timesteps - 1
        if self._total_steps % self._forward_steps_in_memory != 0:
            raise ValueError(
                f"Total number of steps ({self._total_steps}) must be divisible by "
                f"forward_steps_in_memory ({self._forward_steps_in_memory})."
            )

        # self._dataset._get_files_stats()
        dataset_size = max(1, self._dataset.total_timesteps)
        # print(
        #     f"Dataset has {dataset_size}. Total steps: {self._total_steps}. Forward steps in memory: {self._forward_steps_in_memory}."
        #     f"n_samples={params.n_samples}. ds._n_initial_conditions={dataset._n_initial_conditions}. ds.total_timesteps={dataset.total_timesteps}. ds.n_steps={dataset.n_steps}."
        # )
        # How many times to "copy" the data?
        if params.dataset.n_repeats is None:
            self._n_repeats = self._total_steps // dataset_size
            self._dataset_size = dataset_size
        else:
            self._n_repeats = params.dataset.n_repeats
            self._dataset_size = None

        if self._n_repeats > 1:
            print(
                f"Repeating target data {self._n_repeats} times. Dataset size: {dataset_size}. Total steps: {self._total_steps}."
            )
        self.n_samples = params.n_samples  # public attribute
        self._start_indices = params.start_indices.as_indices()

    def __getitem__(self, index) -> BatchData:
        dist = Distributed.get_instance()
        # 0 -> 0 1-> 100 2-> 200 3-> 0
        i_start = index * self._forward_steps_in_memory
        if self._dataset_size is not None:
            i_start = i_start % self._dataset_size
        sample_tuples = []
        for i_sample in range(self.n_samples):
            # check if sample is one this local rank should process
            if i_sample % dist.world_size != dist.rank:
                continue
            i_window_start = i_start + self._start_indices[i_sample]
            i_window_end = i_window_start + self._forward_steps_in_memory + 1
            window_time_slice = slice(i_window_start, i_window_end)
            tensors, times = self._dataset.get_sample_by_time_slice(window_time_slice)
            if times.shape[0] != self._forward_steps_in_memory + 1:
                assert self._n_repeats > 1, f"n_repeats={self._n_repeats}"
                # Fill sample with data from the beginning of the dataset
                assert (
                    index + 1
                ) * self._forward_steps_in_memory % self._dataset_size < i_start, (
                    f"Expected {(index + 1) * self._forward_steps_in_memory % self._dataset_size} < {i_start}"
                )
                diff = self._forward_steps_in_memory + 1 - times.shape[0]
                window_time_slice_start = slice(0, diff)
                logging.info(
                    f"Index {index} with i_start={i_start}. window_time_slice={window_time_slice} "
                    f"Filling with {diff} time steps from the beginning of the dataset. +1 i_start is"
                    f"{(index + 1) * self._forward_steps_in_memory % self._dataset_size}."
                )
                tensors_start, times_start = self._dataset.get_sample_by_time_slice(window_time_slice_start)
                for k in tensors.keys():
                    if tensors[k].shape[0] == self._forward_steps_in_memory + 1:
                        # This occurs for static variables which are copied already
                        # print(f"Skipping {k} because it has the right shape. tensors[k].shape={tensors[k].shape}.")
                        continue
                    tensors[k] = torch.cat([tensors[k], tensors_start[k]], dim=0)
                times = xr.concat([times, times_start], dim="time")
            else:
                pass
                # logging.info(
                #     f"Index {index} with i_start={i_start}. window_time_slice={window_time_slice} "
                #     f"forward_steps_in_memory={self._forward_steps_in_memory}, total_steps={self._total_steps},"
                #     f" n_samples={self.n_samples}. start_indices={self._start_indices}. ds_size={self._dataset_size}."
                # )
            sample_tuples.append((tensors, times))
            assert times.shape[0] == self._forward_steps_in_memory + 1, (
                f"Expected {self._forward_steps_in_memory + 1} time steps, "
                f"got {sample_tuples[-1][1].shape[0]}. sample_tuples[-1][1].shape={sample_tuples[-1][1].shape}"
            )
        result = BatchData.from_sample_tuples(sample_tuples)
        assert result.times.shape[1] == self._forward_steps_in_memory + 1
        assert result.times.shape[0] == self.n_samples // dist.world_size
        return result

    def __len__(self) -> int:
        return self._total_steps // self._forward_steps_in_memory

    @property
    def sigma_coordinates(self) -> SigmaCoordinates:
        return self._sigma_coordinates

    @property
    def metadata(self) -> xr.Dataset:
        return self._metadata

    @property
    def area_weights(self) -> xr.DataArray:
        return self._area_weights

    @property
    def horizontal_coordinates(self) -> HorizontalCoordinates:
        return self._horizontal_coordinates
