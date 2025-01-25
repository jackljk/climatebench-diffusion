# BSD 3-Clause License
#
# Copyright (c) 2022, FourCastNet authors
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# The code was authored by the following people:
#
# Jaideep Pathak - NVIDIA Corporation
# Shashank Subramanian - NERSC, Lawrence Berkeley National Laboratory
# Peter Harrington - NERSC, Lawrence Berkeley National Laboratory
# Sanjeev Raja - NERSC, Lawrence Berkeley National Laboratory
# Ashesh Chattopadhyay - Rice University
# Morteza Mardani - NVIDIA Corporation
# Thorsten Kurth - NVIDIA Corporation
# David Hall - NVIDIA Corporation
# Zongyi Li - California Institute of Technology, NVIDIA Corporation
# Kamyar Azizzadenesheli - Purdue University
# Pedram Hassanzadeh - Rice University
# Karthik Kashinath - NVIDIA Corporation
# Animashree Anandkumar - California Institute of Technology, NVIDIA Corporation

import logging
import os
from collections import namedtuple
from typing import List, Mapping

import netCDF4
import numpy as np
import torch
from fme.core.device import using_gpu
from fme.core.distributed import Distributed
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

# import cv2
from .data_loader_params import DataLoaderParams
from .data_requirements import DataRequirements


def get_data_loader(
    params: DataLoaderParams,
    split: str,
    requirements: DataRequirements,
):
    assert split in ["train", "validation", "test"], f"Invalid split: {split}"
    is_train = split == "train"
    dist = Distributed.get_instance()
    # TODO: move this default to the DataLoaderParams init
    if params.data_type is None:
        params.data_type = "ERA5"
    if params.data_type == "ERA5":
        raise NotImplementedError("ERA5 data loader is not implemented. ")
    elif params.data_type in ["FV3GFS", "E3SMV2"]:
        data_path = os.path.join(params.data_path, split)
        dataset = FV3GFSDataset(params, data_path, requirements=requirements)
        if params.num_data_workers > 0:
            # netCDF4 __getitem__ fails with
            # "RuntimeError: Resource temporarily unavailable"
            # if num_data_workers > 0
            # TODO: move this logic to the DataLoaderParams initialization
            logging.warning(
                f"If data_type=={params.data_type}, must use num_data_workers=0. "
                "Got num_data_workers="
                f"{params.num_data_workers}, but it is being set to 0."
            )
            params.num_data_workers = 0
    else:
        raise NotImplementedError(f"{params.data_type} does not have an implemented data loader")

    sampler = DistributedSampler(dataset, shuffle=is_train) if dist.is_distributed() else None
    batch_size = params.batch_size if is_train else params.batch_size_eval
    dataloader = DataLoader(
        dataset,
        batch_size=dist.local_batch_size(int(batch_size)),
        num_workers=params.num_data_workers,
        shuffle=(sampler is None) and is_train,
        sampler=sampler if is_train else None,
        drop_last=True,
        pin_memory=using_gpu(),
    )

    if is_train:
        return dataloader, dataset, sampler
    else:
        return dataloader, dataset


# Old dataset
def load_series_data(idx: int, n_steps: int, ds: netCDF4.MFDataset, names: List[str]):
    # flip the lat dimension so that it is increasing
    arrays = {n: torch.as_tensor(np.flip(ds.variables[n][idx : idx + n_steps, :, :], axis=-2).copy()) for n in names}
    return arrays


VariableMetadata = namedtuple("VariableMetadata", ["units", "long_name"])


class FV3GFSDataset(Dataset):
    def __init__(self, params: DataLoaderParams, data_path, requirements: DataRequirements):
        print("FV3GFSDataset init")
        self.params = params
        self.in_names = requirements.in_names
        self.out_names = requirements.out_names
        self.names = requirements.names
        self.n_in_channels = len(self.in_names)
        self.n_out_channels = len(self.out_names)
        self.path = data_path
        print("self.path", self.path)
        self.full_path = os.path.join(self.path, "*.nc")
        self.n_steps = requirements.n_timesteps  # one input, one output timestep
        print("self.n_steps", self.n_steps)
        self._get_files_stats()
        if params.n_samples is not None:
            self.n_samples_total = params.n_samples

    def _get_files_stats(self):
        logging.info(f"Opening data at {self.full_path}")
        self.ds = netCDF4.MFDataset(self.full_path)
        self.ds.set_auto_mask(False)
        # minus one since don't have an output for the last step
        self.n_samples_total = len(self.ds.variables["time"][:]) - self.n_steps + 1
        # provided ERA5 dataloader gets the "wrong" x/y convention (x is lat, y is lon)
        # so we follow that convention here for consistency
        if "grid_xt" in self.ds.variables:
            self.img_shape_x = len(self.ds.variables["grid_yt"][:])
            self.img_shape_y = len(self.ds.variables["grid_xt"][:])
        else:
            self.img_shape_x = len(self.ds.variables["lat"][:])
            self.img_shape_y = len(self.ds.variables["lon"][:])
        logging.info(f"Found {self.n_samples_total} samples.")
        logging.info(f"Image shape is {self.img_shape_x} x {self.img_shape_y}.")
        logging.info(f"Following variables are available: {list(self.ds.variables)}.")

    @property
    def metadata(self) -> Mapping[str, VariableMetadata]:
        result = {}
        for name in self.names:
            if hasattr(self.ds.variables[name], "units") and hasattr(self.ds.variables[name], "long_name"):
                result[name] = VariableMetadata(
                    units=self.ds.variables[name].units,
                    long_name=self.ds.variables[name].long_name,
                )
        return result

    def __len__(self):
        return self.n_samples_total

    def __getitem__(self, idx):
        return load_series_data(idx=idx, n_steps=self.n_steps, ds=self.ds, names=self.names)
