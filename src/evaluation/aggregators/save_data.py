from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import torch
import xarray as xr
from tensordict import TensorDict

from src.evaluation.aggregators._abstract_aggregator import AbstractAggregator
from src.utilities.utils import get_logger, rrearrange, to_tensordict, torch_to_numpy


log = get_logger(__name__)


class SaveToDiskAggregator(AbstractAggregator):
    """
    Aggregator for spectra metrics.
    """

    def __init__(
        self,
        final_dims_of_data: List[str],  # e.g. ["channel", "latitude", "longitude"], or ["latitude", "longitude"]
        var_names: Optional[List[str]] = None,
        coords: Optional[Dict[str, np.ndarray]] = None,  # Xarray coordinates
        concat_dim_name: Optional[str] = None,
        batch_dim_name: Optional[str] = "batch",
        max_ensemble_members: Optional[int] = 5,  # Number of ensemble members to save (if applicable)
        save_to_path: Optional[str] = None,
        save_to_wandb: Optional[bool] = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.var_names = var_names
        self.final_dims_of_data = final_dims_of_data
        self._running_data = None
        self._metadatas = []
        self._data_coords = coords
        self.concat_dim_name = concat_dim_name
        self.batch_dim_name = batch_dim_name
        self.max_ensemble_members = max_ensemble_members
        self.save_to_path = save_to_path
        self.save_to_wandb = save_to_wandb
        self.dims = None
        if coords is not None:
            for k in coords.keys():
                assert k in final_dims_of_data, f"coord {k} must be in final_dims_of_data ({final_dims_of_data=})"

    @torch.inference_mode()
    def _record_batch(
        self,
        target_data: Mapping[str, torch.Tensor],
        gen_data: Mapping[str, torch.Tensor],
        target_data_norm: Mapping[str, torch.Tensor] = None,
        gen_data_norm: Mapping[str, torch.Tensor] = None,
        concat_dim_key: str = None,
        metadata: Mapping[str, Any] = None,
    ):
        batch_dim = 0
        if self._is_ensemble:
            if self.max_ensemble_members is not None:
                gen_data = gen_data[: self.max_ensemble_members, ...]
            # Re-arrange ensemble dim (e, b, h, w) from the front to (b, e, h, w)
            gen_data = rrearrange(gen_data, "e b ... -> b e ...")

        if torch.is_tensor(target_data):  # add dummy key
            data = {"targets": target_data, "preds": gen_data}
        else:
            data = {
                **{f"{k}_targets": v for k, v in target_data.items()},
                **{f"{k}_preds": v for k, v in gen_data.items()},
            }
            
        batch_size = gen_data.shape[:1] if isinstance(gen_data, TensorDict) else gen_data[self.var_names[0]].shape[:1]
                                                        
        data = to_tensordict(data, device="cpu", batch_size=batch_size).to("cpu")
        if concat_dim_key is None:
            if self._running_data is None:
                self._running_data = data
            else:
                # Simply concatenate the data along the batch dimension
                self._running_data = torch.cat([self._running_data, data], dim=batch_dim)
        else:
            # E.g. concat_dim_key = "t1", "t4", "t8" etc.
            if self._running_data is None:
                self._running_data = dict()
            if concat_dim_key not in self._running_data.keys():
                # Initialize the running data with the new data
                self._running_data[concat_dim_key] = data
            else:
                # Concatenate the data into specific dimension for values with the same concat_dim_key
                self._running_data[concat_dim_key] = torch.cat(
                    [self._running_data[concat_dim_key], data], dim=batch_dim
                )
        if metadata is not None and (concat_dim_key is None or concat_dim_key == list(self._running_data.keys())[0]):
            self._metadatas.append(torch_to_numpy(metadata))

    @torch.inference_mode()
    def _get_logs(self, label: str = "", epoch: Optional[int] = None, metadata=None) -> Dict[str, float]:
        """Converts running data to xarray dataset."""
        if self._running_data is None:
            log.warning("No data to log.")
            return {}
        metadata = metadata or {}
        if epoch is not None:
            metadata["epoch"] = epoch  # Add epoch information if provided

        if self._metadatas:
            # Check if self.batch_dim_name is provided
            if self.batch_dim_name in self._metadatas[0]:
                self._data_coords[self.batch_dim_name] = []
                for m in self._metadatas:
                    # Check if datetime was converted to .astype('datetime64[s]').astype('int64') => convert back
                    v = m.pop(self.batch_dim_name)
                    if isinstance(v, np.int64):
                        v = np.datetime64(v, "s")
                        # Conver to a year, month, day, hour format
                        v = v.astype("datetime64[h]")  # .astype('int64')

                    if torch.is_tensor(v):
                        v = v.cpu().item()
                    # handle case where v is a list
                    if isinstance(v, list):
                        self._data_coords[self.batch_dim_name].extend(v)
                    else:
                        self._data_coords[self.batch_dim_name].append(v)

        # Handle case where data is stored with concat dimensions
        # todo: implement gather operation when using DDP (gather to rank 0 only)
        if isinstance(self._running_data, dict):
            # First concatenate along the concat dimension
            concat_dim = self.concat_dim_name or "concat_dim"
            data = torch.stack(list(self._running_data.values()), dim=1)
            final_ds = self._tensordict_to_dataset(data, {concat_dim: list(self._running_data.keys())})
        else:
            # Direct conversion for data without concat dimensions
            final_ds = self._tensordict_to_dataset(self._running_data)

        final_ds.attrs["label"] = label
        # Add metadata if available
        if self._metadatas:
            for key, value in self._metadatas[0].items():
                # TEMPORARY: Dont add 'ssp' metadata
                if key == "ssp":
                    continue

                if isinstance(value, (np.ndarray, dict)):
                    log.info(f"Adding {type(value)} to metadata is not supported. Skipping {key}")
                    continue

                final_ds.attrs[key] = [m[key] for m in self._metadatas if m[key] is not None]
                log.info(f"Added {key} to metadata: {final_ds.attrs[key]}")

        for key, value in metadata.items():
            final_ds.attrs[key] = value

        # Save to file if path is provided
        save_to_path = (
            self.save_to_path + f"{label}-epoch{epoch}-results.nc"
            if self.save_to_path
            else f"{label}-epoch{epoch}-results.nc"
        )
        log.info(f"Saving results to {save_to_path}")
        # predictions/6h-1AR_Attn23_ADM_EMA_256x1-2-3-4d_WMSE_54lr_LC5:200_15wd_fLV_11seed_19h03mOct18_3423514-5214396-hor30-TAG-ENS=5-max_val_samples=1-val_slice=20210329_20210430-possible_initial_times=12-prediction_horizon=30-TAG-epoch199.nc
        final_ds.to_netcdf(save_to_path)
        if self.save_to_wandb:
            import wandb
            wandb.save(save_to_path)

        # Reset running data
        self._running_data = None
        self._metadatas = []

        return {}, {}, {}

    def _tensordict_to_dataset(
        self, tensordict: TensorDict, concat_coord: Optional[Dict[str, Any]] = None
    ) -> xr.Dataset:
        """
        Convert a tensor dictionary to xarray dataset with proper coordinates.

        Args:
            tensordict: Dictionary of tensors or TensorDict
            concat_coord: Optional coordinate to add for concat dimension

        Returns:
            xarray Dataset with proper coordinates
        """
        coords = {}
        dims = [self.batch_dim_name]

        # Set up basic coordinates if provided
        if self._data_coords is not None:
            coords.update(self._data_coords)

        # Add concat coordinate if provided
        if concat_coord is not None:
            assert len(concat_coord) == 1, "Only one concat dimension is supported."
            coords.update(concat_coord)
            dims.append(list(concat_coord.keys())[0])

        data_vars = {}
        # Convert each tensor to data variable
        for name, tensor in tensordict.items():
            dims_here = dims.copy()
            # Move tensor to CPU and convert to numpy
            tensor_np = torch_to_numpy(tensor)

            # Determine dimensions based on tensor shape
            if self._is_ensemble and "preds" in name:
                dims_here.append("ensemble")
            dims_here.extend(self.final_dims_of_data)

            # Ensure dims match tensor shape
            assert len(tensor_np.shape) == len(dims_here), f"{tensor_np.shape=} does not match dims {dims_here}"
            # dims = dims[:len(tensor_np.shape)]
            data_vars[name] = (dims_here, tensor_np)

        # Create dataset
        ds = xr.Dataset(data_vars=data_vars, coords=coords)
        return ds
