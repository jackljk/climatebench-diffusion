from typing import Any, Dict, List

import torch
import xarray as xr


class StandardNormalizer(torch.nn.Module):
    """
    Responsible for normalizing tensors.
    """

    def __init__(self, means: Dict[str, torch.Tensor], stds: Dict[str, torch.Tensor], names=None):
        super().__init__()
        # if isinstance(means, dict):
        #     for k in means.keys():
        #         if means[k].ndim == 1:
        #             Add singleton dimensions for broadcasting over lat/lon dimensions
                    # means[k] = torch.reshape(means[k], (-1, 1, 1))
                    # stds[k] = torch.reshape(stds[k], (-1, 1, 1))
                # elif means[k].ndim > 1:
                #     raise ValueError(f"Means tensor {k} has more than one dimension!")
        # Make sure that means and stds move to the same device
        self.means = means
        self.stds = stds
        self.names = names

        if torch.is_tensor(means) or isinstance(means, float):
            self._normalize = _normalize
            self._denormalize = _denormalize
        else:
            assert isinstance(means, dict), "Means and stds must be either both tensors, floats, or dictionaries!"
            self._normalize = _normalize_dict
            self._denormalize = _denormalize_dict

    def _apply(self, fn, recurse=True):
        super()._apply(fn)  # , recurse=recurse)
        if isinstance(self.means, dict):
            self.means = {k: fn(v) if torch.is_tensor(v) else v for k, v in self.means.items()}
            self.stds = {k: fn(v) if torch.is_tensor(v) else v for k, v in self.stds.items()}
        else:
            self.means = fn(self.means) if torch.is_tensor(self.means) else self.means
            self.stds = fn(self.stds) if torch.is_tensor(self.stds) else self.stds

    def normalize(self, tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self._normalize(tensors, means=self.means, stds=self.stds)

    def denormalize(self, tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if self.names is not None:
            assert (
                len(set(tensors.keys()) - set(self.names)) == 0
            ), f"Some keys would not be denormalized: {set(tensors.keys()) - set(self.names)}!"
        return self._denormalize(tensors, means=self.means, stds=self.stds)

    def __copy__(self):
        return StandardNormalizer(self.means, self.stds, self.names)

    def clone(self):
        return self.__copy__()


@torch.jit.script
def _normalize_dict(
    tensors: Dict[str, torch.Tensor],
    means: Dict[str, torch.Tensor],
    stds: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    return {k: (t - means[k]) / stds[k] for k, t in tensors.items()}


@torch.jit.script
def _denormalize_dict(
    tensors: Dict[str, torch.Tensor],
    means: Dict[str, torch.Tensor],
    stds: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    return {k: t * stds[k] + means[k] for k, t in tensors.items()}


@torch.jit.script
def _normalize(tensor: torch.Tensor, means: torch.Tensor, stds: torch.Tensor) -> torch.Tensor:
    return (tensor - means) / stds


@torch.jit.script
def _denormalize(tensor: torch.Tensor, means: torch.Tensor, stds: torch.Tensor) -> torch.Tensor:
    return tensor * stds + means


def get_normalizer(
    global_means_path, global_stds_path, names: List[str], sel: Dict[str, Any] = None, is_2d_flattened=False
) -> StandardNormalizer:
    mean_ds = xr.open_dataset(global_means_path)
    std_ds = xr.open_dataset(global_stds_path)
    if sel is not None:
        mean_ds = mean_ds.sel(**sel)
        std_ds = std_ds.sel(**sel)
    if is_2d_flattened:
        means, stds = dict(), dict()
        for name in names:
            if name in mean_ds.keys():
                means[name] = torch.as_tensor(mean_ds[name].values, dtype=torch.float)
                stds[name] = torch.as_tensor(std_ds[name].values, dtype=torch.float)
            else:
                # Retrieve <var_name>_<pressure_level> variables
                var_name, pressure_level = "_".join(name.split("_")[:-1]), name.split("_")[-1]
                assert (
                    pressure_level.isdigit()
                ), f"{name=} is not in the format <var_name>_<pressure_level>! {mean_ds.keys()=}"
                pressure_level = int(pressure_level)
                try:
                    means[name] = torch.as_tensor(
                        mean_ds[var_name].sel(level=pressure_level).values, dtype=torch.float
                    )
                    stds[name] = torch.as_tensor(std_ds[var_name].sel(level=pressure_level).values, dtype=torch.float)
                except KeyError as e:
                    print(mean_ds.coords.values)
                    raise KeyError(
                        f"Variable {name} with var_name {var_name} and level ``{pressure_level}`` not found in the dataset!"
                    ) from e
    else:
        means = {name: torch.as_tensor(mean_ds[name].values, dtype=torch.float) for name in names}
        stds = {name: torch.as_tensor(std_ds[name].values, dtype=torch.float) for name in names}
    return StandardNormalizer(means=means, stds=stds, names=names)


def load_Dict_from_netcdf(path, names):
    ds = xr.open_dataset(path)
    return {name: ds[name].values for name in names}
