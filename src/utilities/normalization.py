from functools import partial
from typing import Any, Dict, List

import torch
import xarray as xr


class StandardNormalizer(torch.nn.Module):
    """
    Responsible for normalizing tensors.
    """

    def __init__(
        self, means: Dict[str, torch.Tensor], stds: Dict[str, torch.Tensor], names=None, var_to_transform_name=None
    ):
        super().__init__()
        self.means = means
        self.stds = stds
        self.var_to_transform_name = var_to_transform_name

        if torch.is_tensor(means) or isinstance(means, float):
            assert (
                var_to_transform_name is None
            ), f"{var_to_transform_name=} must be None if means and stds are floats!"
            assert names is None, f"{names=} must be None if means and stds are floats!"
            self.names = None
            self._normalize = _normalize
            self._denormalize = _denormalize
        else:
            self.names = names if names is not None else list(means.keys())
            assert isinstance(means, dict), "Means and stds must be either both tensors, floats, or dictionaries!"
            assert all(name in means for name in self.names), "All names must be keys in the means dictionary!"
            assert all(name in stds for name in self.names), "All names must be keys in the stds dictionary!"
            if var_to_transform_name is None or len(var_to_transform_name) == 0:
                self._normalize = _normalize_dict
                self._denormalize = _denormalize_dict
            else:
                assert isinstance(var_to_transform_name, dict), "var_to_transform_name must be a dict!"
                transforms, inverse_transforms = dict(), dict()
                for name in self.names:
                    transforms_name = var_to_transform_name.get(name, "null")
                    transforms[name] = TRANSFORMS[transforms_name]["transform"]
                    inverse_transforms[name] = TRANSFORMS[transforms_name]["inverse"]
                self._normalize = partial(_normalize_dict_with_transform, transforms=transforms)
                self._denormalize = partial(_denormalize_dict_with_transform, inverse_transforms=inverse_transforms)

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
        if self.names is not None:  # todo: remove this check
            assert (
                len(set(tensors.keys()) - set(self.names)) == 0
            ), f"Some keys would not be denormalized: {set(tensors.keys()) - set(self.names)}!"
        return self._denormalize(tensors, means=self.means, stds=self.stds)

    def __copy__(self):
        return StandardNormalizer(self.means, self.stds, self.names, self.var_to_transform_name)

    def clone(self):
        return self.__copy__()


@torch.jit.script
def _normalize_dict(
    tensors: Dict[str, torch.Tensor],
    means: Dict[str, torch.Tensor],
    stds: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    return {k: (t - means[k]) / stds[k] for k, t in tensors.items()}


# @torch.jit.script
def _normalize_dict_with_transform(
    tensors: Dict[str, torch.Tensor],
    means: Dict[str, torch.Tensor],
    stds: Dict[str, torch.Tensor],
    transforms,  # e.g. precip: lambda x: torch.log(x + 1), temperature: lambda x: x
) -> Dict[str, torch.Tensor]:
    return {k: (transforms[k](t) - means[k]) / stds[k] for k, t in tensors.items()}


@torch.jit.script
def _denormalize_dict(
    tensors: Dict[str, torch.Tensor],
    means: Dict[str, torch.Tensor],
    stds: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    return {k: t * stds[k] + means[k] for k, t in tensors.items()}


# @torch.jit.script
def _denormalize_dict_with_transform(
    tensors: Dict[str, torch.Tensor],
    means: Dict[str, torch.Tensor],
    stds: Dict[str, torch.Tensor],
    inverse_transforms,  # e.g. precip: lambda x: torch.exp(x) - 1, temperature: lambda x: x
) -> Dict[str, torch.Tensor]:
    return {k: inverse_transforms[k](t * stds[k] + means[k]) for k, t in tensors.items()}


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


@torch.jit.script
def log1p_transform(x):
    return torch.log(x + 1)


@torch.jit.script
def log1p_transform_inverse(x):
    return torch.exp(x) - 1


@torch.jit.script
def log_transform(x):
    return torch.log(x + 1e-8)


@torch.jit.script
def log_transform_inverse(x):
    return torch.exp(x) - 1e-8


@torch.jit.script
def log_transform_general(x, factor: float, offset: float):
    return torch.log(x * factor + offset)


@torch.jit.script
def log_transform_general_inverse(x, factor: float, offset: float):
    return (torch.exp(x) - offset) / factor


TRANSFORMS = {
    "log1p": {"transform": log1p_transform, "inverse": log1p_transform_inverse},
    "log": {"transform": log_transform, "inverse": log_transform_inverse},
    "log_mm_day_1": {
        "transform": partial(log_transform_general, factor=86400, offset=1),
        "inverse": partial(log_transform_general_inverse, factor=86400, offset=1),
    },
    "log_mm_day_001": {
        "transform": partial(log_transform_general, factor=86400, offset=0.01),
        "inverse": partial(log_transform_general_inverse, factor=86400, offset=0.01),
    },
    "null": {"transform": lambda x: x, "inverse": lambda x: x},
}
TRANSFORMS["log_1"] = TRANSFORMS["log1p"]
TRANSFORMS["log_1e-8"] = TRANSFORMS["log"]


# for n in np.linspace(0, 10, 500):
#     assert log_transform_inverse(log_transform(torch.tensor(n))) == torch.tensor(n)
