import dataclasses
from typing import Literal, Optional


# todo: remove for config.py


@dataclasses.dataclass
class XarrayDataParams:
    """
    Attributes:
        data_path: Path to the data.
        n_repeats: Number of times to repeat the dataset (in time).
        engine: Backend for xarray.open_dataset. Currently supported options
            are "netcdf4" (the default) and "h5netcdf". Only valid when using
            XarrayDataset.
        sub_paths: List of sub-paths to use as mask for globbing files (instead of using all files).
    """

    data_path: str
    n_repeats: int = 1
    engine: Optional[Literal["netcdf4", "h5netcdf"]] = None


@dataclasses.dataclass
class DataLoaderParams:
    """
    Attributes:
        dataset: Parameters to define the dataset.
        batch_size: Number of samples per batch.
        num_data_workers: Number of parallel workers to use for data loading.
        data_type: Type of data to load.
        subset: Slice defining a subset of the XarrayDataset to load. For
            data_type="ensemble_xarray" case this will be applied to each ensemble
            member before concatenation.
    """

    dataset: XarrayDataParams
    batch_size: int
    num_data_workers: int
    data_type: Literal["xarray", "ensemble_xarray"]
    n_samples: Optional[int] = None

    def __post_init__(self):

        if self.dataset.n_repeats != 1 and self.data_type == "ensemble_xarray":
            raise ValueError("n_repeats must be 1 when using ensemble_xarray")
