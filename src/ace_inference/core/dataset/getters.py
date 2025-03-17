import os
import warnings
from typing import List, Optional, Sequence, Tuple, Type

import torch.utils.data

from src.ace_inference.core.dataset.config import XarrayDataConfig
from src.ace_inference.core.dataset.xarray import (
    DatasetProperties,
    get_xarray_dataset,
)

from .requirements import DataRequirements


def get_datasets(
    dataset_configs: Sequence[XarrayDataConfig],
    requirements: DataRequirements,
    strict: bool = True,
    dataset_class: Optional[Type] = None,
    **kwargs,
) -> Tuple[List, DatasetProperties]:
    """Get a list of dataset instances and their properties.

    Args:
        dataset_configs: Configurations for each dataset to create.
        requirements: Data requirements for the model.
        strict: Whether to error on property inconsistencies between datasets.
        dataset_class: Optional dataset class to use. If None, uses XarrayDataset.
        **kwargs: Additional keyword arguments to pass to the dataset constructor.

    Returns:
        Tuple of list of dataset instances and their properties.
    """
    datasets = []
    properties: Optional[DatasetProperties] = None
    for config in dataset_configs:
        dataset, new_properties = get_xarray_dataset(config, requirements, dataset_class=dataset_class, **kwargs)
        datasets.append(dataset)
        if properties is None:
            properties = new_properties
        elif not strict:
            try:
                properties.update(new_properties)
            except ValueError as e:
                warnings.warn(f"Metadata for each ensemble member are not the same: {e}")
        else:
            properties.update(new_properties)
    if properties is None:
        raise ValueError("At least one dataset must be provided.")

    return datasets, properties


def get_dataset(
    dataset_configs: Sequence[XarrayDataConfig],
    requirements: DataRequirements,
    strict: bool = True,
    dataset_class: Optional[Type] = None,
    sub_paths: Optional[List[str]] = None,
    **kwargs,
) -> Tuple[torch.utils.data.ConcatDataset, DatasetProperties]:
    """Get a concatenated dataset and its properties.

    Args:
        dataset_configs: Configurations for each dataset to create.
        requirements: Data requirements for the model.
        strict: Whether to error on property inconsistencies between datasets.
        dataset_class: Optional dataset class to use. If None, uses XarrayDataset.
        sub_paths: Optional list of subdirectories for ensemble mode.
        **kwargs: Additional keyword arguments to pass to the dataset constructor.

    Returns:
        Tuple of concatenated dataset and its properties.
    """
    # Handle sub_paths for ensemble mode if provided
    if sub_paths and len(dataset_configs) == 1:
        # Create new configs for each sub_path
        base_config = dataset_configs[0]
        configs = []

        for sub_path in sub_paths:
            # Create a new config with the sub_path
            sub_config = XarrayDataConfig(
                data_path=os.path.join(base_config.data_path, sub_path),
                file_pattern=base_config.file_pattern,
                n_repeats=base_config.n_repeats,
                engine=base_config.engine,
                spatial_dimensions=base_config.spatial_dimensions,
                subset=base_config.subset,
                infer_timestep=base_config.infer_timestep,
                torch_dtype=base_config.torch_dtype,
                fill_nans=base_config.fill_nans,
                renamed_variables=base_config.renamed_variables,
                overwrite=base_config.overwrite,
            )
            configs.append(sub_config)

        # Use the new configs instead of the original
        dataset_configs = configs

    datasets, properties = get_datasets(
        dataset_configs, requirements, strict=strict, dataset_class=dataset_class, **kwargs
    )
    ensemble = torch.utils.data.ConcatDataset(datasets)

    # Add properties attribute to the concatenated dataset for compatibility
    ensemble.properties = properties

    return ensemble, properties
