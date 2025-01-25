import dataclasses
from typing import Optional


@dataclasses.dataclass
class DataLoaderParams:
    """
    Attributes:
        data_path: Path to the data.
        data_type: Type of data to load.
        horizon: Number of steps to predict into the future.
        batch_size: Batch size for training.
        batch_size_eval: Batch size for evaluation/validation.
        num_data_workers: Number of parallel data workers.
        multistep_strategy: Strategy for loading multistep data. Options are:
            - "random": Randomly select a step within the horizon to predict.
            - "sequential": Return all steps within the horizon.
            - None: Return only the last step of the horizon.
        n_samples: Number of samples to load, starting at the beginning of the data.
            If None, load all samples.
    """

    data_path: str
    data_type: str
    horizon: int
    batch_size: int
    batch_size_eval: int
    num_data_workers: int
    multistep_strategy: Optional[str] = None
    n_samples: Optional[int] = None

    def __post_init__(self):
        assert self.horizon > 0, f"horizon ({self.horizon}) must be positive"
        if self.n_samples is not None and self.batch_size > self.n_samples:
            raise ValueError(
                f"batch_size ({self.batch_size}) must be less than or equal to "
                f"n_samples ({self.n_samples}) or no batches would be produced"
            )
        if self.multistep_strategy == "null":
            self.multistep_strategy = None
