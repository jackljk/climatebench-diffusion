from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import torch
from tensordict import TensorDictBase

from src.utilities.utils import ellipsis_torch_dict_boolean_tensor, get_logger, to_tensordict


class AbstractAggregator(ABC):
    def __init__(
        self,
        is_ensemble: bool = False,
        area_weights: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        name: str | None = None,
        verbose: bool = True,
    ):
        self.log_text = get_logger(name=self.__class__.__name__)

        self.mask = mask
        if mask is not None:
            self.log_text.info(f"{name}: Using mask for evaluation of shape {mask.shape}") if verbose else None
            if area_weights is not None:
                area_weights = area_weights[mask]

        if area_weights is not None and verbose:
            self.log_text.info(f"{name}: Using area weights for evaluation of shape {area_weights.shape}")
        self._area_weights = area_weights
        self._is_ensemble = is_ensemble
        self.name = name

    @abstractmethod
    def _record_batch(self, **kwargs) -> None: ...

    def record_batch(self, predictions_mask: Optional[torch.Tensor] = None, **kwargs) -> None:
        assert predictions_mask is None, f"Deprecated predictions_mask {predictions_mask}"
        if self.mask is not None:
            # Apply mask to all tensors
            for key, data in kwargs.items():
                # print(f"{key} Shape before ellipsis_torch_dict_boolean_tensor: {data.shape}")
                if torch.is_tensor(data):
                    kwargs[key] = data[..., self.mask]
                elif isinstance(data, TensorDictBase):
                    kwargs[key] = to_tensordict(
                        {k: ellipsis_torch_dict_boolean_tensor(v, self.mask) for k, v in data.items()},
                        find_batch_size_max=True,
                    )
                else:
                    raise ValueError(f"Unsupported data type {type(data)}")
                # print(f"{key} Shape after ellipsis_torch_dict_boolean_tensor: {kwargs[key].shape}")

        return self._record_batch(**kwargs)

    @torch.inference_mode()
    def get_logs(self, prefix: str, **kwargs) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        prefix = "" if prefix is None else prefix
        if self.name is not None and self.name not in prefix:
            prefix = f"{prefix}/{self.name}".replace("//", "/").rstrip("/").lstrip("/")
        logs_values, logs_media = self._get_logs(**kwargs)
        logs_values = {f"{prefix}/{key}": value for key, value in logs_values.items()}
        logs_media = {f"{prefix}/{key}": value for key, value in logs_media.items()}
        return logs_values, logs_media

    @abstractmethod
    def _get_logs(self, epoch: Optional[int] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]: ...
