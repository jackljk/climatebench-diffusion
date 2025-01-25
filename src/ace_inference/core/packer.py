from typing import Dict, List

import torch
import torch.jit
from tensordict import TensorDict


class DataShapesNotUniform(ValueError):
    """Indicates that a set of tensors do not all have the same shape."""

    pass


class NoPacker:
    def pack(self, tensors: Dict[str, torch.Tensor], axis=0) -> torch.Tensor:
        return tensors

    def unpack(self, tensor: torch.Tensor, axis=0) -> Dict[str, torch.Tensor]:
        return tensor


class Packer:
    """
    Responsible for packing tensors into a single tensor.
    """

    def __init__(self, names: List[str], axis=None):
        self.names = names
        self.axis = axis

    def pack(self, tensors: Dict[str, torch.Tensor], axis=None) -> torch.Tensor:
        """
        Packs tensors into a single tensor, concatenated along a new axis

        Args:
            tensors: Dict from names to tensors.
            axis: index for new concatenation axis.
        """
        axis = axis if axis is not None else self.axis
        return _pack(tensors, self.names, axis=axis)

    def unpack(self, tensor: torch.Tensor, axis=None) -> TensorDict:
        axis = axis if axis is not None else self.axis
        # packed shape is tensor.shape with axis removed
        packed_shape = list(tensor.shape)
        packed_shape.pop(axis)
        return TensorDict(_unpack(tensor, self.names, axis=axis), batch_size=packed_shape)

    def get_state(self):
        """
        Returns state as a serializable data structure.
        """
        return {"names": self.names}

    @classmethod
    def from_state(self, state) -> "Packer":
        """
        Loads state from a serializable data structure.
        """
        return Packer(state["names"])


@torch.jit.script
def _pack(tensors: Dict[str, torch.Tensor], names: List[str], axis: int) -> torch.Tensor:
    return torch.cat([tensors[n].unsqueeze(axis) for n in names], dim=axis)


@torch.jit.script
def _unpack(tensor: torch.Tensor, names: List[str], axis: int) -> Dict[str, torch.Tensor]:
    return {n: tensor.select(axis, index=i) for i, n in enumerate(names)}
