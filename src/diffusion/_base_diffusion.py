from __future__ import annotations

import inspect
from abc import abstractmethod
from typing import Any

from src.models._base_model import BaseModel


class BaseDiffusion(BaseModel):
    def __init__(
        self,
        model: BaseModel,
        timesteps: int,
        sigma_data: float = None,  # = 0.5,              # Expected standard deviation of the training data.
        sampling_timesteps: int = None,
        sampling_schedule=None,
        **kwargs,
    ):
        signature = inspect.signature(BaseModel.__init__).parameters
        base_kwargs = {k: model.hparams.get(k) for k in signature if k in model.hparams}
        base_kwargs.update(kwargs)  # override base_kwargs with kwargs
        self._sigma_data = sigma_data
        self._USE_SIGMA_DATA = False
        super().__init__(**base_kwargs)
        if model is None:
            raise ValueError(
                "Arg ``model`` is missing..." " Please provide a backbone model for the diffusion model (e.g. a Unet)"
            )
        self.save_hyperparameters(ignore=["model"])
        # self.sampling_timesteps = default(sampling_timesteps, timesteps)
        self.model = model

        self.spatial_shape_in = model.spatial_shape_in
        self.spatial_shape_out = model.spatial_shape_out
        self.num_input_channels = model.num_input_channels
        self.num_output_channels = model.num_output_channels
        self.num_conditional_channels = model.num_conditional_channels
        self.num_timesteps = int(timesteps)

        # if hasattr(model, 'example_input_array'):
        #     self.example_input_array = model.example_input_array
        self.model.criterion = None

    @property
    def short_description(self) -> str:
        name = super().short_description
        name += f" (timesteps={self.num_timesteps})"
        return name

    @property
    def sigma_data(self):
        return self._sigma_data

    @sigma_data.setter
    def sigma_data(self, value):
        self._sigma_data = value
        if hasattr(self, "guidance_model") and self.guidance_model is not None:
            self.guidance_model.model.sigma_data = value
        for k, v in self.criterion.items():  # Set the sigma_data for the loss function
            if hasattr(v, "sigma_data"):
                v.sigma_data = value

    def sample(self, condition=None, num_samples=1, **kwargs):
        # sample from the model
        raise NotImplementedError()

    def predict_forward(self, *inputs, condition=None, metadata: Any = None, **kwargs):
        assert len(inputs) == 1, "Only one input tensor is allowed for the forward pass"
        inputs = inputs[0]
        if inputs is not None and condition is not None:
            raise ValueError("Only one of the inputs or condition should be provided. Need to refactor the code.")
        elif condition is not None:
            raise NotImplementedError("Condition is not implemented yet.")
        else:  # if inputs is not None:
            inital_condition = inputs

        _ = kwargs.pop("lookback", None)  # remove the lookback argument
        return self.sample(inital_condition, **kwargs)

    @abstractmethod
    def p_losses(self, *args, **kwargs):
        """Compute the loss for the given targets and condition.

        Args:
            targets (Tensor): Target data tensor of shape :math:`(B, C_{out}, *)`
            condition (Tensor): Condition data tensor of shape :math:`(B, C_{in}, *)`
            t (Tensor): Timestep of shape :math:`(B,)`
        """
        raise NotImplementedError(f"Method ``p_losses`` is not implemented for {self.__class__.__name__}!")

    def forward(self, *args, **kwargs):
        return self.p_losses(*args, **kwargs)

    def get_loss(self, *args, **kwargs):
        raise NotImplementedError(f"Plese implement the ``get_loss`` method for {self.__class__.__name__}!")
