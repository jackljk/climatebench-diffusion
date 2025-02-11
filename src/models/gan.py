from typing import Any, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from src.models._base_model import BaseModel
from src.models.unet_simple import UNet


class BaseGAN(BaseModel):
    """
    All GANs should inherit from this class.
    Whenever you add a new GAN class, make sure to give it a
    descriptive brief name in src.utilities.naming.py (clean_name function).
    """

    generator: nn.Module
    discriminator: nn.Module


class GAN(BaseGAN):
    def __init__(
        self,
        dim: int,  # hidden dimension
        with_time_emb: bool = False,
        dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        in_channels = self.num_input_channels + self.num_conditional_channels
        out_channels = self.num_output_channels

        # Change anything below this line
        assert dropout == 0.0, "Dropout is not supported in this model"
        self.generator = UNet(
            num_input_channels=in_channels,
            num_output_channels=out_channels,
            dim=dim,
            with_time_emb=with_time_emb,
            dropout=dropout,
        )
        self.discriminator = nn.Sequential(
            nn.Conv2d(in_channels + out_channels, dim, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dim, dim * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dim * 2, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )
        self.automatic_optimization = False

    def forward(self, inputs, time=None, condition=None, return_time_emb: bool = False, **kwargs):
        # Shapes could be (with batch size=B), for example:
        # inputs: [B, 8, 10, 10],
        # time: [B]
        # condition: [B, 1, 10, 10]

        # Preprocess inputs for shape
        if self.num_conditional_channels > 0:
            x = torch.cat([inputs, condition], dim=1)  # now x has shape [B, 9, 10, 10]
        else:
            x = inputs
            assert condition is None

        # Change anything below this line, making sure to use x as input and time as time embedding
        y = self.generator(x, time=time, **kwargs)  # y may have shape [B, 4, 10, 10]

        if return_time_emb:
            return y, time
        return y

    def get_loss(
        self,
        inputs: Tensor,
        targets: Tensor,
        raw_targets: Tensor = None,
        condition: Tensor = None,
        time: Tensor = None,
        metadata: Any = None,
        predictions_mask: Optional[Tensor] = None,
        return_predictions: bool = False,
        optimizer_idx: int = 0,
        **kwargs,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """Get the loss for the given inputs and targets.

        Args:
            inputs (Tensor): Input data tensor of shape :math:`(B, *, C_{in})`
            targets (Tensor): Target data tensor of shape :math:`(B, *, C_{out})`
            raw_targets (Tensor): Raw target data tensor of shape :math:`(B, *, C_{out})`
            condition (Tensor): Conditional data tensor of shape :math:`(B, *, C_{cond})`
            time (Tensor): Time tensor of shape :math:`(B)`
            metadata (Any): Optional metadata
            predictions_mask (Tensor): Mask for the predictions, before computing the loss. Default: None (no mask)
            return_predictions (bool): Whether to return the predictions or not. Default: False.
                                    Note: this will return all the predictions, not just the masked ones (if any).
            optimizer_idx (int): The optimizer index (0 for generator, 1 for discriminator)
        """
        assert predictions_mask is None, "GAN does not support predictions_mask"
        # Predict
        predictions = self(inputs, condition=condition, time=time, **kwargs)  # generated predictions

        # See https://lightning.ai/docs/pytorch/1.8.4/notebooks/lightning_examples/basic-gan.html
        if optimizer_idx == 0:
            # Train Generator
            fake_pred = self.discriminator(torch.cat([inputs, predictions], dim=1))
            # Generator loss (we want discriminator to think fake images are real)
            loss = F.binary_cross_entropy(fake_pred, torch.ones_like(fake_pred))
        if optimizer_idx == 1:
            # Train Discriminator
            # You may want to make sure to use time as input too (tho it might not be necessary)
            real_pred = self.discriminator(torch.cat([inputs, targets], dim=1))
            real_loss = F.binary_cross_entropy(real_pred, torch.ones_like(real_pred))
            # Fake images
            fake_pred = self.discriminator(torch.cat([inputs, predictions.detach()], dim=1))
            fake_loss = F.binary_cross_entropy(fake_pred, torch.zeros_like(fake_pred))
            # Total loss
            loss = (real_loss + fake_loss) / 2
        if return_predictions:
            return loss, predictions
        return loss
