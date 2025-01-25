# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from typing import Any

import torch

from src.diffusion._base_diffusion import BaseDiffusion
from src.diffusion.schedulers.scheduling_ddpm import DDPMScheduler


class PDERefiner(BaseDiffusion):
    def __init__(
        self,
        predict_difference: bool = False,
        difference_weight: float = 1.0,
        num_refinement_steps: int = 3,
        min_noise_std: float = 4e-7,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.save_hyperparameters(ignore=["pdeconfig", "model"])
        # For Diffusion models and models in general working on small errors,
        # it is better to evaluate the exponential average of the model weights
        # instead of the current weights. If an appropriate scheduler with
        # cooldown is used, the test results will be not influenced.
        # >>> Implemented by BaseModel! self.ema = ExponentialMovingAverage(self.model, decay=self.hparams.ema_decay)
        # We use the Diffusion implementation here. Alternatively, one could
        # implement the denoising manually.
        betas = [min_noise_std ** (k / num_refinement_steps) for k in reversed(range(num_refinement_steps + 1))]
        self.scheduler = DDPMScheduler(
            num_train_timesteps=num_refinement_steps + 1,
            trained_betas=betas,
            prediction_type="v_prediction",
            clip_sample=False,
        )
        # Multiplies k before passing to frequency embedding.
        self.time_multiplier = 1000 / num_refinement_steps

    def train_step(self, batch, **kwargs):
        x, y, cond = batch
        if self.hparams.predict_difference:
            # Predict difference to next step instead of next step directly.
            y = (y - x[:, -1:]) / self.hparams.difference_weight
        k = torch.randint(0, self.scheduler.config.num_train_timesteps, (x.shape[0],), device=x.device)
        noise_factor = self.scheduler.alphas_cumprod.to(x.device)[k]
        noise_factor = noise_factor.view(-1, *[1 for _ in range(x.ndim - 1)])
        signal_factor = 1 - noise_factor
        noise = torch.randn_like(y)
        y_noised = self.scheduler.add_noise(y, noise, k)
        x_in = torch.cat([x, y_noised], axis=1)
        pred = self.model(x_in, time=k * self.time_multiplier, condition=cond, **kwargs)
        target = (noise_factor**0.5) * noise - (signal_factor**0.5) * y
        loss = self.criterion(pred, target)
        return loss, pred, target

    def predict_next_solution(self, x, cond, **kwargs):
        y_noised = torch.randn(
            size=(x.shape[0], self.num_output_channels, *x.shape[2:]), dtype=x.dtype, device=x.device
        )
        for k in self.scheduler.timesteps:
            time = torch.zeros(size=(x.shape[0],), dtype=x.dtype, device=x.device) + k
            x_in = torch.cat([x, y_noised], axis=1)
            pred = self.model.predict_forward(x_in, time=time * self.time_multiplier, condition=cond, **kwargs)
            y_noised = self.scheduler.step(pred, k, y_noised).prev_sample
        y = y_noised
        if self.hparams.predict_difference:
            y = y * self.hparams.difference_weight + x[:, -1:]
        return y

    def predict_forward(self, inputs, condition=None, metadata: Any = None, **kwargs):
        """Called during evaluation/inference."""
        assert len(kwargs) == 0, f"Unknown kwargs: {kwargs}"
        return self.predict_next_solution(inputs, condition, **kwargs)
