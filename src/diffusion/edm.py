from abc import abstractmethod
from typing import Dict

import numpy as np
import torch
from einops import repeat

# from src.utilities.torch_utils import persistence
from src.diffusion._base_diffusion import BaseDiffusion
from src.losses.losses import AbstractWeightedLoss, crps_ensemble
from src.utilities.random_control import StackedRandomGenerator
from src.utilities.utils import get_logger, rrearrange


log = get_logger(__name__)


# ----------------------------------------------------------------------------
# Preconditioning corresponding to the variance preserving (VP) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".


# @persistence.persistent_class
class VPPrecond(BaseDiffusion):
    def __init__(
        self,
        use_fp16=False,  # Execute the underlying model at FP16 precision?
        beta_d=19.9,  # Extent of the noise level schedule.
        beta_min=0.1,  # Initial slope of the noise level schedule.
        M=1000,  # Original number of timesteps in the DDPM formulation.
        epsilon_t=1e-5,  # Minimum t-value used during training.
        **kwargs,  # Keyword arguments for the underlying model.
    ):
        super().__init__(**kwargs)
        self.use_fp16 = use_fp16
        self.beta_d = beta_d
        self.beta_min = beta_min
        self.M = M
        self.epsilon_t = epsilon_t
        self.sigma_min = float(self.sigma(epsilon_t))
        self.sigma_max = float(self.sigma(1))
        self.criterion = VPLoss(beta_d=beta_d, beta_min=beta_min, epsilon_t=epsilon_t)

    def forward(self, x, sigma, class_labels=None, force_fp32=False, **model_kwargs):
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        class_labels = (
            None
            if self.label_dim == 0
            else (
                torch.zeros([1, self.label_dim], device=x.device)
                if class_labels is None
                else class_labels.to(torch.float32).reshape(-1, self.label_dim)
            )
        )
        dtype = torch.float16 if (self.use_fp16 and not force_fp32 and x.device.type == "cuda") else torch.float32

        c_skip = 1
        c_out = -sigma
        c_in = 1 / (sigma**2 + 1).sqrt()
        c_noise = (self.M - 1) * self.sigma_inv(sigma)

        F_x = self.model((c_in * x).to(dtype), c_noise.flatten(), class_labels=class_labels, **model_kwargs)
        assert F_x.dtype == dtype
        D_x = c_skip * x + c_out * F_x.to(torch.float32)
        return D_x

    def get_loss(self, targets, labels=None, augment_pipe=None):
        return self.criterion(self, targets, labels, augment_pipe)

    def sigma(self, t):
        return self.loss.sigma(t)

    def sigma_inv(self, sigma):
        sigma = torch.as_tensor(sigma)
        return ((self.beta_min**2 + 2 * self.beta_d * (1 + sigma**2).log()).sqrt() - self.beta_min) / self.beta_d

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)


# ----------------------------------------------------------------------------
# Preconditioning corresponding to the variance exploding (VE) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".


# @persistence.persistent_class
class VEPrecond(BaseDiffusion):
    def __init__(
        self,
        use_fp16=False,  # Execute the underlying model at FP16 precision?
        sigma_min=0.02,  # Minimum supported noise level.
        sigma_max=100,  # Maximum supported noise level.
        **kwargs,  # Keyword arguments for the underlying model.
    ):
        super().__init__(**kwargs)
        self.use_fp16 = use_fp16
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.criterion = VELoss(sigma_min=sigma_min, sigma_max=sigma_max)

    def forward(self, x, sigma, class_labels=None, force_fp32=False, **model_kwargs):
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        class_labels = (
            None
            if self.label_dim == 0
            else (
                torch.zeros([1, self.label_dim], device=x.device)
                if class_labels is None
                else class_labels.to(torch.float32).reshape(-1, self.label_dim)
            )
        )
        dtype = torch.float16 if (self.use_fp16 and not force_fp32 and x.device.type == "cuda") else torch.float32

        c_skip = 1
        c_out = sigma
        c_in = 1
        c_noise = (0.5 * sigma).log()

        F_x = self.model((c_in * x).to(dtype), c_noise.flatten(), class_labels=class_labels, **model_kwargs)
        assert F_x.dtype == dtype
        D_x = c_skip * x + c_out * F_x.to(torch.float32)
        return D_x

    def get_loss(self, targets, labels=None, augment_pipe=None):
        return self.criterion(self, targets, labels, augment_pipe)

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)


# ----------------------------------------------------------------------------
# Preconditioning corresponding to improved DDPM (iDDPM) formulation from
# the paper "Improved Denoising Diffusion Probabilistic Models".


class iDDPMPrecond(torch.nn.Module):
    def __init__(
        self,
        img_resolution,  # Image resolution.
        img_channels,  # Number of color channels.
        label_dim=0,  # Number of class labels, 0 = unconditional.
        use_fp16=False,  # Execute the underlying model at FP16 precision?
        C_1=0.001,  # Timestep adjustment at low noise levels.
        C_2=0.008,  # Timestep adjustment at high noise levels.
        M=1000,  # Original number of timesteps in the DDPM formulation.
        model_type="DhariwalUNet",  # Class name of the underlying model.
        **model_kwargs,  # Keyword arguments for the underlying model.
    ):
        super().__init__()
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.label_dim = label_dim
        self.use_fp16 = use_fp16
        self.C_1 = C_1
        self.C_2 = C_2
        self.M = M
        self.model = globals()[model_type](
            img_resolution=img_resolution,
            in_channels=img_channels,
            out_channels=img_channels * 2,
            label_dim=label_dim,
            **model_kwargs,
        )

        u = torch.zeros(M + 1)
        for j in range(M, 0, -1):  # M, ..., 1
            u[j - 1] = ((u[j] ** 2 + 1) / (self.alpha_bar(j - 1) / self.alpha_bar(j)).clip(min=C_1) - 1).sqrt()
        self.register_buffer("u", u)
        self.sigma_min = float(u[M - 1])
        self.sigma_max = float(u[0])

    def forward(self, x, sigma, class_labels=None, force_fp32=False, **model_kwargs):
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        class_labels = (
            None
            if self.label_dim == 0
            else (
                torch.zeros([1, self.label_dim], device=x.device)
                if class_labels is None
                else class_labels.to(torch.float32).reshape(-1, self.label_dim)
            )
        )
        dtype = torch.float16 if (self.use_fp16 and not force_fp32 and x.device.type == "cuda") else torch.float32

        c_skip = 1
        c_out = -sigma
        c_in = 1 / (sigma**2 + 1).sqrt()
        c_noise = self.M - 1 - self.round_sigma(sigma, return_index=True).to(torch.float32)

        F_x = self.model((c_in * x).to(dtype), c_noise.flatten(), class_labels=class_labels, **model_kwargs)
        assert F_x.dtype == dtype
        D_x = c_skip * x + c_out * F_x[:, : self.img_channels].to(torch.float32)
        return D_x

    def alpha_bar(self, j):
        j = torch.as_tensor(j)
        return (0.5 * np.pi * j / self.M / (self.C_2 + 1)).sin() ** 2

    def round_sigma(self, sigma, return_index=False):
        sigma = torch.as_tensor(sigma)
        index = torch.cdist(
            sigma.to(self.u.device).to(torch.float32).reshape(1, -1, 1), self.u.reshape(1, -1, 1)
        ).argmin(2)
        result = index if return_index else self.u[index.flatten()].to(sigma.dtype)
        return result.reshape(sigma.shape).to(sigma.device)


# ----------------------------------------------------------------------------
# Improved preconditioning proposed in the paper "Elucidating the Design
# Space of Diffusion-Based Generative Models" (EDM).


# @persistence.persistent_class
class EDMPrecond(BaseDiffusion):
    def __init__(
        self,
        use_fp16=False,  # Execute the underlying model at FP16 precision?
        sigma_min=0,  # Minimum supported noise level.
        sigma_max=None,  # Maximum supported noise level.
        sigma_max_inf=80,  # Maximum supported noise level.
        sigma_min_train=None,
        P_mean=-1.2,  # Mean of the noise level distribution.
        P_std=1.2,  # Standard deviation of the noise level distribution.
        noise_distribution: str = "lognormal",  # Distribution of the noise level.
        use_noise_logvar: bool = False,
        when_3d_concat_condition_to: str = None,  # When using 3D model: Concat to 'time' or 'channel' dimension?
        force_unconditional=False,  # Ignore conditioning information?
        # Sampling parameters.
        num_steps=18,  # Number of steps in the sampling loop.
        rho=7,  # Exponent of the time step discretization.
        S_churn=0,  # Maximum noise increase per step.
        S_min=0,  # Minimum noise level for increased noise.
        S_max=float("inf"),  # Maximum noise level for increased noise.
        S_noise=1,  # Noise level for increased noise.
        heun: bool = True,  # Use Heun's method for the sampling loop.
        compute_loss_per_sigma: bool = False,  # Compute loss for each sigma in the range.
        dtype="double",  # double or float
        **kwargs,  # Keyword arguments for the underlying model.
    ):
        kwargs["timesteps"] = num_steps
        super().__init__(**kwargs)
        self._USE_SIGMA_DATA = True
        self.use_fp16 = use_fp16
        self.sigma_min_train = sigma_min_train or sigma_min
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max or float("inf")
        self.sigma_max_inf = sigma_max_inf or self.sigma_max
        assert self.sigma_min < self.sigma_max_inf <= self.sigma_max
        self.heun = heun
        self.label_dim = 0
        self.log_text.info(
            f"EDM: {sigma_min=}, {self.sigma_max_inf=}, {num_steps=}, {rho=}, {S_churn=}, {S_min=}, {S_max=}"
        )

    def _get_loss_callable_from_name_or_config(self, loss_function: str, **kwargs):
        """Return the loss function used for training.
        Function will be called when needed by the BaseModel class.
        Better to do it here in case self.* parameters are changed."""
        loss_kwargs = dict(
            sigma_data=self.sigma_data,
            use_logvar=self.hparams.use_noise_logvar,
            **kwargs,
        )
        if self.hparams.noise_distribution == "lognormal":
            loss_kwargs.update({"P_mean": self.hparams.P_mean, "P_std": self.hparams.P_std})
        elif self.hparams.noise_distribution == "uniform":
            loss_kwargs.update({"P_mean": None, "P_std": None})
        else:
            raise ValueError(f"Unknown noise distribution: {self.hparams.noise_distribution}")

        log.info(f"Using EDM loss function: {loss_function}")
        if loss_function in ["mse", "l1"]:
            loss_kwargs.pop("reduction", None)
            return EDMLoss(**loss_kwargs) if loss_function == "mse" else EDMLossMAE(**loss_kwargs)
        elif loss_function == "wmse":
            return WeightedEDMLoss(**loss_kwargs, loss_type="L2")
        elif loss_function == "wmae":
            return WeightedEDMLoss(**loss_kwargs, loss_type="L1")
        elif loss_function == "wcrps":
            return WeightedEDMLossCRPS(**loss_kwargs)
        else:
            raise ValueError(f"Unknown loss type: {loss_function}")

    def forward(self, x, sigma, force_fp32=False, **model_kwargs):
        if self.hparams.force_unconditional:
            if isinstance(self.hparams.force_unconditional, float):
                # Implement unconditional sampling by setting the condition to 0 with probability force_unconditional
                if self.training:
                    # Set batch elems with p < force_unconditional to 0
                    mask = torch.rand(x.shape[0], device=x.device) < self.hparams.force_unconditional
                    x[mask] = 0
                else:
                    pass  # conditional sampling.
            else:
                _ = model_kwargs.pop("condition", None)
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32)  # .reshape(-1, 1, 1, 1)
        dtype = torch.float16 if (self.use_fp16 and not force_fp32 and x.device.type == "cuda") else torch.float32

        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2).sqrt()
        c_in = 1 / (self.sigma_data**2 + sigma**2).sqrt()
        c_noise = sigma.flatten().log() / 4
        x_in = (c_in * x).to(dtype)
        if self.model.is_3d:
            # Concatenate initial condition prompt with the noisy input/sequence.
            assert x_in.ndim == 5, f"Expected 5D input for 3D model, got {x_in.ndim}D"  # (B, C, T, H, W)
            condition = model_kwargs.pop("condition", None)  # (B, C, T_cond, H, W), x_in: (B, C, T_gen, H, W)
            if condition is not None:
                # Ensure to concat condition with x_in (after applying c_in!), and put it to the front of the sequence.
                if self.hparams.when_3d_concat_condition_to == "time":
                    x_in = torch.cat([condition, x_in], dim=2)  # (B, C, T_cond+T_gen, H, W)
                elif self.hparams.when_3d_concat_condition_to == "channel":
                    condition = rrearrange(condition, "b c tcond ... -> b (c tcond) ...")
                    condition = repeat(condition, "b ctcond ... -> b ctcond tgen ...", tgen=x_in.shape[2])
                    model_kwargs["condition"] = condition  # will be concatenated to x_in on the channel dimension
                else:
                    raise ValueError(f"Invalid {self.hparams.when_3d_concat_condition_to=}")

        F_x = self.model(x_in, c_noise, **model_kwargs)
        if self.model.is_3d and self.hparams.when_3d_concat_condition_to == "time":
            # Remove condition from the model output.
            F_x = F_x[:, :, -x.shape[2] :, ...]  # (B, C, T_gen, H, W)
        D_x = c_skip * x + c_out * F_x.to(torch.float32)
        return D_x

    def get_loss(self, inputs, targets, return_predictions=False, **kwargs):
        # Shouldn't be needed anymore after using predictions_post_process inside the losses:
        # if len(targets.shape) == 5:  # (B, T, C, H, W)
        #     targets = targets.squeeze(1)  # (B, C, H, W)
        if self.hparams.noise_distribution == "uniform":
            # Sample noise levels by uniformly steps from 0 to 1 (sigma_min to sigma_max).
            steps = torch.rand(inputs.shape[0], device=inputs.device)
            sigmas = self.edm_discretization(steps=steps, sigma_min=self.sigma_min_train, sigma_max=self.sigma_max)
            kwargs["sigma"] = sigmas

        loss = self.criterion["preds"](self, targets=targets, condition=inputs, **kwargs)
        # condition will be fed back to .forward() above as part of model_kwargs
        if self.hparams.compute_loss_per_sigma:
            loss = {"loss": loss} if torch.is_tensor(loss) else loss
            _ = kwargs.pop("sigma", None)
            loss.update(self.get_loss_vs_sigmas(inputs, targets, **kwargs))
        if return_predictions:
            return loss, None
        return loss

    def get_loss_vs_sigmas(self, inputs, targets, **kwargs) -> Dict[str, float]:
        sigmas = self.edm_discretization(steps=200)
        losses = dict()  # defaultdict(list)
        for sigma in sigmas:
            loss_sigma = self.criterion["preds"](self, targets=targets, condition=inputs, sigma=sigma, **kwargs)
            loss_sigma = {"loss": loss_sigma} if torch.is_tensor(loss_sigma) else loss_sigma
            if "raw_loss" not in loss_sigma:
                loss_sigma["raw_loss"] = loss_sigma["loss"]
            for k, v in loss_sigma.items():
                # losses[f"{k}_per_noise_level"].append(float(v))
                losses[f"{k}_per_noise_level/sigma{sigma:.3f}"] = float(v)

        # return_dict = {"x_axes": {"sigma": list(sigmas.cpu())}}
        losses["x_axes"] = {"sigma": list(sigmas.cpu())}
        return losses

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)

    def edm_discretization(self, steps, sigma_min: float = None, sigma_max: float = None, rho: float = None):
        sigma_min = sigma_min or self.sigma_min  # max(sigma_min, self.sigma_min)
        sigma_max = sigma_max or self.sigma_max_inf  # min(sigma_max, self.sigma_max)
        rho = rho or self.hparams.rho
        if isinstance(steps, int):
            step_indices = torch.arange(steps, dtype=self.dtype, device=self.device)
            steps = step_indices / (steps - 1)
        return (sigma_max ** (1 / rho) + steps * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho

    def edm_sampler(
        self,
        noise,
        randn_like=torch.randn_like,
        # sigma_min=0.002, sigma_max=80,
        **kwargs,
    ):
        dtype = torch.float64 if self.hparams.dtype == "double" else torch.float32
        dtype = torch.float32

        def denoise(x, t):
            denoised = self(x, t, **kwargs).to(dtype)
            if self.hparams.guidance == 1:
                return denoised
            elif self.hparams.guidance_interval is not None and (
                t < self.hparams.guidance_interval[0] or t > self.hparams.guidance_interval[1]
            ):  # todo: ensure that guidance interval is exactly at step boundaries to satisfy smoothness requiremen
                return denoised  # No guidance outside the interval.
            # Guided denoiser.
            kwargs_g = kwargs
            if self.guidance_model.model.hparams.force_unconditional:
                kwargs_g = {k: v for k, v in kwargs.items() if k != "dynamical_condition"}
            ref_Dx = self.guidance_model(x, t, **kwargs_g).to(dtype)
            denoised = ref_Dx.lerp(denoised, self.hparams.guidance)
            # = ref_Dx + guidance * (denoised - ref_Dx) = guidance * denoised + (1 - guidance) * ref_Dx
            return denoised

        # Adjust noise levels based on what's supported by the network.
        S_churn = self.hparams.S_churn
        S_min = self.hparams.S_min
        S_max = self.hparams.S_max
        S_noise = self.hparams.S_noise
        num_steps = self.hparams.num_steps
        # Time step discretization.
        t_steps = self.edm_discretization(num_steps)
        t_steps = torch.cat([self.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])
        # t_N = 0, but never actually given to network.
        # if self.hparams.dtype == "double":
        # self.model.double()
        # Main sampling loop
        x_next = noise.to(dtype) * t_steps[0]
        # kwargs = {k: v.to(dtype) if torch.is_tensor(v) else v for k, v in kwargs.items()}
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):  # 0, ..., N-1
            x_cur = x_next

            # Increase noise temporarily.
            if S_churn > 0 and S_min <= t_cur <= S_max:
                gamma = min(S_churn / num_steps, np.sqrt(2) - 1)
                t_hat = t_cur + gamma * t_cur  # = (1 + gamma) * t_cur
                x_hat = x_cur + (t_hat**2 - t_cur**2).sqrt() * S_noise * randn_like(x_cur)
            else:
                t_hat = t_cur
                x_hat = x_cur

            # Euler step.
            denoised = denoise(x_hat, t_hat).to(dtype)
            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur
            #      = x_hat + (t_next - t_hat) * (x_hat - denoised) / t_hat.
            # When last step, i.e.: t_next = 0, this becomes: x_next = x_hat - 1 * (x_hat - denoised) = denoised.

            # Apply 2nd order correction.
            if self.heun and i < num_steps - 1:
                denoised = denoise(x_next, t_next).to(dtype)
                d_prime = (x_next - denoised) / t_next
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

        return x_next.to(self.dtype)

    @torch.inference_mode()
    def sample(self, condition, batch_seeds=None, **kwargs):
        batch_size = condition.shape[0]
        batch_seeds = batch_seeds or torch.randint(0, 2**32, (batch_size,), device=condition.device)
        rnd = StackedRandomGenerator(self.device, batch_seeds)  # todo: check how much this makes a difference
        if self.model.is_3d:
            # 3D model, so we need to add the temporal dimension explicitly.
            nt_gen = self.num_temporal_channels
            if self.hparams.when_3d_concat_condition_to != "channel":
                nt_gen -= condition.shape[2]  # Remove the condition time dimensions from latents.
            init_latents_shape = (batch_size, self.num_input_channels, nt_gen, *self.spatial_shape_out)
        else:
            init_latents_shape = (batch_size, self.num_input_channels, *self.spatial_shape_out)
        latents = rnd.randn(
            init_latents_shape, dtype=condition.dtype, layout=condition.layout, device=condition.device
        )
        return self.edm_sampler(latents, condition=condition, **kwargs)


# ----------------------------------------------------------------------------

# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Loss functions used in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""


# ----------------------------------------------------------------------------
# Loss function corresponding to the variance preserving (VP) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".


# @persistence.persistent_class
class VPLoss:
    def __init__(self, beta_d=19.9, beta_min=0.1, epsilon_t=1e-5):
        self.beta_d = beta_d
        self.beta_min = beta_min
        self.epsilon_t = epsilon_t

    def __call__(self, net, targets, labels, augment_pipe=None):
        rnd_uniform = torch.rand([targets.shape[0], 1, 1, 1], device=targets.device)
        sigma = self.sigma(1 + rnd_uniform * (self.epsilon_t - 1))
        weight = 1 / sigma**2
        y, augment_labels = augment_pipe(targets) if augment_pipe is not None else (targets, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return {"loss": loss}

    def sigma(self, t):
        t = torch.as_tensor(t)
        return ((0.5 * self.beta_d * (t**2) + self.beta_min * t).exp() - 1).sqrt()


# ----------------------------------------------------------------------------
# Loss function corresponding to the variance exploding (VE) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".


# @persistence.persistent_class
class VELoss:
    def __init__(self, sigma_min=0.02, sigma_max=100):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def __call__(self, net, targets, labels, augment_pipe=None):
        rnd_uniform = torch.rand([targets.shape[0], 1, 1, 1], device=targets.device)
        sigma = self.sigma_min * ((self.sigma_max / self.sigma_min) ** rnd_uniform)
        weight = 1 / sigma**2
        y, augment_labels = augment_pipe(targets) if augment_pipe is not None else (targets, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return {"loss": loss}


# ----------------------------------------------------------------------------
# Normalize given tensor to unit magnitude with respect to the given
# dimensions. Default = all dimensions except the first.


def normalize(x, dim=None, eps=1e-4):
    if dim is None:
        dim = list(range(1, x.ndim))
    norm = torch.linalg.vector_norm(x, dim=dim, keepdim=True, dtype=torch.float32)
    norm = torch.add(eps, norm, alpha=np.sqrt(norm.numel() / x.numel()))
    return x / norm.to(x.dtype)


class MPFourier(torch.nn.Module):
    def __init__(self, num_channels, bandwidth=1):
        super().__init__()
        self.register_buffer("freqs", 2 * np.pi * torch.randn(num_channels) * bandwidth)
        self.register_buffer("phases", 2 * np.pi * torch.rand(num_channels))

    def forward(self, x):
        y = x.to(torch.float32)
        y = y.ger(self.freqs.to(torch.float32))
        y = y + self.phases.to(torch.float32)
        y = y.cos() * np.sqrt(2)
        return y.to(x.dtype)


class MPConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel, zero_init=False):
        super().__init__()
        self.out_channels = out_channels
        if zero_init:
            self.weight = torch.nn.Parameter(torch.zeros(out_channels, in_channels, *kernel))
        else:
            self.weight = torch.nn.Parameter(torch.randn(out_channels, in_channels, *kernel))

    def forward(self, x, gain=1):
        w = self.weight.to(torch.float32)
        if self.training:
            with torch.no_grad():
                self.weight.copy_(normalize(w))  # forced weight normalization
        w = normalize(w)  # traditional weight normalization
        w = w * (gain / np.sqrt(w[0].numel()))  # magnitude-preserving scaling
        w = w.to(x.dtype)
        if w.ndim == 2:
            return x @ w.t()
        assert w.ndim == 4
        return torch.nn.functional.conv2d(x, w, padding=(w.shape[-1] // 2,))


# ----------------------------------------------------------------------------
# Improved loss function proposed in the paper "Elucidating the Design Space
# of Diffusion-Based Generative Models" (EDM).


# @persistence.persistent_class
class EDMLossAbstract:  # For some reason this cannot inherit (torch.nn.Module) when using wmse/wmae loss - why?
    def __init__(self, P_mean, P_std, sigma_data, use_logvar: bool = False):
        super().__init__()
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.use_logvar = use_logvar
        if use_logvar:
            logvar_channels = 128  # Intermediate dimensionality for uncertainty estimation.
            log.info(f"Using log-variance with {logvar_channels} intermediate channels for noise weighting.")
            self.logvar_fourier = MPFourier(logvar_channels)
            self.logvar_linear = MPConv(logvar_channels, 1, kernel=[])

    @abstractmethod
    def loss(self, preds, targets, sigma_weights, **kwargs):
        pass

    def __call__(self, net, targets, predictions_post_process=None, targets_pre_process=None, sigma=None, **kwargs):
        y = targets_pre_process(targets) if targets_pre_process is not None else targets
        n_dims1 = (1,) * (y.ndim - 1)
        if sigma is None:
            # Sample noise level from the prior distribution. Only specify sigma for analysis of loss vs. sigma.
            rnd_normal = torch.randn([targets.shape[0], *n_dims1], device=targets.device)
            sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        else:
            sigma = sigma.reshape(-1, *n_dims1)

        weight = (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2

        try:
            n = torch.randn_like(y) * sigma
        except RuntimeError as e:
            raise RuntimeError(f"Shape mismatch: y={y.shape}, sigma={sigma.shape}") from e

        D_yn = net(y + n, sigma, **kwargs)
        if predictions_post_process is not None:
            D_yn = predictions_post_process(D_yn)
            diff_shape = len(D_yn.shape) - len(weight.shape)
            if diff_shape != 0:
                assert diff_shape == 1, f"Shape mismatch: {D_yn.shape=} and {weight.shape=}"
                weight = weight.unsqueeze(1)  # add missing dimension (e.g. time)

        loss_kwargs = {}
        if self.use_logvar:
            loss_kwargs["batch_logvars"] = self.logvar_linear(self.logvar_fourier(sigma.flatten().log() / 4))
        loss = self.loss(D_yn, targets, weight, **loss_kwargs)
        return {"loss": loss} if torch.is_tensor(loss) else loss


class EDMLoss(EDMLossAbstract):
    def loss(self, preds, targets, sigma_weights, **kwargs):
        assert len(kwargs) == 0, f"Unknown kwargs: {kwargs}. Consider using a weighted loss like 'wmse' instead?"
        # loss y, , n, and D_yn have the same shape (B, C, H, W). weight has shape (B, 1, 1, 1)
        return (sigma_weights * ((preds - targets) ** 2)).mean()


class EDMLossMAE(EDMLossAbstract):
    def loss(self, preds, targets, sigma_weights, **kwargs):
        assert len(kwargs) == 0, f"Unknown kwargs: {kwargs}. Consider using a weighted loss like 'wmse' instead?"
        return (sigma_weights * ((preds - targets).abs())).mean()


class WeightedEDMLossAbstract(AbstractWeightedLoss, EDMLossAbstract):
    def __init__(self, P_mean, P_std, sigma_data, use_logvar: bool = False, **kwargs):
        AbstractWeightedLoss.__init__(self, use_batch_logvars=use_logvar, **kwargs)
        EDMLossAbstract.__init__(self, P_mean, P_std, sigma_data, use_logvar=use_logvar)

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, weights):
        if weights is not None:
            # Need to add batch dimension to weights since we will multiply the lambda(\sigma) weights with it
            if weights.ndim == 3:
                weights = weights.unsqueeze(0)  # Add batch dimension
            elif weights.ndim == 2:
                weights = weights.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
        self._weights = weights

    def forward(self, *args, **kwargs):
        return EDMLossAbstract.__call__(self, *args, **kwargs)


class WeightedEDMLoss(WeightedEDMLossAbstract):
    def __init__(self, loss_type="L2", **kwargs):
        super().__init__(**kwargs)
        if loss_type == "L2":
            self.loss_func = lambda x: x**2
        elif loss_type == "L1":
            self.loss_func = lambda x: x.abs()
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

    def loss(self, preds, targets, sigma_weights, **kwargs):
        return self.weigh_loss(self.loss_func(preds - targets), multiply_weight=sigma_weights, **kwargs)


class WeightedEDMLossCRPS(WeightedEDMLossAbstract):
    def __call__(self, net, targets, predictions_post_process=None, targets_pre_process=None, **kwargs):
        rnd_normal1 = torch.randn([targets.shape[0], 1, 1, 1], device=targets.device)
        rnd_normal2 = torch.randn([targets.shape[0], 1, 1, 1], device=targets.device)
        sigma1 = (rnd_normal1 * self.P_std + self.P_mean).exp()
        sigma2 = (rnd_normal2 * self.P_std + self.P_mean).exp()
        weight1 = (sigma1**2 + self.sigma_data**2) / (sigma1 * self.sigma_data) ** 2
        weight2 = (sigma2**2 + self.sigma_data**2) / (sigma2 * self.sigma_data) ** 2

        y = targets
        n1 = torch.randn_like(y) * sigma1
        n2 = torch.randn_like(y) * sigma2
        D_yn1 = net(y + n1, sigma1, **kwargs)
        D_yn2 = net(y + n2, sigma2, **kwargs)
        D_yn = torch.stack([D_yn1, D_yn2], dim=0)  # (2, B, C, H, W), where 2 is the ensemble size
        # Take the min of the two weights
        weight_lam = torch.min(weight1, weight2)
        return self.weigh_loss(
            crps_ensemble(predictions=D_yn, observations=y, reduction="none"), multiply_weight=weight_lam
        )
        # Copilot suggestion:
        # CRPS = E[|D_yn1 - D_yn2|] - 0.5 * E[|D_yn1 - y|] - 0.5 * E[|D_yn2 - y|]
        # crps = (D_yn1 - D_yn2).abs().mean() - 0.5 * (D_yn1 - y).abs().mean() - 0.5 * (D_yn2 - y).abs().mean()
