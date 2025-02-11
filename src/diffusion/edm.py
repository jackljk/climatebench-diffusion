from abc import abstractmethod

import numpy as np
import torch

# from src.utilities.torch_utils import persistence
from src.diffusion._base_diffusion import BaseDiffusion
from src.losses.losses import AbstractWeightedLoss, crps_ensemble
from src.utilities.utils import get_logger


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

    def get_loss(self, images, labels=None, augment_pipe=None):
        return self.criterion(self, images, labels, augment_pipe)

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

    def get_loss(self, images, labels=None, augment_pipe=None):
        return self.criterion(self, images, labels, augment_pipe)

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)


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
        P_mean=-1.2,  # Mean of the noise level distribution.
        P_std=1.2,  # Standard deviation of the noise level distribution.
        force_unconditional=False,  # Ignore conditioning information?
        # Sampling parameters.
        num_steps=18,  # Number of steps in the sampling loop.
        rho=7,  # Exponent of the time step discretization.
        S_churn=0,  # Maximum noise increase per step.
        S_min=0,  # Minimum noise level for increased noise.
        S_max=float("inf"),  # Maximum noise level for increased noise.
        S_noise=1,  # Noise level for increased noise.
        heun: bool = True,  # Use Heun's method for the sampling loop.
        dtype="double",  # double or float
        **kwargs,  # Keyword arguments for the underlying model.
    ):
        kwargs["timesteps"] = num_steps
        super().__init__(**kwargs)
        self._USE_SIGMA_DATA = True
        self.use_fp16 = use_fp16
        self.sigma_min = sigma_min
        assert sigma_max is None, "Use sigma_max_inf instead of sigma_max for inference. "
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
        loss_kwargs = dict(P_mean=self.hparams.P_mean, P_std=self.hparams.P_std, sigma_data=self.sigma_data, **kwargs)
        log.info(f"Using EDM loss function: {loss_function}")
        if loss_function == "mse":
            loss_kwargs.pop("reduction", None)
            return EDMLoss(**loss_kwargs)
        elif loss_function == "wmse":
            return WeightedEDMLoss(**loss_kwargs, loss_type="L2")
        elif loss_function == "wmae":
            return WeightedEDMLoss(**loss_kwargs, loss_type="L1")
        elif loss_function == "wcrps":
            return WeightedEDMLossCRPS(**loss_kwargs)
        else:
            raise ValueError(f"Unknown loss type: {loss_function}")

    def forward(self, x, sigma, class_labels=None, force_fp32=False, **model_kwargs):
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
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        if self.label_dim == 0:
            class_labels = None
        elif class_labels is None:
            class_labels = torch.zeros([1, self.label_dim], device=x.device)
        else:
            class_labels = class_labels.to(torch.float32).reshape(-1, self.label_dim)
        dtype = torch.float16 if (self.use_fp16 and not force_fp32 and x.device.type == "cuda") else torch.float32

        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2).sqrt()
        c_in = 1 / (self.sigma_data**2 + sigma**2).sqrt()
        c_noise = sigma.log() / 4

        F_x = self.model((c_in * x).to(dtype), c_noise.flatten(), class_labels=class_labels, **model_kwargs)
        # assert F_x.dtype == dtype, f"{F_x.dtype} != {dtype}"
        D_x = c_skip * x + c_out * F_x.to(torch.float32)
        return D_x

    def get_loss(self, inputs, targets, return_predictions=False, **kwargs):
        # Shouldn't be needed anymore after using predictions_post_process inside the losses:
        # if len(targets.shape) == 5:  # (B, T, C, H, W)
        #     targets = targets.squeeze(1)  # (B, C, H, W)
        loss = self.criterion["preds"](self, images=targets, condition=inputs, **kwargs)
        # condition will be fed back to .forward() above as part of model_kwargs
        if return_predictions:
            return loss, None
        return loss

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)

    @staticmethod
    def edm_discretization(steps, sigma_min: float, sigma_max: float, rho: float):
        return (sigma_max ** (1 / rho) + steps * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho

    def edm_sampler(
        self,
        noise,
        randn_like=torch.randn_like,
        # sigma_min=0.002, sigma_max=80,
        **kwargs,
    ):
        dtype = torch.float64 if self.hparams.dtype == "double" else torch.float32

        def denoise(x, t):
            denoised = self(x, t, **kwargs).to(dtype)
            if self.hparams.guidance == 1:
                return denoised
            # Guided denoiser.
            kwargs_g = kwargs
            if self.guidance_model.model.hparams.force_unconditional:
                kwargs_g = {k: v for k, v in kwargs.items() if k != "dynamical_condition"}
            ref_Dx = self.guidance_model(x, t, **kwargs_g).to(dtype)
            denoised = ref_Dx.lerp(denoised, self.hparams.guidance)
            # = ref_Dx + guidance * (denoised - ref_Dx) = guidance * denoised + (1 - guidance) * ref_Dx
            return denoised

        # Adjust noise levels based on what's supported by the network.
        sigma_min = self.sigma_min  # max(sigma_min, self.sigma_min)
        sigma_max = self.sigma_max_inf  # min(sigma_max, self.sigma_max)
        rho = self.hparams.rho
        S_churn = self.hparams.S_churn
        S_min = self.hparams.S_min
        S_max = self.hparams.S_max
        S_noise = self.hparams.S_noise
        num_steps = self.hparams.num_steps
        # Time step discretization.
        step_indices = torch.arange(num_steps, dtype=dtype, device=noise.device)
        step_indices_normed = step_indices / (num_steps - 1)
        t_steps = self.edm_discretization(step_indices_normed, sigma_min, sigma_max, rho)
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
                t_hat = t_cur + gamma * t_cur
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
        batch_seeds = batch_seeds or torch.randint(0, 2**32, (condition.shape[0],), device=condition.device)
        rnd = StackedRandomGenerator(self.device, batch_seeds)
        init_latents_shape = (
            condition.shape[0],
            self.num_input_channels,
            self.spatial_shape_out[0],
            self.spatial_shape_out[1],
        )
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

    def __call__(self, net, images, labels, augment_pipe=None):
        rnd_uniform = torch.rand([images.shape[0], 1, 1, 1], device=images.device)
        sigma = self.sigma(1 + rnd_uniform * (self.epsilon_t - 1))
        weight = 1 / sigma**2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
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

    def __call__(self, net, images, labels, augment_pipe=None):
        rnd_uniform = torch.rand([images.shape[0], 1, 1, 1], device=images.device)
        sigma = self.sigma_min * ((self.sigma_max / self.sigma_min) ** rnd_uniform)
        weight = 1 / sigma**2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return {"loss": loss}


# ----------------------------------------------------------------------------
# Improved loss function proposed in the paper "Elucidating the Design Space
# of Diffusion-Based Generative Models" (EDM).


# @persistence.persistent_class
class EDMLossAbstract:
    def __init__(self, P_mean, P_std, sigma_data):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    @abstractmethod
    def loss(self, preds, targets, sigma_weights):
        pass

    def __call__(self, net, images, predictions_post_process=None, targets_pre_process=None, **kwargs):
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2
        y = images
        if targets_pre_process is not None:
            y = targets_pre_process(y)
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

        return {"loss": self.loss(D_yn, images, weight)}


class EDMLoss(EDMLossAbstract):
    def loss(self, preds, targets, sigma_weights):
        # loss y, , n, and D_yn have the same shape (B, C, H, W). weight has shape (B, 1, 1, 1)
        return (sigma_weights * ((preds - targets) ** 2)).mean()


class WeightedEDMLossAbstract(AbstractWeightedLoss, EDMLossAbstract):
    def __init__(self, P_mean, P_std, sigma_data, **kwargs):
        AbstractWeightedLoss.__init__(self, **kwargs)
        EDMLossAbstract.__init__(self, P_mean, P_std, sigma_data)

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

    def loss(self, preds, targets, sigma_weights):
        return self.weigh_loss(self.loss_func(preds - targets), multiply_weight=sigma_weights)


class WeightedEDMLossCRPS(WeightedEDMLossAbstract):
    def __call__(self, net, images, predictions_post_process=None, targets_pre_process=None, **kwargs):
        rnd_normal1 = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        rnd_normal2 = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma1 = (rnd_normal1 * self.P_std + self.P_mean).exp()
        sigma2 = (rnd_normal2 * self.P_std + self.P_mean).exp()
        weight1 = (sigma1**2 + self.sigma_data**2) / (sigma1 * self.sigma_data) ** 2
        weight2 = (sigma2**2 + self.sigma_data**2) / (sigma2 * self.sigma_data) ** 2

        y = images
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


# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# Wrapper for torch.Generator that allows specifying a different random seed
# for each sample in a minibatch.


class StackedRandomGenerator:
    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds]

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators])

    def randn_like(self, input):
        return self.randn(input.shape, dtype=input.dtype, layout=input.layout, device=input.device)

    def randint(self, *args, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randint(*args, size=size[1:], generator=gen, **kwargs) for gen in self.generators])
