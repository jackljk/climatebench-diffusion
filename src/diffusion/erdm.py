import os
import time
from collections import defaultdict
from functools import partial
from typing import Optional, Tuple, Union, Dict, Any

import einops
import numpy as np
import torch
from einops import rearrange, repeat
from tqdm import tqdm

from src.diffusion.edm import MPFourier, MPConv
from src.diffusion._base_diffusion import BaseDiffusion
from src.utilities.utils import get_logger, sample_random_real


log = get_logger(__name__)

# ----------------------------------------------------------------------------

import numpy as np
from scipy.stats import norm
import math

def discretize_lognormal(P_mean, P_std, N, sigma_min, sigma_max, offset, batch_size):
    """
    Discretizes a lognormal distribution into N segments with denser segments
    around the mode.

    Args:
        P_mean (float): Mean of the lognormal distribution.
        P_std (float): Standard deviation of the lognormal distribution.
        N (int): Number of segments.
        sigma_min (float): Minimum value of the range.
        sigma_max (float): Maximum value of the range.
        offset (float): Offset value between 0 and 1 (inclusive).

    Returns:
        list: A list of N values representing the boundaries or interpolated
              values for each segment based on the offset.
    """
    if P_mean <= 0:
        raise ValueError("P_mean must be positive.")

    if P_std < 0:
        raise ValueError("P_std cannot be negative.")

    if sigma_min >= sigma_max:
        raise ValueError("sigma_min must be less than sigma_max.")

    if not isinstance(N, int) or N < 1:
        raise ValueError("N must be a positive integer.")

    # 1. Calculate parameters of the underlying normal distribution
    sigma_log = math.sqrt(math.log(1 + (P_std / P_mean)**2))
    mu_log = math.log(P_mean) - 0.5 * sigma_log**2

    # 2. Calculate log-transformed min and max
    y_min = math.log(sigma_min)
    y_max = math.log(sigma_max)

    # 3. Calculate CDF values at y_min and y_max
    p_min = norm.cdf((y_min - mu_log) / sigma_log)
    p_max = norm.cdf((y_max - mu_log) / sigma_log)

    # 4. Generate N+1 equally spaced CDF values
    p_values = np.linspace(p_min, p_max, N + 1)

    # 5. Find the corresponding log-transformed boundaries
    y_boundaries = norm.ppf(p_values) * sigma_log + mu_log

    # 6. Calculate the boundaries in the original space
    x_boundaries = np.exp(y_boundaries)
    if torch.is_tensor(offset):
        assert offset.size(0) == batch_size
        x_boundaries = torch.tensor(x_boundaries, dtype=offset.dtype, device=offset.device)
        offset = offset.unsqueeze(1).repeat(1, N)
    else:
        x_boundaries = torch.tensor(x_boundaries, dtype=torch.float32, device=torch.device("cuda"))

    x_boundaries = x_boundaries.unsqueeze(0).repeat(batch_size, 1)
    left_ends = x_boundaries[:, :-1]
    right_ends = x_boundaries[:, 1:]

    # Vectorized parabolic interpolation
    parabolic_offset = 2.3 * offset * (1 - offset)
    linear_offset = offset * (2 * offset - 1)
    result = right_ends * (1 - parabolic_offset - linear_offset) + left_ends * (parabolic_offset + linear_offset)
    return result #torch.tensor(result, dtype=torch.float32)

# @persistence.persistent_class
class ERDM(BaseDiffusion):
    def __init__(
        self,
        schedule: str = "edm",  # edm | poly1 | poly2 -- support for different schedules within the window
        sigma_min=0.002,  # Minimum supported noise level. 0.002 in miikas's code
        sigma_max=None,  # Maximum supported noise level. 80.0 in miikas's code
        rho=7,  # Exponent of the time step discretization.
        discretization: str = "uniform",  # lognormal | uniform | normal
        # Range of sigma_min/max/rho for training. If None, always use sigma_min
        sigma_min_training: Optional[Tuple[float, float]] = None,
        sigma_max_training: Optional[Tuple[float, float]] = None,
        rho_training: Optional[Tuple[float, float]] = None,
        training_distribution: str = "uniform",  # uniform | loguniform | normal. Applies if any of the training ranges are specified
        same_schedule_per_batch: bool = True,  # If True, sample a different schedule for each batch
        # P_mean          = -1.2,             # Mean of the noise level distribution.
        # P_std           = 1.2,              # Standard deviation of the noise level distribution.
        use_lambda_weighting: bool = True,  # Use standard lambda weighting from EDM for the loss
        use_noise_logvar: bool = False,
        variance_loss: bool = True,  # Use a Kendall&Gal style loss weighting with learned per-frame variance -- found to be beneficial
        frame_weighting: bool = False,  # Use same (initial) weighting for all frames in the window
        compute_loss_per_sigma: bool = False,  # Compute loss for each sigma separately (for validation/debugging)
        # Use classical global mapping network based conditioning to inform network of fractional frame shift.
        # If False, the network will not be trained with randomly shifted noises, and will only handle the case offset=0! (XXX that mode is currently untested)
        use_map_noise: bool = True,
        force_pure_noise_last_frame: bool = False,  # If True, the last frame will be forced to be randn * sigma_max, instead of any potential initialization frame
        noise_prior: str = "independent",  # independent | mixed | progressive
        noise_prior_alpha: float = 0.0,  # Mixing parameter for the mixed noise prior. Not used if independent.
        # Use explicit learnable per-time per-channel weight and bias in network?
        # NOTE this is strictly different than the conditioning with frame shift above, but these could potentially just be unified under this mechanism?
        # The idea is to allow the network to learn a feature shift and scale based on the frame index.
        # Without this explicit information, the spatially invariant time-direction convolutions must somehow manage to "sense" the noise level within their footprint,
        # which can be learned due to lack of translation invariance, but may not be ideal.
        # Note that this is not an issue when time_to_channels is used, because each frame will be attached to a specific channel id.
        # 'direct' mode simply learns two scalars for each feature channel for each frame index for each conv filter -- might be reasonable for video where frame count is quite low
        # 'fouroer' learns a pair of continuous functions for each channel for each conv filter -- better for long windows, like in audio
        time_cond_mode: Optional[str] = None,  # None | 'direct' | 'fourier'
        time_to_channels: bool = True,
        conditional: bool = False,
        emb_transform: str = None,  # edm or none. if edm: log(sigma) / 4, else: log(sigma/2)
        a: float = None,
        b: float = None,
        # Sampling parameters.
        # how many frames to step per iteration - can be fractional, e.g. 0.3 will do three or four sub-steps per frame.
        # A frame is yielded every time the steps add up to a full integer   # RDM does 4 (?) sub-steps per frame = 0.25
        step=1.0,
        S_churn=0,  # Maximum noise increase per step.
        S_noise=1,  # Noise level for increased noise.
        S_ar_noise=1,
        heun=True,  # Use Heun's method for ODE integration.
        yield_denoised=False,
        denoiser_clip: Optional[float] = None,  # Clip the denoiser output to this range (None for no clipping)
        clamp_frame_idx_to_0: bool = True,  # Clamp the frame index to 0, to avoid negative indices in the network
        non_temporal_guidance: bool = False,  # Use autoguidance for sampling
        guidance_last_frames: Union[int, str] = "all",  # Number of (last) frames to use guidance on
        save_debug_sampling_to_file: bool = False,  # Save debug info to file during sampling
        learnable_schedule: bool = False,  # If True, learn the schedule as a list of scalars
        seq_len=None,  #  if not provided, will be inferred from the datamodule
        verbose=False,  # if True, print debug info
        datamodule_config=None,
        **kwargs,  # Keyword arguments for the underlying model.
    ):
        kwargs["timesteps"] = step
        self.seq_len = seq_len or datamodule_config.get("horizon", 1)  # + self.datamodule_config.get("window", 1)
        self.time_dim = 2  # (batch, channels, time, height, width)  # if 1, need to fix bug in torch.nn.functional.pad
        # self.time_dim = -4  # (batch, time, channels, height, width)  # if 1, need to fix bug in torch.nn.functional.pad
        super().__init__(**kwargs, datamodule_config=datamodule_config)
        self._USE_SIGMA_DATA = True
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max or float("inf")
        self.signal_mean = 0.0
        self.tmin = sigma_min
        self.tmax = sigma_max
        if learnable_schedule:
            if "edm" in schedule:
                if learnable_schedule == "v2":
                    self._rho_max = 100.0  # both for negative and positive, convexity: 3 -> 100 -> -100 -> -3
                    self.raw_sigma_min = torch.nn.Parameter(torch.log(torch.tensor(sigma_min)))
                    self.raw_sigma_delta = torch.nn.Parameter(torch.log(torch.tensor(sigma_max - sigma_min)))
                    rho = torch.nn.Parameter(torch.tensor(torch.pi))

                elif learnable_schedule in [True, "v0"]:
                    rho = torch.nn.Parameter(torch.tensor(float(rho), dtype=torch.float32))  # c))  # clamp to [4, 10]
                elif learnable_schedule == "v1":
                    # y = 7 + 5 * (torch.sigmoid(rho)-0.5)  # sigmoid gives values between 0 and 1, so multiply by 5 to get values between 5 and 10
                    rho = torch.nn.Parameter(torch.tensor(0.0))
                else:
                    raise ValueError(f"Invalid value for learnable_schedule: {learnable_schedule}")
            else:
                raise NotImplementedError(f"learnable_schedule not implemented for schedule {schedule}")

        self.exp = rho
        self.a = a
        self.b = b
        self.verbose = verbose
        self.time_conditioned = self.model.hparams.with_time_emb
        self._noise_rand_cache = dict()
        if noise_prior in ["mixed", "progressive"]:
            # Common parameters for both methods
            alpha_squared = noise_prior_alpha ** 2
            scaling_factor = 1 / (1 + alpha_squared)
            correlation_factor = alpha_squared * scaling_factor

            # Precompute scaling constants
            self._indep_scale = torch.sqrt(torch.tensor(scaling_factor))
            self._corr_scale = torch.sqrt(torch.tensor(correlation_factor))

        if self.time_conditioned:
            self.log_text.info(f"Using time conditioned model (based on {'offset' if time_to_channels else 'sigma'})")
        if schedule == "exp_a_b":
            assert a is not None and b is not None, "a and b must be provided for schedule exp_a_b"
            # assert sigma_min == 0, f"sigma_min must be 0 for schedule exp_a_b, got {sigma_min}"
            noises_min = float(exp_a_b(torch.tensor(0.0), a=a, b=b))
            noises_max = float(exp_a_b(torch.tensor(1.0), a=a, b=b))
            self.noise_func = partial(
                exp_a_b_normalized, tmin=sigma_min, tmax=sigma_max, noises_min=noises_min, noises_max=noises_max
            )

        assert force_pure_noise_last_frame in [
            True,
            False,
            "corrected",
            "plus_data",
        ], f"force_pure_noise_last_frame must be True, False or 'corrected', got {force_pure_noise_last_frame}"
        if use_noise_logvar:
            logvar_channels = 128  # Intermediate dimensionality for uncertainty estimation.
            log.info(f"Using log-variance with {logvar_channels} intermediate channels for noise weighting.")
            self.logvar_fourier = MPFourier(logvar_channels)
            # Zero init of MPConv does not seem to help, better to use randn init
            self.logvar_linear = MPConv(logvar_channels, 1, kernel=[])

        assert (
            0 <= S_churn < 1
        ), "churn must be greater than 0 and less than 1 (churn = 1 indicates a sub-step infinitely far into the future)"
        log.info(f"Using {S_churn=}, {step=}, {heun=}. {use_map_noise=}, {time_to_channels=}")
        if "edm" in schedule:
            assert 0 < sigma_min < sigma_max, "sigma_min must be greater than 0 and less than sigma_max"

    def get_exp(self, **kwargs):
        if self.hparams.learnable_schedule:
            if self.hparams.learnable_schedule == "v2":
                angle = self.exp % (2 * torch.pi)
                if angle < torch.pi:
                    rho = 3 + (self._rho_max - 3) * angle / torch.pi
                else:
                    rho = -self._rho_max + (self._rho_max - 3) * (angle - torch.pi) / torch.pi
                return rho
            elif self.hparams.learnable_schedule in [True, "v0"]:
                return torch.clamp(self.exp, 4, 10)
            elif self.hparams.learnable_schedule == "v1":
                return 7 + 5 * (torch.sigmoid(self.exp) - 0.5)
        elif self.hparams.rho_training is not None and self.training:
            return sample_random_real(*self.hparams.rho_training, self.hparams.training_distribution, **kwargs)
        else:
            return self.exp

    def get_tmax(self, **kwargs):
        if self.hparams.learnable_schedule == "v2":
            return self.get_tmin() + torch.exp(self.raw_sigma_delta)
        elif self.hparams.sigma_max_training is not None and self.training:
            return sample_random_real(*self.hparams.sigma_max_training, self.hparams.training_distribution, **kwargs)
        else:
            return self.tmax

    def get_tmin(self, **kwargs):
        if self.hparams.learnable_schedule == "v2":
            return torch.exp(self.raw_sigma_min)
        elif self.hparams.sigma_min_training is not None and self.training:
            return sample_random_real(*self.hparams.sigma_min_training, self.hparams.training_distribution, **kwargs)
        else:
            return self.tmin

    @property
    def channel_dim(self):
        return self.time_dim - 1  # channels are before time

    def _get_loss_callable_from_name_or_config(self, loss_function: str, **kwargs):
        """Return the loss function used for training.
        Function will be called when needed by the BaseModel class.
        Better to do it here in case self.* parameters are changed."""
        if self.hparams.variance_loss:
            if loss_function in ["l1", "l2"]:
                log.warning(f"The specified {loss_function=} won't be told about the variance loss.")
            else:
                lvdn_to_i_n = kwargs.get("learned_var_dim_name_to_idx_and_n_dims", {})
                lvdn_to_i_n["times"] = (self.time_dim, self.seq_len)  # (batch, channels, time, height, width)
                kwargs["learned_var_dim_name_to_idx_and_n_dims"] = lvdn_to_i_n
                if self.hparams.variance_loss == "with_channel":
                    kwargs["learn_per_dim"] = False  # Learn a (C, T) matrix rather than (C,) and (T,) vectors

        return super()._get_loss_callable_from_name_or_config(loss_function, **kwargs)

    # ----------------------------------------------------------------------------
    # Noise level interface (used by both training and sampling)

    # Computes the noise level to be applied at a specific frame within the window.
    # Note that the frame can be non-integer; this is required to support fractional frame stepping.
    # To reproduce "canonical" EDM schedule within window: noise_on_frame(torch.arange(self.seq_len))
    # Fractionally shifted schedules: noise_on_frame(torch.arange(self.seq_len) - shift) with shift in [0,1]
    # The definition is slightly odd with the built in shift of -1, so as to match the original.
    # Now, the 0'th element will have a small nonzero noise level which previously was explicitly set to 0,
    # and 1st element onward will be the former explicitly hardcoded schedule.
    def noise_on_frame(self, frame, B=None):
        B = B or frame.shape[-1]
        if self.hparams.same_schedule_per_batch:
            B = 1
        # Note: presently leaves a tiny residual amount of noise even at the leftmost frames in the window
        if self.hparams.schedule == "edm":

            def default_v(s):
                return ((1 - s) * self.tmin ** (1 / self.exp) + s * self.tmax ** (1 / self.exp)) ** self.exp

            f = default_v
            # XXX slightly messy but this way we get the same numerical levels as in EDM (not that this schedule is necessarily anything special in this context, but at least it's a baseline)
            noises = f((frame - 1) / (self.seq_len - 2))
        elif self.hparams.schedule == "edm_paper_reversed":

            def edm_reversed(s):
                exp = self.get_exp(size=B, device=frame.device)
                tmax = self.get_tmax(size=B, device=frame.device)
                tmin = self.get_tmin(size=B, device=frame.device)

                return (tmax ** (1 / exp) + s * (tmin ** (1 / exp) - tmax ** (1 / exp))) ** exp

            # assert torch.all((frames+1) / self.seq_len >= 0) and torch.all((frames+1) / (seq_len+1) <= 1)
            normalized = (frame + 1) / self.seq_len
            focus_point = 2
            sharpness = 3
            if self.hparams.discretization == "uniform":
                normalized = 1 - normalized
            elif self.hparams.discretization == "sine":
                # Shift to center around target_point instead of 0.5
                phase = np.pi * (1 - focus_point)
                normalized = (1 - normalized) * (np.sin(normalized * np.pi + phase) / np.sin(phase))
            elif self.hparams.discretization == "poly":
                a = 2.0 * (1.0 - focus_point)
                b = 2.0 * focus_point - 1.0
                # This polynomial preserves endpoints and creates concentration around focus_point
                t = normalized
                normalized = (1 - normalized) * (1 + a * t * (1 - t) + b * t * t * (1 - t))
            elif self.hparams.discretization == "poly2":
                a = 2.0 * (1.0 - focus_point) - 0.5
                b = 2.0 * focus_point - 1.0 + 1
                # This polynomial preserves endpoints and creates concentration around focus_point
                t = normalized
                normalized = (1 - normalized) * (1 + a * t * (1 - t) + b * t * t * (1 - t))
            elif self.hparams.discretization == "poly3":
                a = 2.0 * (1.0 - focus_point)
                b = 2.0 * focus_point - 1.0 +1
                # This polynomial preserves endpoints and creates concentration around focus_point
                t = normalized
                normalized = (1 - normalized) * (1 + a * t * (1 - t) + b * t * t * (1 - t))
            noises = edm_reversed(normalized)
            # frame+1 brings it to the non-negative range, and the division by seq_len brings it to [0,1], 1- reverses the schedule

        # A couple of experimental schedules. Note, these leave some visible noise in the generated video (can be removed by using the denoiser output; could this loosened requirement be beneficial for learning?)
        elif self.hparams.schedule == "poly1":
            func = lambda s: 0.1 + 1 * s + 4 * s**2 + 550 * s**16
            # t = frame / (self.seq_len - 1)
            t = (frame + 1) / self.seq_len
            noises_raw = func(t)
            noise_min = func(0)
            noise_max = func(1)
            noises = (noises_raw - noise_min) / (noise_max - noise_min) * (self.tmax - self.tmin) + self.tmin
        elif self.hparams.schedule == "poly2":
            # t = frame / (self.seq_len - 1)
            t = (frame + 1) / self.seq_len
            func = lambda s: 0.1 + 1 * s + 4 * s**3 + 300 * s**20
            noises_raw = func(t)
            noise_min = func(0)
            noise_max = func(1)
            noises = (noises_raw - noise_min) / (noise_max - noise_min) * (self.tmax - self.tmin) + self.tmin

        elif self.hparams.schedule == "exp_a_b":
            t = (frame + 1) / self.seq_len  # (self.seq_len - 1)
            noises = self.noise_func(t, a=self.a, b=self.b)
            # log.info(f"min_frame={frame.min().item()}, unique(t)={torch.unique(t).numpy()}, unique(noises)={torch.unique(noises).numpy().astype(np.float32)}")
            # f = lambda s: 1 - np.exp(-self.a * s ** self.b)
            # raw_values = f(t)
            # raw_min, raw_max = f(0), f(1)
            # normalized_values = (raw_values - raw_min) / (raw_max - raw_min) * (self.tmax - self.tmin) + self.tmin
            # return normalized_values
        else:
            raise NotImplementedError(f"Unknown schedule: {self.hparams.schedule}")
        # Ensure that the noise level is within the bounds, important when using padded sequences
        return torch.clamp(noises, max=self.tmax)

    # Whereas noise_on_frame() gives the plain list of noise level for a list of time instances,
    # noise_coeff() arranges these into a tensor that can be used to modulate an appropriately shaped
    # randn so that it'll be scaled to those noise levels. This is used in the sampler as well.
    def noise_coeff(self, x: torch.Tensor, offset=0) -> torch.Tensor:
        """Produce the tensor that modulates a randn addition on x"""
        B = x.shape[0]
        T = x.shape[self.time_dim]
        # NOTE: supporting other sequence lengths than self.seq_dim also, for sampling convenience and unification
        frames = einops.repeat(torch.arange(T, dtype=x.dtype, device=x.device), "t -> t b", b=B) # Frame idxs
        if not self.hparams.use_map_noise:
            if torch.is_tensor(offset):
                assert torch.all(offset == 0), f"Offset must be 0 if use_map_noise=False, got {offset}"
            else:
                assert offset in [0, 1], (f"Offset must be 0 if usecccccbngkrfvcnurubeltvvjetgbujnhtllfkjubbftl"
                                          f"_map_noise=False, got {offset}")

        if self.hparams.schedule == "experimental":
            noise_std = discretize_lognormal(1, 2.5, T, self.tmin, self.tmax, offset, B)
            noise_std = rearrange(noise_std, "b t -> t b")
        else:
            frames = frames - offset  # ... pushed into future by offset (XXX sign convention confusing)
            if self.hparams.clamp_frame_idx_to_0:
                # Clamp frames to be >= -1
                frames = torch.clamp(frames, min=-1)
            # log.info(f"len(frames)={len(frames)}. Unique values={torch.unique(frames).cpu().numpy()}")
            noise_std = self.noise_on_frame(frames, B=B)

        # Push the noisy dimension to the correct slot, e.g. 'b 1 t 1 1' if self.time_dim=2 and ndim = 5
        dim_selection = "b 1 " + " ".join(["t" if self.time_dim == dim else "1" for dim in range(2, x.ndim)])
        noise_std = einops.rearrange(noise_std, "t b -> " + dim_selection)
        return noise_std

    # Implements EDM paper preconditioning formulas for inputs and training loss
    def precond_scales(self, noise_std):
        noisy_var = self.sigma_data**2 + noise_std**2
        if self.hparams.force_pure_noise_last_frame == "corrected":
            # log.info("Correcting the last frame to be pure noise")
            assert self.time_dim == 2, "Only time_dim=2 supported for force_pure_noise_last_frame"
            assert (
                noise_std.shape[self.time_dim] >= self.seq_len
            ), f"Expected noise_std to have at least {self.seq_len} frames, got {noise_std.shape}"
            # Correct the last frame to be pure noise (since we hard code the corresponding frames to have 0 signal)
            noisy_var[:, :, self.seq_len - 1 :] = noise_std[:, :, self.seq_len - 1 :] ** 2

        noisy_std = noisy_var.sqrt()

        c_in = 1 / noisy_std

        c_skip = self.sigma_data**2 / noisy_var
        c_out = self.sigma_data * noise_std / noisy_std
        sqrt_lambda = noisy_std / (self.sigma_data * noise_std)
        #           = (sigma_data ** 2 + noise_std ** 2)**0.5 / (sigma_data * noise_std)
        #           = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        return c_in, c_out, c_skip, sqrt_lambda

    def sample_noise_tensor(self, x_like: torch.Tensor, rnd_gen=None, use_cache: bool = False) -> torch.Tensor:
        """Sample noise for the given tensor, with the same shape."""
        rndn_handle = torch.randn_like if rnd_gen is None else rnd_gen.randn_like
        if self.hparams.noise_prior == "independent":
            return rndn_handle(x_like)
        elif self.hparams.noise_prior in ["mixed", "progressive"]:
            # See "Preserve Your Own Correlation: A Noise Prior for Video Diffusion Models"
            # Recommended alpha values from there: mixed=1 or progressive=2
            assert self.time_dim == 2, f"Only time_dim=2 supported for {self.hparams.noise_prior} noise prior"
            
            # Get time dimension length and initialize noise tensor
            time_len = x_like.shape[self.time_dim]
            if self.hparams.noise_prior == "mixed":
                # Sample shared noise from N(0, alpha**2/(1+alpha**2) I) and indep noise i from N(0, 1/(1+alpha**2) I)
                # Noise for frame i = shared + indep_i

                # Sample shared noise component common to all frames
                shared_noise = rndn_handle(x_like[:, :, 0:1, ...])
                if use_cache:
                    self._noise_rand_cache["shared_noise"] = shared_noise

                # Sample independent noise components for each frame
                indep_noise = self._indep_scale * rndn_handle(x_like)
                
                # Combine shared and independent noise
                noise = self._corr_scale * shared_noise.expand_as(x_like) + indep_noise
            else:  # progressive
                # Sample init noise from N(0,I) and indep noise i from N(0, 1/(1+alpha**2) I)
                # Noise for frame i eps_i = alpha**2/(1+alpha**2) eps_{i-1} + indep_i
                noise = torch.zeros_like(x_like)
                # Sample initial noise for the first frame
                noise[:, :, 0] = rndn_handle(x_like[:, :, 0])
                
                # Progressively generate noise for subsequent frames
                for t in range(1, time_len):
                    # Auto-regressive component from previous frame
                    ar_noise = noise[:, :, t-1]
                    
                    # Independent noise component
                    indep_noise = self._indep_scale * rndn_handle(x_like[:, :, t])
                    
                    # Combine for current frame
                    noise[:, :, t] = self._corr_scale * ar_noise + indep_noise
                if use_cache:
                    self._noise_rand_cache["shared_noise"] = noise[:, :, -1:]
                
            return noise
        else:
            raise ValueError(f"Unknown noise_prior: {self.hparams.noise_prior}. Expected one of: independent, mixed, progressive")

    def sample_additional_noise_frame(self, size, rnd=None, **kwargs):
        rndn_handle = torch.randn if rnd is None else rnd.randn
        if self.hparams.noise_prior == "independent":
            return rndn_handle(size, **kwargs)
        elif self.hparams.noise_prior in ["mixed", "progressive"]:
            assert size[self.time_dim] == 1, f"Only support sampling a single frame, got {size}"
            shared_noise: torch.Tensor = self._noise_rand_cache.get("shared_noise", None)
            assert shared_noise is not None, "Shared noise must be cached for mixed or progressive noise prior"
            new_noise = self._corr_scale * shared_noise + self._indep_scale * rndn_handle(size, **kwargs)
            if self.hparams.noise_prior == "progressive":
                # Update the shared noise cache
                self._noise_rand_cache["shared_noise"] = new_noise
            return new_noise
        else:
            raise ValueError(f"Unknown noise_prior: {self.hparams.noise_prior=}.")


    def forward(
        self,
        x: torch.Tensor,
        loss: bool = False,
        # If True, calculate the training loss. Assume x is a clean video sample; noise will be added internally.
        offset=None,
        # Fractional frame offset in [0,1] -- used to condition the network to inform it of the fractionally shifted noise levels (and shift added noise if loss==True)
        dtype=torch.float32,
        condition: Optional[torch.Tensor] = None,
        return_debug: bool = False,
        force_add_rolling_noise: bool = False,
        loss_kwargs: Optional[Dict[str, Any]] = None,
        **forward_kwargs,
    ) -> Dict[str, Any]:
        loss_kwargs = loss_kwargs or {}
        assert (
            x.shape[self.time_dim] == self.seq_len
        ), f"Expected sequence length {self.seq_len} at time dim{self.time_dim}, got {x.shape}"

        bs = x.size(0)  # batch size

        is_manual_offset = offset is not None
        if is_manual_offset:
            assert not torch.is_tensor(offset) or offset.ndim == 0, f"Offset must be scalar, got {offset.ndim=}"
            # if a scalar is given, apply the same offset to all entries in the batch
            offset = offset * torch.ones([bs], device=x.device, dtype=x.dtype)
        if offset is None:
            # if no offset is given, randomize (this is for training)
            offset = torch.rand([bs], device=x.device, dtype=x.dtype)
        if not self.hparams.use_map_noise:
            # XXX intended behavior: setting self.use_map_noise=False will disable any capabilities to handle offset !=0; this also means random jittering is disabled in training. XXX untested
            offset = 0 * offset  # offset_label = 2 * offset - 1  # scale it to range [-1,1] for conditioning

        # given the offset, these are the noise levels per frame (sigmas)
        noise_std = self.noise_coeff(x, offset=offset)
        if is_manual_offset:
            loss_kwargs["sigmas"] = noise_std
        if loss or force_add_rolling_noise:
            # Add the appropriate amount of noise if training
            if self.hparams.force_pure_noise_last_frame in [True, "corrected"]:
                # Set last frame to pure noise (last index in the time dimension)
                assert self.time_dim == 2, "Only time_dim=2 supported for force_pure_noise_last_frame"
                # Set to zero, so that randn * sigma_max will be pure noise (since it'll be added to 0)
                x[:, :, -1] = torch.zeros_like(x[:, :, -1])
            x_noisy = x + self.sample_noise_tensor(x, use_cache=False) * noise_std
        else:
            # If not training, the data is already noise
            x_noisy = x

        x_noisy = x_noisy - self.signal_mean

        c_in, c_out, c_skip, sqrt_lambda = self.precond_scales(noise_std)
        # x_in is the noisy clip of same shape as original x, and normalized to unit stdev at each frame -- will be fed to the network
        x_in = c_in * x_noisy

        if self.hparams.time_to_channels:
            if self.hparams.conditional:
                # Concatenate condition in time dimension before stacking time and channel dimensions
                assert condition is not None, "Condition must be provided for conditional RDM"
                try:
                    x_in = torch.cat([condition, x_in], dim=self.time_dim)
                except Exception as e:
                    raise ValueError(
                        f"Condition shape {condition.shape} does not match x_in shape {x_in.shape}. time_dim={self.time_dim}"
                    ) from e
            # Bring the self.time_dim dimension next to channel dimension, and merge (see other comments on time_to_channels for the idea)
            # D = len(x.shape)  # number of dims
            t_dim = x_in.shape[self.time_dim]  # might be different from self.seq_len if we're sampling with padding
            if self.hparams.conditional:
                t_dim_out = t_dim - 1
            else:
                t_dim_out = t_dim
            ein_orig = "n c t ..."  # e.g. 'n c d2 t d4
            ein_packed = "n (c t) ..."  # e.g. 'n (c t) d2 d4'
            x_in = einops.rearrange(x_in, ein_orig + " -> " + ein_packed)
            time_cond = offset

        else:
            # EDM paper preconditioning: use the log of the noise level as a time-conditioning signal
            if self.hparams.emb_transform == "edm":
                time_cond = noise_std.flatten(start_dim=1).log() / 4
            else:
                # flatten squeezes all singleton dims after 1.
                time_cond = (0.5 * noise_std.flatten(start_dim=1)).log()
            if self.hparams.conditional:
                assert condition is not None, "Condition must be provided for conditional RDM"
                # Reshape condition to repeat it along the time dimension, copying it to the batch dimension
                if condition.shape[2] == 1:
                    raise ValueError(
                        "Condition shape must have time dimension, got 1"
                    )  # not necessary, but let's be sure for now we don't mess up
                    condition = einops.repeat(condition, "b c 1 h w -> (b t) c h w", t=self.seq_len)
                else:
                    # condition.shape=torch.Size([B, 69, 8, 240, 121]), x_in.shape=torch.Size([B, 69, 8, 240, 121])
                    assert condition.shape[2] == self.seq_len, f"{condition.shape=} does not match {x_in.shape=}"
                    condition = einops.rearrange(condition, "b c t h w -> (b t) c h w")
                forward_kwargs["condition"] = condition  # Let the network use the condition signal on its own.

        # Run the network to obtain the raw prediction (note: this is NOT the denoising, but, depending on the choices, the mixture prediction as per EDM preconditioning principles)
        # log.info(f"{x_in.shape=}, {time_cond.shape=}, {len(forward_kwargs)=}, {x.shape=}")
        Fx = (
            self.model(x_in, **forward_kwargs).to(dtype)
            if not self.time_conditioned
            else self.model(x_in, time_cond, **forward_kwargs).to(dtype)
        )

        if self.hparams.time_to_channels:
            # Revert the dimension shuffle from corresponding block above
            Fx = einops.rearrange(Fx, ein_packed + " -> " + ein_orig, n=bs, t=t_dim_out)

        # mix the network prediction and noisy signal as per EDM preconditioning, to obtain the denoised video
        x_denoised = c_skip * x_noisy + c_out * Fx
        x_denoised = x_denoised + self.signal_mean  # add back the mean we subtracted earlier

        # If we're training, compute the loss and return it, instead of returning the denoised clip
        if loss:
            multiply_weight = None
            # multiply_weight = 1  # Currently not using any noise level dependent gradient scaling (note, situation is different from MC sampling of training noise levels in EDM paper -- we are doing all of them at once). TODO think of what would be the right thing
            # error = x - x_denoised
            # Old code:
            # error = error * sqrt_lambda  # Scale the per-frame gradient to unity (at init)
            # error = error.square()
            # if self.hparams.variance_loss:
            #     # A Kendall&Gal style variance-weighted loss. In their paper the idea is to predict the uncertainty per pixel, alongside the prediction.
            #     # The mechanism adds a weighting term that scales the usual L2 error. This has the side effect of dynamically scaling the gradient magnitude to unity.
            #     # We don't care about the uncertainty, but we do find the gradient scaling generally helpful.
            #     # In practice it's a not ideal to use a neural network to predict the per-pixel uncertainties and average them (as in k-diffusion github repo).
            #     # Instead here we simply learn the weighting for each frame as a scalar, and use it to implement the dynamic gradient normalization.
            #     # (This is related to something we'll be publishing as part of a different paper later -- contact Miika or Tero if clarification needed...)
            #     # Move the time dimension to zeroth position
            #     # log.info(f"{error.shape=} {self.log_var.shape=}, {sqrt_lambda.shape=}")  # [B, C, T, H, W], [T], [B, T]
            #     error = torch.permute(
            #         error, (self.time_dim,) + tuple(i for i in range(len(error.shape)) if i != self.time_dim)
            #     )
            #     # Average over all dimensions but first (time) -- this is the per-timestep error
            #     error = error.mean(dim=tuple(range(1, len(error.shape))))
            #     # log.info(f"{error.shape=}")  # [T,]
            #     # We now have the per-timestep error. Weight with learned variance and add the corresponding penalty.
            #     # NOTE self.log_var is a learnable parameter, so training will modify it
            #     error = error / self.log_var.exp() + self.log_var
            #     # loss = vloss = (vloss * weight).mean().float()
            #
            # # Just the standard error averaged uniformly over frames
            # loss = (error * weight).mean() #.float()
            # return {"loss": loss}
            if self.hparams.use_lambda_weighting == "inverse":
                multiply_weight = 1 / sqrt_lambda  #(sqrt_lambda ** 2)
            elif self.hparams.use_lambda_weighting:
                multiply_weight = sqrt_lambda**2  # Will be applied after squaring the error
            if self.hparams.frame_weighting == "reverse":
                # Assign higher loss weights to last frames
                f_weight = torch.arange(1, self.seq_len + 1, device=x.device, dtype=x.dtype) / self.seq_len
                f_weight = f_weight.reshape((1, *sqrt_lambda.shape[1:]))
                multiply_weight = multiply_weight * f_weight if multiply_weight is not None else f_weight
            elif "lognormal" in self.hparams.frame_weighting:
                P_mean, P_std = 0.5, 1.2
                if "_" in self.hparams.frame_weighting:
                    # Split P_mean (and optionally P_std) from the string depending on the number of underscores
                    split = self.hparams.frame_weighting.split("_")
                    assert len(split) in [2, 3], f"Expected 2 or 3 parts in {self.hparams.frame_weighting=}"
                    P_mean, P_std = map(float, split[1:]) if len(split) == 3 else (float(split[1]), P_std)

                f_weight = calculate_lognormal_weight(noise_std, P_mean, P_std)
                multiply_weight = multiply_weight * f_weight if multiply_weight is not None else f_weight

            if self.hparams.use_noise_logvar:
                logvars = self.logvar_linear(self.logvar_fourier(noise_std.flatten().log() / 4))
                assert self.time_dim == 2, "Only time_dim=2 supported for use_noise_logvar"
                # Reshape back to (B, T) shape and specify the dimensions to apply logvars to
                loss_kwargs["batch_logvars"] = (logvars.reshape(bs, -1), (0, self.time_dim))

            # Weight will be appropriately multiplied or combined with other weights in the loss function
            loss = self.loss_wrapper(preds=x_denoised, targets=x, multiply_weight=multiply_weight, **loss_kwargs)
            if not self.training and not self.hparams.use_lambda_weighting:
                # Compute what lambda weighted loss would be, for comparison
                loss_lambda = self.loss_wrapper(preds=x_denoised, targets=x, multiply_weight=sqrt_lambda**2, **loss_kwargs)
                for k, v in loss_lambda.items():
                    loss[k.replace("loss", "loss_lambda")] = v.detach().cpu() if torch.is_tensor(v) else v
            return loss

        elif return_debug:
            return dict(
                x_denoised=x_denoised,
                x_noisy=x_noisy,
                x_in=x_in,
                noise_std=noise_std,
                c_in=c_in,
                c_out=c_out,
                c_skip=c_skip,
                sqrt_lambda=sqrt_lambda,
                Fx=Fx,
                offset=offset,
            )

        return x_denoised

    def loss_wrapper(self, sigmas: torch.Tensor = None, **kwargs):
        loss = self.criterion["preds"](**kwargs)
        if kwargs.get("return_intermediate", False):
            if sigmas is not None:
                sigmas = sigmas.squeeze()
                if sigmas.ndim > 1:
                    assert torch.allclose(sigmas[0], sigmas[1]), f"{sigmas.shape=} {sigmas[0]=}, {sigmas[1]=}."
                    sigmas = sigmas[0, :]
            # Make sure the loss is a scalar:
            #   keys: unweighted_loss, weighted_loss_before_vars, weighted_loss_after_vars
            temp_losses_keys = ["unweighted_loss", "weighted_loss_before_vars", "weighted_loss_after_vars"]
            for k in temp_losses_keys:
                if k in loss.keys():
                    loss_values = loss.pop(k)  # Vectorized across time dimension
                    if isinstance(loss_values, float) or loss_values.numel() == 1:
                        loss[k] = float(loss_values)
                        continue  # Skip if it's a scalar
                    elif self.hparams.use_noise_logvar:  # Vectorized across batch and time dimension
                        loss_values = loss_values.mean(dim=0)  # Take mean over batch dimension
                    elif not self.hparams.variance_loss:
                        loss_values = loss_values.mean(dim=[i for i in range(0, loss_values.ndim) if i != self.time_dim])
                    if sigmas is not None:
                        assert loss_values.shape == sigmas.shape, f"{k=}, {sigmas.shape=}, got {loss_values.shape=}"
                    for i, loss_value_i in enumerate(loss_values):
                        if sigmas is not None:
                            sigma = sigmas[i].item()
                            n_decimals = 9 if sigma < 0.03 else 7 if sigma < 1 else 3
                            key = f"{k}_per_noise_level/sigma{sigmas[i].item():.{n_decimals}f}"
                        else:
                            key = f"{k}/time_{i}"
                        loss[key] = float(loss_value_i)
            if sigmas is not None:
                loss["x_axes"] = {"sigma": list(sigmas.cpu())}
        return loss

    def get_loss(self, inputs, targets, return_predictions=False, predictions_post_process=None, **kwargs):
        assert inputs is None, f"Expected inputs to be None, got {inputs}"
        loss = self.forward(targets, loss=True, **kwargs)
        if self.hparams.compute_loss_per_sigma:
            loss.update(self.get_loss_vs_sigmas(targets, **kwargs))
        if return_predictions:
            return loss, None
        return loss


    def get_loss_vs_sigmas(self, targets, **kwargs) -> Dict[str, float]:
        """Calculate the loss as a function of noise levels."""
        losses = dict()
        offsets = torch.linspace(0, 1, 50, device=targets.device)
        for i, offset in enumerate(tqdm(offsets, desc="Calculating loss vs sigmas")):
            loss_offset = self.forward(targets, loss=True, offset=offset, **kwargs)
            for k, v in loss_offset.items():
                if "_per_noise_level" in k:
                    if k in losses:
                        print(f"Key {k} already in loss dict {v=}, {losses[k]=}")
                        losses[k] = [losses[k], v] if isinstance(losses[k], float) else losses[k] + [v]
                    else:
                        losses[k] = v
                elif k == "x_axes":
                    if i == 0:
                        losses[k] = v
                    else:
                        for x_axis_name, x_axis_subset in v.items():
                            losses[k][x_axis_name] = losses[k][x_axis_name] + x_axis_subset
                else:
                    losses[k + f"/offset_{offset:.3f}"] = float(v)
        # Take mean over any repeated losses
        for k in list(losses.keys()):
            if "_per_noise_level" in k and isinstance(losses[k], list):
                losses[k] = np.mean(losses[k])
        # Sort the sigma values, from smallest to largest
        losses["x_axes"] = {k: sorted(v) for k, v in losses["x_axes"].items()}
        return losses

    @torch.inference_mode()
    def sample(
        self, prompt, dtype=torch.float32, batch_seeds=None, condition=None, **kwargs
    ):  # Salva: dtype=torch.float64!
        batch_seeds = batch_seeds or torch.randint(0, 2**32, (prompt.shape[0],), device=prompt.device)
        rnd = StackedRandomGenerator(self.device, batch_seeds)
        rnd_churn = StackedRandomGenerator(self.device, batch_seeds + 123) if self.hparams.S_churn > 0 else None
        # init_latents_shape = (prompt.shape[0], self.num_input_channels, self.spatial_shape[0], self.spatial_shape[1])
        # latents = rnd.randn(init_latents_shape, dtype=prompt.dtype, layout=prompt.layout, device=prompt.device)
        self.nfe = 0

        net = self
        step = self.hparams.step
        churn = self.hparams.S_churn
        t0 = 0.0  # start timing from 0.0
        step_ode = step / (1 - churn)  # length of the step we'll take forward following the ODE
        S_ar_noise = self.hparams.S_ar_noise
        S_noise = self.hparams.S_noise
        heun = self.hparams.heun
        denoiser_clip = self.hparams.denoiser_clip
        yield_denoised = self.hparams.yield_denoised

        if not self.hparams.conditional and condition is not None:
            del condition
            condition = None

        # We'll maintain a window that's slightly longer than that net.seq_len,
        # because we might be accessing noise from farther into future.
        # In some settings, step could be much more than 1 (e.g. in audio, we'll want to generate hundreds of PCM samples
        # per iteration), all of this generality ensures the computations land right. # Salva: Why this for audio?
        padding = int(step_ode) + 1

        # Helper function to extract an i-offset network-seq-len window
        # from the full working sequence (to be fed to the network)
        def win(x, i):
            nonlocal net  # variable is defined in the outer scope and will be modified inside the function.
            return select_range(x, dim=net.time_dim, start=i, end=i + net.seq_len)

        # Another helper to "un-win()": used to extend the network estimate to the trivial pad frames
        def pad(x, pre, post, mode="replicate"):
            nonlocal net
            # An annoying dimension calculation due to parametrization of .pad
            # Make a leading pair of (0,0)'s for every dimension to the right of the time dimension (so counted from right).
            # Furthermore, we can't just leave it there, but have to zero-fill the full number of "spatial dimensions",
            # i.e. anything beyond B, C
            return torch.nn.functional.pad(
                x, (0, 0) * (len(x.shape) - 1 - net.time_dim) + (pre, post) + (0, 0) * (net.time_dim - 2), mode=mode
            )

        # Insert copied frames into the initial prompt window according to the padded sequence length we'll be using
        # - they'll be fully under the noise, so no big deal what the content is (?)
        prompt = pad(prompt, 0, padding)  # pad `padding` frames to the right (i.e. to the future) of the prompt
        if self.hparams.force_pure_noise_last_frame in [True, "corrected"]:
            assert self.time_dim == 2, "Only time_dim=2 supported for force_pure_noise_last_frame"
            assert (
                prompt.shape[self.time_dim] == self.seq_len + padding
            ), f"Expected sequence length {self.seq_len + padding} at time dim{self.time_dim}, got {prompt.shape} (padding={padding})."
            # Set to zero, so that randn * sigma_max will be pure noise (since it'll be added to 0)
            prompt[:, :, self.seq_len - 1 :] = torch.zeros_like(prompt[:, :, self.seq_len - 1 :])
            # seq_len - 1 is the last frame of the original prompt, seq_len is the first frame of the padding, set all of them to 0

        # self.log.info(f"ADDING NOISE TO PROMPT")
        # Our initial working window, with noise applied according to the schedule ramp specified in the network.
        # If you visualize this tensor, you'll see a short video clip (of length net.seq_len+padding)
        # with clean early frames and noise level exploding over time
        per_frame_noise_coeffs = net.noise_coeff(prompt, offset=0.0)
        # log.info(f"Shapes of prompt: {prompt.shape}, noise_coeff: {per_frame_noise_coeffs.shape}")
        # log.info(f"Frame-wise coeffs=", per_frame_noise_coeffs[0].squeeze())
        # log.info(f"Padding={padding}, step={step}, step_ode={step_ode}, yield_denoised={yield_denoised}")  # Padding=1, step=0.7, step_ode=0.7368421052631579
        # assert that noise_coeff is the same for all batch elements (sanity check)
        assert torch.all(
            per_frame_noise_coeffs[0].squeeze() == per_frame_noise_coeffs[1].squeeze()
        ), f"per_frame_noise_coeffs.shape={per_frame_noise_coeffs.shape}. prompt.shape={prompt.shape}\nidx[0]={per_frame_noise_coeffs[0].squeeze()}, idx[1]={per_frame_noise_coeffs[1].squeeze()}"
        # noise = S_ar_noise * torch.randn(prompt.shape, dtype=dtype, generator=latent_rng, device=prompt.device)
        noise = S_ar_noise * self.sample_noise_tensor(prompt, rnd, use_cache=True)
        x0 = prompt + per_frame_noise_coeffs * noise
        # self.log.info(f"NOISE ADDED TO PROMPT")

        # Denoiser (will be called twice if heun==True, so isolated here)
        def denoise(x, t, sigmas=None, cond=None, **denoise_kwargs):
            # extract from the current integer offset into our frame window as the noisy sequence,
            # pass the fractional offset part as the offset, and denoise with network
            x_in_net = win(x, int(t))
            denoise_kwargs["condition"] = cond
            Dx = net(x_in_net, offset=frac(t), dtype=dtype, **denoise_kwargs)
            # log.info(f"Shapes of x: {x.shape}, win(x, int(t)): {win(x, int(t)).shape}, Dx: {Dx.shape},"
            #       f" t={t}, int(t)={int(t)} frac(t): {frac(t)}, padding: {pad(Dx, int(t), padding - int(t)).shape}")
            if self.hparams.guidance != 1:
                guidance_kwargs = denoise_kwargs.copy()
                if self.hparams.non_temporal_guidance:
                    guidance_kwargs["skip_temporal_blocks"] = True
                if self.guidance_model is None:  # kind of autoguidance
                    assert guidance_kwargs["skip_temporal_blocks"]
                    ref_Dx = net(x_in_net, offset=frac(t), dtype=dtype, **guidance_kwargs)
                elif isinstance(self.guidance_model.model, ERDM):
                    self.guidance_model.to(self.device)
                    ref_Dx = self.guidance_model(x_in_net, offset=frac(t), dtype=dtype, **guidance_kwargs)
                else:
                    self.guidance_model.to(self.device)
                    x_in_img_edm = x_in_net
                    sigmas = win(sigmas, int(t)).flatten(start_dim=1)
                    if self.hparams.guidance_last_frames != "all":
                        sigmas = sigmas[:, -self.hparams.guidance_last_frames :]
                        x_in_img_edm = x_in_img_edm[:, :, -self.hparams.guidance_last_frames :]
                    num_t = x_in_img_edm.shape[net.time_dim]
                    sigmas = rearrange(sigmas, "b t -> (b t)")
                    x_in_img_edm = rearrange(x_in_img_edm, "b c t h w -> (b t) c h w")
                    if kwargs.get("static_condition") is not None:
                        # Copy the static condition to all time steps b c' h w -> (b t) c' h w
                        guidance_kwargs["static_condition"] = repeat(
                            guidance_kwargs["static_condition"], "b c2 h w -> (b t) c2 h w", t=num_t
                        )

                    ref_Dx = self.guidance_model(x_in_img_edm, sigma=sigmas, **guidance_kwargs)
                    ref_Dx = rearrange(ref_Dx, "(b t) c h w -> b c t h w", t=num_t)
                if self.hparams.guidance_last_frames == "all":
                    Dx = ref_Dx.lerp(Dx, self.hparams.guidance)
                else:
                    assert net.time_dim == 2, "Only time_dim=2 supported for guidance_last_frames"
                    nf = self.hparams.guidance_last_frames
                    if ref_Dx.shape[net.time_dim] > nf:
                        ref_Dx = ref_Dx[:, :, -nf:]
                    Dx[:, :, -nf:] = ref_Dx.lerp(Dx[:, :, -nf:], self.hparams.guidance)
            # Extend the denoiser output to cover the entire window, by appropriate replication padding
            Dx = pad(Dx, int(t), padding - int(t))  # for t=0, this pads the right side of the frame
            range_kwargs = dict(dim=net.time_dim, start=None, end=int(t))
            # dx_old = Dx.clone()
            Dx[slice_range(**range_kwargs, of=len(Dx.shape))] = select_range(x, **range_kwargs)
            # For int(t)=0 ==> torch.allclose(Dx, dx_old) == True
            if denoiser_clip:
                Dx = torch.clip(Dx, -denoiser_clip, denoiser_clip)
            self.nfe += 1
            return Dx

        curr_step = 0  # not used by algorithm, but just to keep track...
        curr_frame = 0

        # Iterate forever over steps
        # Create a dict that logs all important tensors and values for each step
        log_dict = defaultdict(list) if self.hparams.save_debug_sampling_to_file else None
        while True:
            # By this point, t0 has accumulated all the fractional steps we've taken so far,
            # minus all the full frames we've yielded; it's the float offset into the padded frame window we maintain
            t1 = t0 + step_ode  # the global time we'll land after the ODE substep
            t2 = t0 + step  # ... and after the churn/backtracking (if it's enabled)
            # To avoid floating point impurities, round to 6 decimal places
            t0, t1, t2 = round(t0, 6), round(t1, 6), round(t2, 6)

            # Collect some noise level tensors which can be used to set a randn() output
            # to the noise ramps with appropriate time offsets
            # These are one-dimensional tensors of length net.seq_len, padded with appropriate dimensions for broadcast
            # (for video, it'll be of shape [batchsize, 1, net.seq_len, 1, 1])
            sigma_t0 = net.noise_coeff(x0, offset=t0)  # noise level at beginning of step
            sigma_t1 = net.noise_coeff(x0, offset=t1)  # ... at end of ODE step but before churn
            sigma_t2 = net.noise_coeff(x0, offset=t2)  # ... and after adding churn (passed to next iter)
            # print(f"Shapes of sigma_t0: {sigma_t0.shape}, sigma_t1: {sigma_t1.shape}, sigma_t2: {sigma_t2.shape}")
            # print(f"Frame-wise coeffs=", sigma_t0[0].squeeze(), sigma_t1[0].squeeze(), sigma_t2[0].squeeze())

            # Note that there is no kind of a "variance preserving" or such scale schedule -
            # the added noise levels are very large, e.g. up to 80 by default (where the data is in [-1,1]).
            # The network internal preconditioning takes care of scaling the signal to a palatable form,
            # and we don't worry about that in this function at all. This also keeps the mixing formulas as simple lerps.
            # The ramp picture in my presentation can be a bit misleading in this way
            # (it's not a "sigmoid" but an unbounded growth)

            # some debug info to indicate the integer offsets into the window, and fractionals offset within those
            # print(f'frame {curr_frame}, step {curr_step}: ' + ' -> '.join([f'{int(t)} + {frac(t)}' for t in [t0, t1, t2]]))

            ###################
            # ODE SUB-STEP

            # Call the network on the appropriate window
            # Denoised frame window. If you visualize this, you'll see a short video of increasingly blurry frames,
            # reflecting the uncertainty of the future evolution.
            # x0,Dx0.shape = [B, C, seq_len+padding, H, W]
            Dx0 = denoise(x0, t0, sigmas=sigma_t0, cond=condition, **kwargs)

            # Mix between noisy and denoised frames in the exact per-frame proportion that
            # leaves us with noise levels in sigma_t1
            # Count num nan or inf in sigma_t1/sigma_t0
            if self.verbose:
                div = 1 - sigma_t1 / sigma_t0  # future noise level / current noise level
                assert torch.allclose(div[0], div[1]), f"div[0] != div[1] for frame {curr_frame} and step {curr_step}"
                # log.info(f"t0={t0}, t1={t1}, t2={t2}.\t Num nan in sigma_t1/sigma_t0: {torch.isnan(div).sum()}, in sigma_t2={torch.isnan(sigma_t2).sum()}")
                # join a str sigma_t0[0] -> sigma_t2[0] -> div[0] \t sigma_t0[1] -> sigma_t2[1] -> div[1] ...
                # print_steps = [0, 1, sigma_t0.shape[2] // 2, sigma_t0.shape[2] - 1 - padding]
                # log.info("\t".join([f"{i}: {sigma_t0[0, 0, i].item():.4f} -> {sigma_t1[0, 0, i].item():.4f} -> {sigma_t2[0, 0, i].item():.4f}; {div[0, 0, i].item():.4f}" for i in print_steps]))
                # log.info(f"sigma_t0.min;max={sigma_t0.min():.4f};{sigma_t0.max()}, sigma_t1.min;max={sigma_t1.min():.4f};{sigma_t1.max()}, sigma_t2.min;max={sigma_t2.min():.4f};{sigma_t2.max()}, div.min;max={div.min():.4f};{div.max()}")

            d_cur = (x0 - Dx0) / sigma_t0
            # Mix between noisy and denoised frames in the exact per-frame proportion that
            x1 = x0 + (sigma_t1 - sigma_t0) * d_cur  # same as torch.lerp(x0, Dx0, 1 - sigma_t1 / sigma_t0)
            # EDM=> x0 + (t_next - t_hat) * (x0 - Dx0) / t_hat  # t_next ~= sigma_t1, t_hat ~= sigma_t0
            #       x0 + (t_next / t_hat - 1) * (x0 - Dx0)
            #       x0 + (1 - t_next / t_hat) * (Dx0 - x0)
            #   = x0 + (1 - sigma_t1 / sigma_t0) * (Dx0 - x0)
            #   = x0 + (1 - sigma_t1 / sigma_t0) * Dx0 - (1 - sigma_t1 / sigma_t0) * x0
            #   = (1 - sigma_t1 / sigma_t0) * Dx0 + sigma_t1 / sigma_t0 * x0
            #   = Dx0 + sigma_t1 / sigma_t0 * (x0 - Dx0) = torch.lerp(Dx0, x0, sigma_t1 / sigma_t0)
            # NOTE: theory is a bit shaky (unfinished) here, but this way we maintain the correct noise levels explicitly
            # (doing otherwise would likely be disastrous), while following the spirit of the ODE per-frame
            # -- probably can be justified precisely, but TODO
            if heun:
                # Heun correction sub-substep.
                # Denoise at the levels and time offset of the initial step
                Dx1 = denoise(x1, t1, sigmas=sigma_t1, cond=condition, **kwargs)

                # Apply Heun formulas per-frame (like EDM algorithm)
                d_prime = (x1 - Dx1) / sigma_t1
                x1 = x0 + (sigma_t1 - sigma_t0) * (0.5 * d_cur + 0.5 * d_prime)  # corrected x1

            ###################
            # CHURN SUB-STEP
            if churn > 0:  # stochasticity enabled
                # Add a frame-dependent amount of noise that gets us to the ramp corresponding to t2
                noise = S_noise * rnd_churn.randn(x1.shape, dtype=dtype, device=x1.device)
                # noise = S_noise * torch.randn(x1.shape, dtype=dtype, generator=churn_rng, device=x1.device)
                x2 = x1 + torch.sqrt(sigma_t2**2 - sigma_t1**2) * noise
                # self.print(f"curr_frame={curr_frame}\tAdding churn={S_noise}*{torch.sqrt(sigma_t2 ** 2 - sigma_t1 ** 2)[0].squeeze()} noise to x1.")
            else:
                x2 = x1

            ############################
            # YIELD FINISHED FRAMES
            x0_next = x2.clone()
            t0_next = t2

            # If we've crossed a full time interval, remove the resolved frames from left, and add fresh noise to end
            # t0 indicates how far we are inside our window -- if it's >=1, this means there are frame(s) we'll no longer
            # be touching on the left, so we'll yield those and cut them off
            num_finished = int(t0_next)
            if self.hparams.save_debug_sampling_to_file:

                def process_tensor(t):
                    if t is None:
                        return None
                    return t.mean(dim=0).cpu()

                if curr_frame <= 9:
                    log_dict["sigma_t0"].append(process_tensor(sigma_t0))
                    log_dict["sigma_t1"].append(process_tensor(sigma_t1))
                    log_dict["sigma_t2"].append(process_tensor(sigma_t2))
                log_dict["t0"].append(t0)
                log_dict["t1"].append(t1)
                log_dict["t2"].append(t2)
                log_dict["num_finished"].append(num_finished)
                log_dict["curr_step"].append(curr_step)
                log_dict["curr_frame"].append(curr_frame)
                d = (x0 - Dx0) / sigma_t0
                add_full_tensor = False
                if add_full_tensor:
                    log_dict["x0"].append(process_tensor(x0))
                    log_dict["x1"].append(process_tensor(x1))
                    log_dict["x2"].append(process_tensor(x2))
                    log_dict["Dx0"].append(process_tensor(Dx0))
                    log_dict["d"].append(process_tensor(d))
                log_dict["x0_norm_l2"].append(torch.linalg.norm(x0).item())
                log_dict["x1_norm_l2"].append(torch.linalg.norm(x1).item())
                log_dict["x2_norm_l2"].append(torch.linalg.norm(x2).item())
                log_dict["Dx0_norm_l2"].append(torch.linalg.norm(Dx0).item())
                log_dict["d_norm_l2"].append(torch.linalg.norm(d).item())
                # Log L1 and inf norms
                log_dict["x0_squared_mean"].append((x0**2).mean().item())
                log_dict["x1_squared_mean"].append((x1**2).mean().item())
                log_dict["x2_squared_mean"].append((x2**2).mean().item())
                log_dict["Dx0_squared_mean"].append((Dx0**2).mean().item())
                log_dict["d_squared_mean"].append((d**2).mean().item())
                log_dict["x0_abs_mean"].append(x0.abs().mean().item())
                log_dict["x1_abs_mean"].append(x1.abs().mean().item())
                log_dict["x2_abs_mean"].append(x2.abs().mean().item())
                log_dict["Dx0_abs_mean"].append(Dx0.abs().mean().item())
                log_dict["d_abs_mean"].append(d.abs().mean().item())
                log_dict["x0_max"].append(x0.max().item())
                log_dict["x1_max"].append(x1.max().item())
                log_dict["x2_max"].append(x2.max().item())
                log_dict["Dx0_max"].append(Dx0.max().item())
                log_dict["d_max"].append(d.max().item())
                log_dict["x0_min"].append(x0.min().item())
                log_dict["x1_min"].append(x1.min().item())
                log_dict["x2_min"].append(x2.min().item())
                log_dict["Dx0_min"].append(Dx0.min().item())
                log_dict["d_min"].append(d.min().item())
                log_dict["x0_mean"].append(x0.mean().item())
                log_dict["x1_mean"].append(x1.mean().item())
                log_dict["x2_mean"].append(x2.mean().item())
                log_dict["Dx0_mean"].append(Dx0.mean().item())
                log_dict["d_mean"].append(d.mean().item())
                log_dict["x0_std"].append(x0.std().item())
                log_dict["x1_std"].append(x1.std().item())
                log_dict["x2_std"].append(x2.std().item())
                log_dict["Dx0_std"].append(Dx0.std().item())
                log_dict["d_std"].append(d.std().item())
                log_dict["x0_median"].append(x0.median().item())
                log_dict["x1_median"].append(x1.median().item())
                log_dict["x2_median"].append(x2.median().item())
                log_dict["Dx0_median"].append(Dx0.median().item())
                log_dict["d_median"].append(d.median().item())

            if num_finished > 0:
                # Extract the finished frames (use the denoised versions if yield_denoised is set)
                x_finished = select_range(
                    Dx0 if yield_denoised else x0_next, dim=net.time_dim, start=None, end=num_finished
                )
                if condition is not None:
                    # x_finished=[10, 69, 1, 240, 121], condition=[10, 69, 8, 240, 121], num_finished=1 net.time_dim=2
                    # only support for one frame condition
                    new_conds = select_range(x_finished, dim=net.time_dim, start=num_finished - 1, end=num_finished)
                    assert net.time_dim == 2, "Only time_dim=2 supported for condition"
                    if condition.shape[2] == 1:
                        condition = new_conds
                    else:
                        # Discard the first conditional frame and add the newly denoised frame for the next iteration
                        condition = torch.cat((condition[:, :, 1:], new_conds), dim=2)

                # return a "video" of all the finished frames with time dimension intact (there might be more than one if
                # step > 1 -- this is much more efficient than yielding each one separately, e.g. for audio)
                # Also output some extra tensors we might be interested in visualizing.
                try:
                    yield dict(
                        frames=x_finished,
                        noisy_future=x0,  # the full window of frames with noise
                        denoised_future=Dx0,  # The denoised version of the above
                    )
                except GeneratorExit:
                    # If the caller wants to end video generation, they'll call .close() on this generator, and we'll quit
                    # Before, save the current state in the log_dict to disk
                    if self.hparams.save_debug_sampling_to_file:
                        if "WANDB_ID" in os.environ:
                            prefix = f"run-{os.environ['WANDB_ID']}-"
                        else:
                            prefix = ""
                        params_str = f"step{step}-churn{churn}-S_noise{S_noise}-S_ar_noise{S_ar_noise}-heun{heun}-denoiser_clip{denoiser_clip}-yield_denoised{yield_denoised}"
                        params_str = prefix + params_str
                        file_name = f"results/debuglog_dict_{params_str}.pth"
                        abs_file_name = os.path.abspath(file_name)
                        os.makedirs(os.path.dirname(abs_file_name), exist_ok=True)
                        log.info(f"Saving log dict to {abs_file_name}")
                        torch.save(log_dict, abs_file_name)
                        log.info(f"Log dict saved to {abs_file_name}")
                        # sleep for a while to ensure the file is saved before exiting
                        time.sleep(50)
                    return

                # The caller got their frames, we no longer need them -- clip them off
                x0_next = select_range(x0_next, dim=net.time_dim, start=num_finished)
                # update the offset into the window of frames to match dropping of num_finished
                t0_next -= num_finished

                # Our window has now become shorter by num_finished -- draw new noise to the right (with correct
                # noise levels) to bring it to back to proper length
                # NOTE implicitly we've padded the window with black frames with the maximal noise levels
                # ... might still be worth validating some choices
                sigma_fresh = select_range(net.noise_coeff(x0, offset=t0_next), dim=net.time_dim, start=-num_finished)
                # self.print(f"sigma_fresh.min;max={sigma_fresh.min():.4f};{sigma_fresh.max()}")
                # Get a block of appropriately shaped white
                xs = x0_next.shape
                fresh_noise = self.sample_additional_noise_frame(
                        [num_finished if dim == net.time_dim else xs[dim] for dim in range(len(xs))],
                        dtype=dtype,
                        device=x0_next.device,
                    rnd=rnd
                )
                fresh_noise = sigma_fresh * S_ar_noise * fresh_noise  # shape: [B, C, num_finished, H, W]
                if self.hparams.force_pure_noise_last_frame == "plus_data":
                    fresh_noise = x_finished + fresh_noise
                # Cat it onto the frame window, scaled by the sigmas found above
                x0_next = torch.cat((x0_next, fresh_noise), net.time_dim)

                curr_frame += num_finished
                self.curr_frame = curr_frame
            else:
                sigma_fresh = None

            if self.hparams.save_debug_sampling_to_file:
                log_dict["sigma_fresh"].append(sigma_fresh)
                # if add_full_tensor:
                # log_dict["x0_next"].append(x0_next)
                # log_dict["t0_next"].append(t0_next)

            # Adopt the updated window and time offset for the next iteration
            t0 = t0_next
            x0 = x0_next

            curr_step += 1



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


# return a slice object the interval start:end of dimension dim
def slice_range(dim, of, start, end=None):
    # slice_range simply returns a list of slices of len `of` with the `dim`-th element being `slice(start, end)`
    return [slice(start, end) if d == dim else slice(None) for d in range(of)]


# use the above slice object to extract a range from tensor z
def select_range(z, dim, start, end=None):
    # sl = [slice(start, end) if d == dim else slice(None) for d in range(len(z.shape))]
    sl = slice_range(dim=dim, of=len(z.shape), start=start, end=end)
    return z[sl]


# We'll often need to separate the integer and fractional part of a float:
def frac(t):
    return t % 1


# Noise schedules
def exp_a_b(s, a: float, b: float):
    return 1 - torch.exp(-a * s**b)


def exp_a_b_normalized(s, a: float, b: float, tmin: float, tmax: float, noises_min: float, noises_max: float):
    noises_raw = exp_a_b(s, a, b)
    noises_normalized = (noises_raw - noises_min) / (noises_max - noises_min) * (tmax - tmin) + tmin
    return noises_normalized


#             noises_min = float(exp_a_b(torch.tensor(0.0), a=a, b=b))
# noises_max = float(exp_a_b(torch.tensor(1.0), a=a, b=b))


@torch.inference_mode()
def sample_simple(self, prompt, dtype=torch.float32, batch_seeds=None, **kwargs):  # Salva: dtype=torch.float64!
    batch_seeds = batch_seeds or torch.randint(0, 2**32, (prompt.shape[0],), device=prompt.device)
    rnd = StackedRandomGenerator(self.device, batch_seeds)
    rnd_churn = StackedRandomGenerator(self.device, batch_seeds + 123) if self.hparams.S_churn > 0 else None
    net = self
    step = self.hparams.step
    churn = self.hparams.S_churn
    t0 = 0.0  # start timing from 0.0
    step_ode = step / (1 - churn)  # length of the step we'll take forward following the ODE
    S_ar_noise = self.hparams.S_ar_noise
    S_noise = self.hparams.S_noise
    yield_denoised = self.hparams.yield_denoised
    padding = int(step_ode) + 1

    def win(x, i):
        nonlocal net  # variable is defined in the outer scope and will be modified inside the function.
        return select_range(x, dim=net.time_dim, start=i, end=i + net.seq_len)

    def pad(x, pre, post, mode="replicate"):
        nonlocal net
        return torch.nn.functional.pad(
            x, (0, 0) * (len(x.shape) - 1 - net.time_dim) + (pre, post) + (0, 0) * (net.time_dim - 2), mode=mode
        )

    # Insert copied frames into the initial prompt window according to the padded sequence length we'll be using
    # - they'll be fully under the noise, so no big deal what the content is (?)
    prompt = pad(prompt, 0, padding)  # pad `padding` frames to the right (i.e. to the future) of the prompt
    per_frame_noise_coeffs = net.noise_coeff(prompt, offset=0.0)
    noise = S_ar_noise * rnd.randn(prompt.shape, dtype=dtype, device=prompt.device)
    x0 = prompt + per_frame_noise_coeffs * noise

    def denoise(x, t, cond=None, **denoise_kwargs):
        Dx = net(win(x, int(t)), offset=frac(t), dtype=dtype, condition=cond, **denoise_kwargs)
        # Extend the denoiser output to cover the entire window, by appropriate replication padding
        Dx = pad(Dx, int(t), padding - int(t))  # for t=0, this pads the right side of the frame
        range_kwargs = dict(dim=net.time_dim, start=None, end=int(t))
        Dx[slice_range(**range_kwargs, of=len(Dx.shape))] = select_range(x, **range_kwargs)
        return Dx

    curr_frame = 0
    while True:
        # By this point, t0 has accumulated all the fractional steps we've taken so far,
        # minus all the full frames we've yielded; it's the float offset into the padded frame window we maintain
        t1 = t0 + step_ode  # the global time we'll land after the ODE substep
        t2 = t0 + step  # ... and after the churn/backtracking (if it's enabled)

        # Collect some noise level tensors which can be used to set a randn() output
        # to the noise ramps with appropriate time offsets
        # These are one-dimensional tensors of length net.seq_len, padded with appropriate dimensions for broadcast
        sigma_t0 = net.noise_coeff(x0, offset=t0)  # noise level at beginning of step
        sigma_t1 = net.noise_coeff(x0, offset=t1)  # ... at end of ODE step but before churn
        sigma_t2 = net.noise_coeff(x0, offset=t2)  # ... and after adding churn (passed to next iter)

        # ODE SUB-STEP
        # Denoised frame window. If you visualize this, you'll see a short video of increasingly blurry frames,
        # reflecting the uncertainty of the future evolution.
        Dx0 = denoise(x0, t0, **kwargs)  # x0,Dx0.shape = [B, C, seq_len+padding, H, W]
        d_cur = (x0 - Dx0) / sigma_t0
        x1 = x0 + (sigma_t1 - sigma_t0) * d_cur  # same as torch.lerp(x0, Dx0, 1 - sigma_t1 / sigma_t0)
        # Heun correction sub-substep.
        Dx1 = denoise(x1, t1, **kwargs)
        dp = (x1 - Dx1) / sigma_t1
        x1 = x0 + (sigma_t1 - sigma_t0) * (d_cur + dp) / 2  # corrected x1

        # CHURN SUB-STEP
        if churn > 0:  # stochasticity enabled
            # Add a frame-dependent amount of noise that gets us to the ramp corresponding to t2
            noise = S_noise * rnd_churn.randn(x1.shape, dtype=dtype, device=x1.device)
            x2 = x1 + torch.sqrt(sigma_t2**2 - sigma_t1**2) * noise
        else:
            x2 = x1

        # YIELD FINISHED FRAMES
        x0_next = x2.clone()
        t0_next = t2

        # If we've crossed a full time interval, remove the resolved frames from left, and add fresh noise to end
        # t0 indicates how far we are inside our window -- if it's >=1, this means there are frame(s) we'll no longer
        # be touching on the left, so we'll yield those and cut them off
        num_finished = int(t0_next)
        if num_finished > 0:
            # Extract the finished frames (use the denoised versions if yield_denoised is set)
            x_finished = select_range(
                Dx0 if yield_denoised else x0_next, dim=net.time_dim, start=None, end=num_finished
            )
            # return a "video" of all the finished frames with time dimension intact (there might be more than one if
            # step > 1 -- this is much more efficient than yielding each one separately, e.g. for audio)
            # Also output some extra tensors we might be interested in visualizing.
            try:
                yield dict(
                    frames=x_finished,
                    noisy_future=x0,  # the full window of frames with noise
                    denoised_future=Dx0,  # The denoised version of the above
                )
            except GeneratorExit:
                return

            # The caller got their frames, we no longer need them -- clip them off
            x0_next = select_range(x0_next, dim=net.time_dim, start=num_finished)
            t0_next -= num_finished  # update the offset to the window of frames to match dropping of num_finished

            # Our window has now become shorter by num_finished -- draw new noise to the right (with correct
            # noise levels) to bring it to back to proper length
            # NOTE implicitly we've padded the window with black frames with the maximal noise levels
            # ... might still be worth validating some choices
            sigma_fresh = select_range(net.noise_coeff(x0, offset=t0_next), dim=net.time_dim, start=-num_finished)
            # Get a block of appropriately shaped white
            xs = x0_next.shape
            fresh_noise = S_ar_noise * rnd.randn(
                [num_finished if dim == net.time_dim else xs[dim] for dim in range(len(xs))],
                dtype=dtype,
                device=x0_next.device,
            )  # shape: [B, C, num_finished, H, W]
            # Cat it onto the frame window, scaled by the sigmas found above
            x0_next = torch.cat((x0_next, sigma_fresh * fresh_noise), net.time_dim)

            curr_frame += num_finished
            self.curr_frame = curr_frame

        # Adopt the updated window and time offset for the next iteration
        t0 = t0_next
        x0 = x0_next

def calculate_lognormal_weight(sigma, P_mean = 0.5, P_std = 1.2):
    """
    Calculate a weight proportional to the frequency of a noise level sigma
    under a lognormal distribution.

    Parameters:
    -----------
    sigma : float or tensor
        The noise level to calculate weight for
    P_mean : float
        Mean parameter of the normal distribution before exponentiation
    P_std : float
        Standard deviation parameter of the normal distribution before exponentiation

    Returns:
    --------
    weight : float or tensor
        Weight proportional to the PDF of the lognormal distribution at sigma
    """
    # Calculate PDF of normal distribution in log space
    # f(x) = (1/(σ√(2π))) * exp(-(x-μ)²/(2σ²))
    exponent = -((torch.log(sigma) - P_mean) ** 2) / (2 * P_std ** 2)
    coefficient = 1 / (P_std * np.sqrt(2 * np.pi))

    # PDF of normal distribution in log space
    normal_pdf = coefficient * torch.exp(exponent)

    # Convert to lognormal PDF
    # For lognormal, we need to divide by sigma
    lognormal_pdf = normal_pdf / sigma

    # The weight is directly proportional to the PDF value
    weight = lognormal_pdf
    return weight

