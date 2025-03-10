from __future__ import annotations

from abc import abstractmethod
from contextlib import ExitStack
from typing import Any, Dict, List, Optional, Sequence, Union

import hydra
import torch
from omegaconf import DictConfig
from torch import Tensor, nn
from tqdm.auto import tqdm

from src.diffusion._base_diffusion import BaseDiffusion
from src.experiment_types.interpolation import NextStepInterpolationExperiment
from src.models._base_model import BaseModel
from src.utilities.checkpointing import get_checkpoint_from_path_or_wandb
from src.utilities.random_control import controlled_rng
from src.utilities.utils import freeze_model, raise_error_if_invalid_value


class BaseDYffusion2(BaseDiffusion):
    def __init__(
        self,
        forward_conditioning: str = "data",
        schedule: str = "linear",
        additional_interpolation_steps: int = 0,
        additional_interpolation_steps_factor: int = 0,
        interpolate_before_t1: bool = False,
        prediction_mode: str = "raw",
        sampling_type: str = "cold",  # 'cold' or 'naive'
        sampling_schedule: Union[List[float], str] = None,
        use_cold_sampling_for_intermediate_steps: bool = False,
        use_cold_sampling_for_last_step: bool = True,
        use_cold_sampling_for_init_of_ar_step: Optional[bool] = None,
        refine_intermediate_predictions: bool = False,
        time_encoding: str = "discrete",
        refine_predictions: str | bool = False,
        refinement_rounds: int = 0,
        prediction_timesteps: Optional[Sequence[float]] = None,
        enable_interpolator_dropout: Union[bool, str] = True,
        enable_predict_last_dropout: bool | str = False,
        use_same_dropout_state_for_sampling: bool = False,
        log_every_t: Union[str, int] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs, sampling_schedule=sampling_schedule)
        sampling_schedule = None if sampling_schedule == "None" else sampling_schedule
        self.save_hyperparameters(ignore=["model"])
        self.num_timesteps = self.hparams.timesteps
        self.SUPPORT_RESIDUAL_PREDICTION = False
        self.prediction_mode = prediction_mode

        fcond_options = [
            "data",
            "none",
            "noise",
            "data|noise",
            "data+noise-v1",
            "data+noise-v2",
            "t0noise",
        ]
        raise_error_if_invalid_value(forward_conditioning, fcond_options, "forward_conditioning")

        refine_predictions_options = [False, "all", "intermediate"]
        raise_error_if_invalid_value(refine_predictions, refine_predictions_options, "refine_predictions")
        if not refine_predictions:
            assert refinement_rounds == 0, "refinement_rounds must be 0 if refine_predictions is False"

        # Add additional interpolation steps to the diffusion steps
        # we substract 2 because we don't_i_next want to use the interpolator in timesteps outside [1, num_timesteps-1]
        horizon = self.num_timesteps  # = self.interpolator_horizon
        assert horizon > 1, f"horizon must be > 1, but got {horizon}. Please use datamodule.horizon with > 1"
        if schedule == "linear":
            assert (
                additional_interpolation_steps == 0
            ), "additional_interpolation_steps must be 0 when using linear schedule"
            self.additional_interpolation_steps_fac = additional_interpolation_steps_factor
            if interpolate_before_t1:
                interpolated_steps = horizon - 1
                self.di_to_ti_add = 0
            else:
                interpolated_steps = horizon - 2
                self.di_to_ti_add = additional_interpolation_steps_factor

            self.additional_diffusion_steps = additional_interpolation_steps_factor * interpolated_steps
        elif schedule == "before_t1_only":
            assert (
                additional_interpolation_steps_factor == 0
            ), "additional_interpolation_steps_factor must be 0 when using before_t1_only schedule"
            assert interpolate_before_t1, "interpolate_before_t1 must be True when using before_t1_only schedule"
            self.additional_diffusion_steps = additional_interpolation_steps
        elif schedule == "before_t1_then_linear":
            assert (
                interpolate_before_t1
            ), "interpolate_before_t1 must be True when using before_t1_then_linear schedule"
            self.additional_interpolation_steps_fac = additional_interpolation_steps_factor
            self.additional_diffusion_steps_pre_t1 = additional_interpolation_steps
            self.additional_diffusion_steps = (
                additional_interpolation_steps + additional_interpolation_steps_factor * (horizon - 2)
            )
        else:
            raise ValueError(f"Invalid schedule: {schedule}")

        self.num_timesteps += self.additional_diffusion_steps
        self.enable_interpolator_dropout = enable_interpolator_dropout
        raise_error_if_invalid_value(
            enable_interpolator_dropout,
            [True, False, "always", "except_dynamical_steps"],
            "enable_interpolator_dropout",
        )
        if enable_predict_last_dropout:
            self.log_text.info("Enabling dropout in forward prediction of last timestep during inference.")
        if use_cold_sampling_for_last_step:
            self.log_text.info("Using cold sampling for the last step.")

        # which diffusion steps to take during sampling
        self.full_sampling_schedule = list(range(0, self.num_timesteps))
        self.sampling_schedule = sampling_schedule or self.full_sampling_schedule

    @property
    def diffusion_steps(self) -> List[int]:
        return list(range(0, self.num_timesteps))

    @property
    def valid_time_range_for_backbone_model(self) -> List[int]:
        valid_time = list(range(1, self.num_timesteps + 1))
        return valid_time

    @property
    def time_subsampling(self) -> int:
        return self.datamodule_config.get("time_subsampling", 1) if self.datamodule_config is not None else 1

    @property
    def sampling_schedule(self) -> List[Union[int, float]]:
        return self._sampling_schedule

    @sampling_schedule.setter
    def sampling_schedule(self, schedule: Union[str, List[Union[int, float]]]):
        """Set the sampling schedule. At the very minimum, the sampling schedule will go through all dynamical steps.
        Notation:
        - N: number of diffusion steps
        - h: number of dynamical steps
        - h_0: first dynamical step

        Options for diffusion sampling schedule trajectories ('<name>': <description>):
        - 'only_dynamics': the diffusion steps corresponding to dynamical steps (this is the minimum)
        - 'only_dynamics_plus_randomFLOAT': add each non-dynamical step with probability FLOAT to the schedule
        - 'only_dynamics_plus_randomINT': add INT random non-dynamical steps to the schedule
        - 'only_dynamics_plus_discreteINT': add INT discrete non-dynamical steps, uniformly drawn between 0 and h_0
        - 'only_dynamics_plusINT': add INT non-dynamical steps (possibly continuous), uniformly drawn between 0 and h_0
        - 'everyINT': only use every INT-th diffusion step (e.g. 'every2' for every second diffusion step)
        - 'firstINT': only use the first INT diffusion steps
        - 'firstFLOAT': only use the first FLOAT*N diffusion steps

        """
        schedule_name = schedule
        assert (
            1 <= schedule[-1] <= self.num_timesteps
        ), f"Invalid sampling schedule: {schedule}, must end with number/float <= {self.num_timesteps}"
        if schedule[0] != 0:
            self.log_text.warning(
                f"Sampling schedule {schedule_name} must start at 0. Adding 0 to the beginning of it."
            )
            schedule = [0] + schedule

        last = schedule[-1]
        if last != self.num_timesteps - 1:
            self.log_text.warning("------" * 20)
            self.log_text.warning(
                f"Are you sure you don't_i_next want to sample at the last timestep? (current last timestep: {last})"
            )
            self.log_text.warning("------" * 20)

        # check that schedule is monotonically increasing
        for i in range(1, len(schedule)):
            assert schedule[i] > schedule[i - 1], f"Invalid sampling schedule not monotonically increasing: {schedule}"

        if all(float(s).is_integer() for s in schedule):
            schedule = [int(s) for s in schedule]
        else:
            self.log_text.info(f"Sampling schedule {schedule_name} uses diffusion steps it has not been trained on!")
        self._sampling_schedule = schedule

    @abstractmethod
    def get_condition(
        self,
        prediction_type: str,
        x_th: Optional[Tensor] = None,
        shape: Sequence[int] = None,
        concat_dim: int = 1,
    ) -> Tensor:
        pass

    def get_time_encoding_forecast(self, time: Tensor) -> Tensor:
        """Given input time, return the encoding for the forecast step. E.g. lead time = T - input_time."""
        return self.num_timesteps - time

    @abstractmethod
    def get_time_encoding_interpolation(self, time: Tensor) -> Tensor:
        pass

    @abstractmethod
    def _interpolate(
        self,
        x_ti: Tensor,
        x_th: Tensor,
        time: Tensor,
        **kwargs,
    ):
        raise NotImplementedError("_interpolate method must be implemented")

    def q_sample(self, *args, random_mode: str = "random", iteration: int = 0, **kwargs):
        random_mode = "random" if random_mode is False else random_mode  # legacy support
        with controlled_rng(random_mode, iteration=iteration):
            return self._q_sample(*args, **kwargs)

    def _q_sample(
        self,
        x_ti,
        x_th,
        time_of_input: Tensor,
        batch_mask: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        # q_sample = using model in interpolation mode
        # just remember that x_ti here refers to t_i_next=0 (the initial conditions)
        # and x_0 (terminology of diffusion models) refers to t_i_next=T, i.e. the last timestep
        if not torch.is_tensor(time_of_input):
            time_of_input = torch.full((x_ti.shape[0],), time_of_input, dtype=self.dtype, device=self.device)
        time = self.get_time_encoding_interpolation(time_of_input)
        if batch_mask is not None:
            x_ti = x_ti[batch_mask]
            x_th = x_th[batch_mask]
            time = time[batch_mask]
            for k, v in kwargs.items():
                if torch.is_tensor(v):
                    kwargs[k] = v[batch_mask]

        do_enable = self.training or self.enable_interpolator_dropout in [True, "always"]

        ipol_handles = [self.interpolator] if hasattr(self, "interpolator") else [self]

        with ExitStack() as stack:
            # inference_dropout_scope of all handles (enable and disable) is managed by the ExitStack
            for ipol in ipol_handles:
                stack.enter_context(ipol.inference_dropout_scope(condition=do_enable))

            x_ti = self._interpolate(x_ti=x_ti, x_th=x_th, time=time, **kwargs)
        return x_ti

    def predict_x_last(
        self,
        x_t: Tensor,
        time_of_input: Union[Tensor, float],
        **kwargs,
    ):
        if not torch.is_tensor(time_of_input):
            time_of_input = torch.full((x_t.shape[0],), time_of_input, dtype=self.dtype, device=self.device)
        assert (0 <= time_of_input).all() and (
            time_of_input < self.num_timesteps
        ).all(), f"Invalid lead time timestep: {time_of_input}"
        time_of_input = self.get_time_encoding_forecast(time_of_input)
        # predict_x_last = using model in forward mode to predict for lead time = t \in (0, horizon]
        forward_cond = self.get_condition(
            prediction_type="forward",
            shape=x_t.shape,
        )
        dropout_condition = self.training
        if not self.training:
            # If sampling, we may enable Monte Carlo dropout
            if self.hparams.enable_predict_last_dropout == "except_last_step":
                dropout_condition = bool((time_of_input < self.num_timesteps - 1).all())
            elif self.hparams.enable_predict_last_dropout:
                dropout_condition = True

        # enable dropout for prediction of last dynamics
        with self.model.inference_dropout_scope(condition=dropout_condition):
            x_last_pred = self._predict_last_dynamics(x_t=x_t, forward_condition=forward_cond, t=time_of_input, **kwargs)
        return x_last_pred

    def _get_loss_callable_from_name_or_config(self, loss_function: str, **kwargs):
        if loss_function == "l1_l2_mix":
            self.log_text.info("=========> Using L1-L2 mix loss to penalize different timesteps differently.")
            return self.l1_l2_mix_loss
        loss_func = super()._get_loss_callable_from_name_or_config(loss_function, **kwargs)

        def loss_func_wrapper(preds, targets, time_of_input=None):
            return loss_func(preds, targets)

        return loss_func_wrapper

    def l1_l2_mix_loss(self, preds: Tensor, targets: Tensor, time_of_input: Tensor) -> Tensor:
        l1_loss = torch.nn.functional.l1_loss(preds, targets, reduction="none")
        l2_loss = torch.nn.functional.mse_loss(preds, targets, reduction="none")
        coeff = time_of_input / (self.num_timesteps - 1)
        # extend coeff to the same shape as l1_loss and l2_loss by broadcasting 1-dimensions
        coeff = coeff.reshape(coeff.shape[0], *([1] * (len(l1_loss.shape) - 1)))
        assert (0 <= coeff).all() and (coeff <= 1).all(), f"Invalid coefficient: {coeff}"
        loss = (1 - coeff) * l1_loss + coeff * l2_loss
        return loss.mean()

    @abstractmethod
    def _predict_last_dynamics(self, x_t: Tensor, forward_condition: Tensor, t: Tensor, **kwargs) -> Tensor:
        pass

    def sample_loop(
        self,
        initial_condition,
        log_every_t: Optional[Union[str, int]] = None,
        num_predictions: int = None,
        x0_ref=None,
        verbose=True,
        **kwargs,
    ):
        # assert num_predictions is None or num_predictions == 1, "num_predictions must not be provided"
        sampling_schedule = self.sampling_schedule

        if len(initial_condition.shape) == 5 and initial_condition.shape[1] == 1:
            initial_condition = initial_condition.squeeze(1)
        else:
            assert len(initial_condition.shape) == 4, f"x_ti.shape: {initial_condition.shape} (should be 4D)"
        intermediates, xhat_th = dict(), None
        last_s_plus_one = sampling_schedule[-1] + 1
        is_cold_sampling = self.hparams.sampling_type in ["cold", "heun1", "heun2", "heun3"]
        for sampling_round in range(0, self.hparams.refinement_rounds + 1):
            desc = f"Refinement round {sampling_round}" if sampling_round > 0 else "Sampling"
            s_and_snext = zip(
                sampling_schedule,
                sampling_schedule[1:] + [last_s_plus_one],
                sampling_schedule[2:] + [last_s_plus_one, last_s_plus_one + 1],
            )
            progress_bar = tqdm(s_and_snext, desc=desc, total=len(sampling_schedule), leave=False)
            x_s = initial_condition
            x_s_prev = None
            for sampling_step, (s, s_next, s_nnext) in enumerate(progress_bar):
                is_first_step = s == 0
                is_last_step = s == self.num_timesteps - 1

                if sampling_round == 0 or (self.hparams.refine_predictions == "all" and not is_first_step):
                    # Forecast x_{t+h} using x_{s} as input
                    xhat_th = self.predict_x_last(x_t=x_s, time_of_input=s, **kwargs)
                else:
                    xhat_th = intermediates[f"t{self.num_timesteps}_preds"]

                # Interpolate x_{s+1} using x_{s} and x_{t+h} as input
                q_sample_kwargs = dict(
                    x_th=xhat_th,
                    num_predictions=num_predictions if is_first_step else 1,
                    random_mode=self.hparams.use_same_dropout_state_for_sampling,
                    iteration=sampling_step,
                    **kwargs,
                )
                if s_next <= self.num_timesteps - 1:
                    x_interpolated_s_next = self.q_sample(x_ti=x_s, time_of_input=s, **q_sample_kwargs)
                else:
                    x_interpolated_s_next = xhat_th  # for the last step, we use the final x0_hat prediction

                if self.hparams.sampling_type == "naive" or (
                    is_cold_sampling
                    and ((not self.hparams.use_cold_sampling_for_last_step and is_last_step) or is_first_step)
                ):
                    if is_last_step and is_cold_sampling and self.hparams.use_cold_sampling_for_init_of_ar_step:
                        x_interpolated_s = self.q_sample(x_ti=x_s_prev, time_of_input=s - 1, **q_sample_kwargs)
                        intermediates["preds_autoregressive_init"] = x_s - x_interpolated_s + xhat_th
                    x_s_prev = x_s
                    x_s = x_interpolated_s_next
                elif is_cold_sampling:
                    # Interpolate x_{s} using x_{s-1} and x_{t+h} as input
                    x_interpolated_s = self.q_sample(x_ti=x_s_prev, time_of_input=s - 1, **q_sample_kwargs)
                    # for s = 0, we have x_s_degraded = x_s, so we just directly return x_interpolated_s_next
                    x_s_prev = x_s
                    d_i1 = x_interpolated_s_next - x_interpolated_s  # x_s - x_interpolated_s
                    if (
                        self.hparams.sampling_type == "cold"
                        or s_nnext > self.num_timesteps - 1
                        or ("heun" in self.hparams.sampling_type and is_first_step)
                    ):
                        x_s = x_s + d_i1  # x_s += x_interpolated_s_next - x_interpolated_s
                    else:
                        assert "heun" in self.hparams.sampling_type
                        # Heun's method
                        xs_tmp = x_s + d_i1
                        x0_hat2 = self.predict_x_last(x_t=xs_tmp, time_of_input=s_next, **kwargs)
                        q_sample_kwargs["x_th"] = x0_hat2
                        x_interpolated_s_nnext = self.q_sample(x_ti=xs_tmp, time_of_input=s_next, **q_sample_kwargs)
                        if self.hparams.sampling_type == "heun1":
                            d_i2 = x_interpolated_s_nnext - xs_tmp  # Seems like the best option
                        elif self.hparams.sampling_type == "heun2":
                            x_interpolated_s_next2 = self.q_sample(
                                x_ti=xs_tmp, time_of_input=s_next, **q_sample_kwargs
                            )
                            d_i2 = x_interpolated_s_nnext - x_interpolated_s_next2
                        elif self.hparams.sampling_type == "heun3":
                            d_i2 = x_interpolated_s_nnext - x_interpolated_s_next
                        x_s = x_s + 0.5 * (d_i1 + d_i2)

                else:
                    raise ValueError(f"unknown sampling type {self.hparams.sampling_type}")

                if self.hparams.use_cold_sampling_for_intermediate_steps or is_last_step:
                    preds_t = x_s
                else:
                    assert not self.hparams.use_cold_sampling_for_intermediate_steps and not is_last_step
                    preds_t = x_interpolated_s_next
                intermediates[f"t{s_next}_preds"] = preds_t
        if self.hparams.refine_intermediate_predictions:
            # Use last prediction of x0 for final prediction of intermediate steps (not the last timestep!)
            q_sample_kwargs["x_th"] = xhat_th
            q_sample_kwargs["random_mode"] = "fixed_global"  #  use the same dropout mask for all steps
            _ = q_sample_kwargs.pop("iteration", None)
            x_s_prev = initial_condition
            for s in range(1, self.num_timesteps):
                x_interpolated_s = self.q_sample(x_ti=x_s_prev, time_of_input=s - 1, **q_sample_kwargs)
                intermediates[f"t{s}_preds"] = x_s_prev = x_interpolated_s

        if last_s_plus_one < self.num_timesteps:
            return x_s, intermediates
        return xhat_th, intermediates

    @torch.inference_mode()
    def sample(self, initial_condition, num_samples=1, **kwargs):
        _, intermediates = self.sample_loop(initial_condition, **kwargs)
        return intermediates


class DYffusionMarkov(BaseDYffusion2):
    def __init__(
        self,
        interpolator: Optional[nn.Module] = None,
        interpolator_run_id: Optional[str] = None,
        interpolator_local_checkpoint_path: Optional[Union[str, bool]] = True,  # if true, search in local path
        interpolator_wandb_ckpt_filename: Optional[str] = None,
        interpolator_overrides: Optional[List[str]] = None,  # a dot list, e.g. ["model.hidden_dims=128"]
        interpolator_wandb_kwargs: Optional[Dict[str, Any]] = None,
        interpolator_use_ema: bool = False,
        lambda_reconstruction: float = 1.0,
        lambda_reconstruction2: float = 0.0,
        reconstruction2_detach_x_last: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters(ignore=["model", "interpolator"])
        # Load interpolator and its weights
        interpolator_wandb_kwargs = interpolator_wandb_kwargs or {}
        interpolator_wandb_kwargs["epoch"] = interpolator_wandb_kwargs.get("epoch", "best")
        if interpolator_wandb_ckpt_filename is not None:
            assert interpolator_wandb_kwargs.get("ckpt_filename") is None, "ckpt_filename already set"
            interpolator_wandb_kwargs["ckpt_filename"] = interpolator_wandb_ckpt_filename
        interpolator_overrides = list(interpolator_overrides) if interpolator_overrides is not None else []
        interpolator_overrides.append("model.verbose=False")
        self.interpolator: NextStepInterpolationExperiment = get_checkpoint_from_path_or_wandb(
            interpolator,
            model_checkpoint_path=interpolator_local_checkpoint_path,
            wandb_run_id=interpolator_run_id,
            reload_kwargs=interpolator_wandb_kwargs,
            model_overrides=interpolator_overrides,
        )
        assert isinstance(
            self.interpolator, NextStepInterpolationExperiment
        ), "interpolator must be a NextStepInterpolationExperiment"
        # freeze the interpolator (and set to eval mode)
        freeze_model(self.interpolator)
        self.interpolator_window = self.interpolator.window
        self.interpolator_horizon = self.interpolator.true_horizon
        assert self.interpolator_horizon == self.num_timesteps, "interpolator horizon must match num_timesteps"

        self._max_ipol_time = self.num_timesteps - 1

    def get_condition(
        self,
        prediction_type: str,
        x_th: Optional[Tensor] = None,
        shape: Sequence[int] = None,
        concat_dim: int = 1,
    ) -> Tensor:
        cond = None
        if prediction_type == "interpolate":
            raise NotImplementedError("get_condition for interpolation not implemented")
        elif prediction_type == "forward":
            assert shape is not None, "shape must be provided for forward prediction"
            cond = None
        else:
            raise ValueError(f"Unknown prediction type {prediction_type}")
        return cond

    def get_time_encoding_interpolation(self, time: Tensor) -> Tensor:
        return time + 1  # use target time as input to the interpolator

    def _interpolate(
        self,
        x_ti: Tensor,
        x_th: Tensor,
        time: Tensor,
        num_predictions: int = 1,
        **kwargs,
    ) -> Tensor:
        interpolator_inputs = torch.cat([x_ti, x_th], dim=1)
        with torch.inference_mode():
            with self.interpolator.ema_scope(condition=self.hparams.interpolator_use_ema):
                x_ipolated = self.interpolator.predict_packed(
                    interpolator_inputs, time=time, **kwargs
                )
        return x_ipolated["preds"]

    def _predict_last_dynamics(self, x_t: Tensor, forward_condition: Tensor, t: Tensor, **kwargs) -> Tensor:
        assert (0 < t).all() and (t <= self.num_timesteps).all(), f"Invalid lead time timestep: {t}"
        x_last_pred = self.model.predict_forward(x_t, time=t, condition=forward_condition, **kwargs)
        return x_last_pred

    def p_losses(
        self,
        dynamics: Tensor,
        verbose=False,
        **kwargs
    ):
        B = dynamics.shape[0]
        assert (
            dynamics.shape[1] == self.num_timesteps + 1
        ), f"dynamics.shape[1] != num_timesteps: {dynamics.shape}[1] != {self.num_timesteps + 1}"
        criterion = self.criterion["preds"]
        x_th = dynamics[:, -1, ...]  # x_h
        # Sample input time step
        t_abs = torch.randint(0, self.num_timesteps, (B,), device=self.device, dtype=self.dtype)
        t_nonzero_mask = t_abs > 0
        t_not_last_mask = t_abs < self.num_timesteps - 1
        t_abs_nonzero = t_abs[t_nonzero_mask].long()
        t_abs_not_last = t_abs[t_not_last_mask].long()

        lam1 = self.hparams.lambda_reconstruction
        lam2 = self.hparams.lambda_reconstruction2

        xtilde_t = dynamics[:, 0, ...].clone()  # .to(xhat_ti.dtype)
        # since we do not need to interpolate xt_0, we can skip all batches where t=0
        if t_nonzero_mask.any():
            # Interpolate x_{t} from x_{t-1} and x_{h} for 1 <= t <= h-1
            xhat_ti = self._q_sample(
                x_ti=dynamics[t_nonzero_mask, t_abs_nonzero - 1, ...],
                x_th=x_th[t_nonzero_mask],
                time_of_input=t_abs_nonzero - 1,
                **{k: v[t_nonzero_mask] if torch.is_tensor(v) else v for k, v in kwargs.items()},
            )

            # Now, simply concatenate the inital_conditions for t=0 with the interpolated data for t>0
            xtilde_t[t_nonzero_mask, ...] = xhat_ti.to(xtilde_t.dtype)
            # assert torch.all(x_ti[t_i_next == 0] == x_ti[t_i_next == 0]), 'x_ti[t_i_next == 0] != x_ti[t_i_next == 0]'

        # Forecast x_{h} from \tilde{x}_{t} for 0 <= t <= h-1
        xhat_th = self.predict_x_last(x_t=xtilde_t, time_of_input=t_abs, **kwargs)
        loss_forward = criterion(xhat_th, x_th, time_of_input=t_abs)

        if lam2 > 0 and t_not_last_mask.any():
            # Interpolate x_{t+1} from \tilde{x}_{t} and \hat{x}_{h} for 1 <= t <= h-1
            if self.hparams.reconstruction2_detach_x_last:
                xhat_th = xhat_th.detach()

            xhat_ti_next = self._q_sample(
                x_ti=xtilde_t,
                x_th=xhat_th,
                time_of_input=t_abs,
                batch_mask=t_not_last_mask,
                **kwargs
            )  # \hat{x}_{t+1}
            # simulate one more step of the reverse diffusion process, i.e.
            # forecast x_{h} from \hat{x}_{t'} for 1 <= t' <= h-1, where t' = t+1
            kwargs = {k: v[t_not_last_mask] if torch.is_tensor(v) else v for k, v in kwargs.items()}
            xhat_th2 = self.predict_x_last(xhat_ti_next, time_of_input=t_abs_not_last + 1, **kwargs)
            loss_forward2 = criterion(xhat_th2, x_th[t_not_last_mask], time_of_input=t_abs_not_last + 1)
        else:
            loss_forward2 = 0.0

        loss = lam1 * loss_forward + lam2 * loss_forward2

        log_prefix = "train" if self.training else "val"
        loss_dict = {
            "loss": loss,
            f"{log_prefix}/loss_forward": loss_forward,
            f"{log_prefix}/loss_forward2": loss_forward2,
        }
        return loss_dict


class DYffusionEndToEndBase(BaseDYffusion2):
    def __init__(
        self,
        reconstruction_start_step: int = 0,
        lambda_interpolation: float = 0.25,
        lambda_interpolation2: float = 0.25,
        lambda_reconstruction: float = 0.5,
        lambda_reconstruction2: float = 0.0,
        reconstruction2_detach_x_last: bool = False,
        detach_interpolated_data: bool = False,
        detach_interpolated_data2: bool = False,
        loss_function_interpolation: Optional[str | DictConfig] = None,
        loss_function_forecast: Optional[str | DictConfig] = None,
        condition_on_x_last: bool = True,
        interpolator_freeze_start_step: Union[int, str] = None,
        refine_intermediate_predictions: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters(ignore=["model"])
        assert not refine_intermediate_predictions, "refine_intermediate_predictions is not supported"

        assert reconstruction_start_step >= 0, "reconstruction_start_step must be >= 0"
        if interpolator_freeze_start_step == "reconstruction":
            if reconstruction_start_step == 0:
                raise ValueError(
                    'reconstruction_start_step must be > 0 if interpolator_freeze_start_step is "reconstruction"'
                )
            interpolator_freeze_start_step = reconstruction_start_step
        assert interpolator_freeze_start_step is None or interpolator_freeze_start_step >= 0
        self.interpolator_freeze_start_step = interpolator_freeze_start_step
        if self.interpolator_freeze_start_step and self.interpolator_freeze_start_step > 0:
            self.log_text.info(f"Freezing interpolator at step {self.interpolator_freeze_start_step}")

        self.condition_on_x_last = condition_on_x_last
        if not self.condition_on_x_last:
            self.log_text.info("condition_on_x_last is False, so we will not use x_th in the interpolator!")

    def get_loss_callable(self, reduction: str = "mean"):
        loss_function_forecast = self._get_loss_callable_from_name_or_config(
            self.hparams.loss_function_forecast, reduction=reduction
        )
        loss_function_interpolation = self._get_loss_callable_from_name_or_config(
            self.hparams.loss_function_interpolation, reduction=reduction
        )
        return dict(forecast=loss_function_forecast, interpolation=loss_function_interpolation)

    @property
    def interpolator_is_frozen(self):
        return (
            self.interpolator_freeze_start_step is not None
            and self.trainer.global_step >= self.interpolator_freeze_start_step
        )

    def _interpolate(
        self,
        x_ti: Tensor,
        x_th: Tensor,
        time: Tensor,
        **kwargs
    ):
        """Draw the intermediate degraded data (given the start/target data and the diffused data)"""
        interpolation_cond = self.get_condition("interpolate", x_th)
        with torch.set_grad_enabled(not self.interpolator_is_frozen):
            x_interpolated = self._predict_interpolated_snapshot(x_ti, x_th, time, interpolation_cond, **kwargs)
            x_interpolated = x_interpolated.detach() if self.interpolator_is_frozen else x_interpolated
        return x_interpolated

    @abstractmethod
    def _predict_interpolated_snapshot(
        self, x_ti: Tensor, x_th: Tensor, time: Tensor, interpolation_condition: Tensor, **kwargs
    ) -> Tensor:
        pass

    def p_losses(self, dynamics, verbose=False, **kwargs):
        r"""

        Args:
            x_t: the time=t_i_next "diffused" data, where t_i_next \in [0, T-1]
            x_last: the start/target data  (time = T)
            condition: the x_ti data  (time = 0)
            t: the time step of the diffusion process
        """
        # x_ti is what multi-horizon exp passes as targets, and x_th is the last timestep of the data dynamics
        # check that the time step is valid (between 0 and horizon-1)
        # E.g. dynamics.shape: torch.Size([3, 7, 34, 180, 360]), static_condition.shape: torch.Size([3, 2, 180, 360])
        # get rank of process
        # rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        # print_gpu_memory_usage(empty_cache=False, prefix=f'p_losses rank={rank}')
        B = dynamics.shape[0]
        assert (
            dynamics.shape[1] == self.num_timesteps + 1
        ), f"dynamics.shape[1] != num_timesteps: {dynamics.shape}[1] != {self.num_timesteps + 1}"
        x_th = dynamics[:, -1, ...]  # x_h
        # Sample input time step
        t_abs = torch.randint(0, self.num_timesteps, (B,), device=self.device, dtype=self.dtype)
        t_nonzero_mask = t_abs > 0
        t_not_last_mask = t_abs < self.num_timesteps - 1
        t_abs_nonzero = t_abs[t_nonzero_mask].long()
        t_abs_not_last = t_abs[t_not_last_mask].long()
        # assert torch.all(1 <= t_lead) and torch.all(t_lead <= self.num_timesteps)

        lam1 = self.hparams.lambda_interpolation
        lam2 = self.hparams.lambda_reconstruction
        lam3 = self.hparams.lambda_interpolation2
        lam4 = self.hparams.lambda_reconstruction2

        loss_function_interpolation = self.criterion["interpolation"]
        loss_function_forecast = self.criterion["forecast"]

        # since we do not need to interpolate x_0, we can skip all batches where t_i_next=0
        compute_interpolation_loss1 = (not self.interpolator_is_frozen) and lam1 > 0
        compute_interpolation_loss2 = (not self.interpolator_is_frozen) and lam3 > 0
        compute_forecast_loss = self.trainer.global_step >= self.hparams.reconstruction_start_step and lam2 > 0
        compute_forecast_loss2 = self.trainer.global_step >= self.hparams.reconstruction_start_step and lam4 > 0
        if (compute_interpolation_loss1 or lam2 > 0) and t_nonzero_mask.any():
            # Interpolate x_{t} from x_{t-1} and x_{h} for 1 <= t <= h-1
            xhat_ti = self._q_sample(
                x_ti=dynamics[t_nonzero_mask, t_abs_nonzero - 1, ...],
                x_th=x_th[t_nonzero_mask],
                time_of_input=t_abs_nonzero - 1,
                **{k: v[t_nonzero_mask] if torch.is_tensor(v) else v for k, v in kwargs.items()},
                # , t_abs_nonzero],
            )  # \hat{x}_{t}

        if compute_interpolation_loss1 and t_nonzero_mask.any():
            loss_interpolation = loss_function_interpolation(xhat_ti, dynamics[t_nonzero_mask, t_abs_nonzero, ...])
        else:
            loss_interpolation = 0.0

        if compute_forecast_loss or compute_interpolation_loss2:
            # xhat_ti may be float16, when using mixed precision, while the x_t0 is float32
            xtilde_t = dynamics[:, 0, ...].clone()  # .to(xhat_ti.dtype)
            if t_nonzero_mask.any():
                if self.hparams.detach_interpolated_data:
                    xhat_ti = xhat_ti.detach()
                # \tilde{x}_{t} = x_{0} for t=0, and \hat{x}_{t} for t>0
                xtilde_t[t_nonzero_mask, ...] = xhat_ti.to(xtilde_t.dtype)
            # assert torch.all(x_ti[t_i_next == 0] == x_ti[t_i_next == 0]), 'x_ti[t_i_next == 0] != x_ti[t_i_next == 0]'

            # Forecast x_{h} from \tilde{x}_{t} for 0 <= t <= h-1
            xhat_th = self.predict_x_last(x_t=xtilde_t, time_of_input=t_abs, **kwargs)  # \hat{x}_{h}

        if compute_forecast_loss:
            loss_forward = loss_function_forecast(xhat_th, x_th, time_of_input=t_abs)
        else:
            loss_forward = 0.0

        if (compute_interpolation_loss2 or compute_forecast_loss2) and t_not_last_mask.any():
            # Interpolate x_{t+1} from \tilde{x}_{t} and \hat{x}_{h} for 1 <= t <= h-1
            if self.hparams.reconstruction2_detach_x_last:
                xhat_th = xhat_th.detach()
            if self.hparams.detach_interpolated_data2:
                xtilde_t = xtilde_t.detach()
            xhat_ti_next = self._q_sample(
                x_ti=xtilde_t,
                x_th=xhat_th,
                time_of_input=t_abs,
                batch_mask=t_not_last_mask,
                **kwargs
            )  # \hat{x}_{t+1}

        if compute_interpolation_loss2 and t_not_last_mask.any():
            loss_interpolation2 = loss_function_interpolation(
                xhat_ti_next, dynamics[t_not_last_mask, t_abs_not_last + 1, ...]
            )
        else:
            loss_interpolation2 = 0.0

        if compute_forecast_loss2 and t_not_last_mask.any():
            # simulate one more step of the reverse diffusion process, i.e.
            # forecast x_{h} from \hat{x}_{t'} for 1 <= t' <= h-1, where t' = t+1
            kwargs = {k: v[t_not_last_mask] if torch.is_tensor(v) else v for k, v in kwargs.items()}
            xhat_th2 = self.predict_x_last(xhat_ti_next, time_of_input=t_abs_not_last + 1, **kwargs)
            loss_forward2 = loss_function_forecast(xhat_th2, x_th[t_not_last_mask], time_of_input=t_abs_not_last + 1)
        else:
            loss_forward2 = 0.0

        loss = lam1 * loss_interpolation + lam2 * loss_forward + lam3 * loss_interpolation2 + lam4 * loss_forward2
        log_prefix = "train" if self.training else "val"
        loss_dict = {
            "loss": loss,
            f"{log_prefix}/loss_forward": loss_forward,
            f"{log_prefix}/loss_forward2": loss_forward2,
            f"{log_prefix}/loss_interpolation": loss_interpolation,
            f"{log_prefix}/loss_interpolation2": loss_interpolation2,
        }
        return loss_dict


class DYffusionEnd2End(DYffusionEndToEndBase):
    def __init__(
        self,
        mask_type: str = "zeros",
        use_separate_heads: Union[bool, str] = False,
        interpolator_head_dropout: float = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters(ignore=["model"])
        okay_mask_types = ["zeros", "embedding", "embedding_zero_init"]
        raise_error_if_invalid_value(mask_type, okay_mask_types, name="mask_type")

        if "embedding" in mask_type:
            # initialize the embedding for the masked out time=T data
            dims_flattened = self.num_input_channels * self.spatial_shape_in[0] * self.spatial_shape_in[1] // 2
            self.mask_embedding = nn.Embedding(1, dims_flattened)
            if mask_type == "embedding_zero_init":
                self.mask_embedding.weight.data.zero_()

        assert interpolator_head_dropout is None or use_separate_heads == "v2"
        if use_separate_heads not in [False]:
            assert hasattr(self.model, "set_head_to_identity"), "model must have a set_head_to_identity method"
            assert hasattr(self.model, "get_head"), "model must have a get_head method"
            self.model.set_head_to_identity()

            self.head_forward = self.model.get_head()
            self.head_interpolate = self.model.get_head()

            if use_separate_heads in ["v2"]:
                self.head_interpolate_block = self.model.get_extra_last_block(dropout=interpolator_head_dropout)

            elif use_separate_heads not in [True, "v1"]:
                raise ValueError(f"Invalid value for use_separate_heads: {use_separate_heads}")
        else:
            self.head_interpolate = nn.Identity()
            self.head_forward = nn.Identity()

    def get_condition(
        self,
        prediction_type: str,
        x_th: Optional[Tensor] = None,
        shape: Sequence[int] = None,
        concat_dim: int = 1,
    ) -> Tensor:
        """Get the x_ti for the model,
        which is the concatenation of the start/target data and the diffused data,
        i.e. with shape (batch_size, 2 * channels, height, width).
        """
        if prediction_type == "interpolate":
            assert x_th is not None, "x_th must be provided for interpolation"
            cond1 = None
            if self.condition_on_x_last:
                cond2 = x_th
            else:
                self.log_text.warning("Not conditioning on x_t+h for interpolation!")
                cond2 = None

        elif prediction_type == "forward":
            assert shape is not None, "shape must be provided for forward prediction"
            cond1 = None
            if not self.condition_on_x_last:
                cond2 = None
            elif self.hparams.mask_type == "zeros":
                cond2 = torch.zeros(*shape, device=self.device)
            elif "embedding" in self.hparams.mask_type:
                # mask out the time=T data with an embedding
                mask = self.mask_embedding(torch.zeros(shape[0], dtype=torch.long, device=self.device))
                cond2 = mask.view(shape)
            else:
                raise ValueError(f"Invalid mask_type: {self.hparams.mask_type}")

        else:
            raise ValueError(f"unknown prediction type {prediction_type}")

        if cond1 is not None and cond2 is not None:
            cond = torch.cat([cond1, cond2], dim=concat_dim)
        elif cond1 is None and cond2 is not None:
            cond = cond2
        elif cond2 is None and cond1 is not None:
            cond = cond1
        else:
            cond = None
        return cond

    def get_time_encoding_interpolation(self, time_of_input: Tensor) -> Tensor:
        # Simply interpolate the next data timestep, i.e. use lead time = 1 as condition for time
        t = torch.ones(time_of_input.shape[0], device=self.device, dtype=self.dtype)
        return t

    def _predict_interpolated_snapshot(
        self, x_ti: Tensor, x_th: Tensor, time: Tensor, interpolation_condition: Tensor, **kwargs
    ) -> Tensor:
        assert (1 == time).all(), f"interpolate time must be == 1, got {time}"
        x_ipolated, t = self.model(x_ti, time=time, condition=interpolation_condition, return_time_emb=True, **kwargs)

        if hasattr(self, "head_interpolate_block"):
            x_ipolated = self.head_interpolate_block(x_ipolated, time_emb=t)
        x_ipolated = self.head_interpolate(x_ipolated)
        return x_ipolated

    def _predict_last_dynamics(self, x_t: Tensor, forward_condition: Tensor, t: Tensor, **kwargs) -> Tensor:
        assert (0 < t).all() and (t <= self.num_timesteps).all(), f"Invalid timestep: {t}"
        if self.hparams.time_encoding == "discrete":
            time = t
        elif self.hparams.time_encoding == "continuous":
            raise NotImplementedError("time_encoding=continuous not implemented")
            time = t / self.num_timesteps
        else:
            raise ValueError(f"Invalid time_encoding: {self.hparams.time_encoding}")

        x_last_pred = self.model.predict_forward(x_t, time=time, condition=forward_condition, **kwargs)
        x_last_pred = self.head_forward(x_last_pred)
        return x_last_pred


class DYffusionEnd2EndSeparateModels(DYffusionEndToEndBase):
    def __init__(self, interpolator: DictConfig, enable_interpolator_inference_dropout: bool = False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters(ignore=["model", "interpolator"])
        ipol_dim_in = self.num_input_channels
        ipol_dim_out = self.num_output_channels
        ipol_dim_cond = self.num_conditional_channels
        if self.condition_on_x_last:
            ipol_dim_cond += ipol_dim_out

        self.log_text.info(
            f"Creating interpolator with dims: input={ipol_dim_in}, output={ipol_dim_out}, conditional={ipol_dim_cond}"
        )
        self.interpolator: BaseModel = hydra.utils.instantiate(
            interpolator,
            num_input_channels=ipol_dim_in,
            num_output_channels=ipol_dim_out,
            num_conditional_channels=ipol_dim_cond,
            spatial_shape_in=self.spatial_shape_in,
            spatial_shape_out=self.spatial_shape_out,
            _recursive_=False,
        )
        self._max_ipol_time = self.num_timesteps - 2
        if hasattr(self.interpolator, "set_min_max_time"):
            self.interpolator.set_min_max_time(min_time=0, max_time=self._max_ipol_time)

    def get_condition(
        self,
        prediction_type: str,
        x_th: Optional[Tensor] = None,
        shape: Sequence[int] = None,
        concat_dim: int = 1,
    ) -> Tensor:
        if prediction_type == "interpolate":
            assert x_th is not None, "x_th must be provided for interpolation"
            if self.condition_on_x_last:
                cond = x_th
            else:
                self.log_text.warning("Not conditioning on x_t+h for interpolation!")
                cond = None
        elif prediction_type == "forward":
            assert shape is not None, "shape must be provided for forward prediction"
            cond = None
        else:
            raise ValueError(f"Unknown prediction type {prediction_type}")
        return cond

    def get_time_encoding_interpolation(self, time: Tensor) -> Tensor:
        return time

    def _predict_interpolated_snapshot(
        self, x_ti: Tensor, x_th: Tensor, time: Tensor, interpolation_condition: Tensor, **kwargs
    ) -> Tensor:
        assert (0 <= time).all() and (
            time <= self._max_ipol_time
        ).all(), f"interpolate time must be in [0, {self._max_ipol_time}], got {time}"
        with self.interpolator.inference_dropout_scope(condition=self.hparams.enable_interpolator_inference_dropout):
            x_ipolated = self.interpolator(x_ti, time=time, condition=interpolation_condition, **kwargs)
        return x_ipolated

    def _predict_last_dynamics(self, x_t: Tensor, forward_condition: Tensor, t: Tensor,  **kwargs) -> Tensor:
        assert (0 < t).all() and (t <= self.num_timesteps).all(), f"Invalid lead time timestep: {t}"
        x_last_pred = self.model.predict_forward(x_t, time=t, condition=forward_condition, **kwargs)
        return x_last_pred
