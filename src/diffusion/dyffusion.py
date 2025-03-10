from __future__ import annotations

import math
from abc import abstractmethod
from contextlib import ExitStack
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import torch
from torch import Tensor, nn
from tqdm.auto import tqdm

from src.diffusion._base_diffusion import BaseDiffusion
from src.experiment_types.interpolation import InterpolationExperiment
from src.utilities.checkpointing import get_checkpoint_from_path_or_wandb
from src.utilities.random_control import controlled_rng
from src.utilities.utils import freeze_model, raise_error_if_invalid_value


class BaseDYffusion(BaseDiffusion):
    #         enable_interpolator_dropout: whether to enable dropout in the interpolator
    def __init__(
        self,
        forward_conditioning: str = "data",
        dynamic_cond_from_t: str = "h",  # 'h', '0', or 't'
        schedule: str = "before_t1_only",
        additional_interpolation_steps: int = 0,
        additional_interpolation_steps_factor: int = 0,
        interpolate_before_t1: bool = True,
        sampling_type: str = "cold",  # 'cold' or 'naive'
        sampling_schedule: Union[List[float], str] = None,
        use_cold_sampling_for_intermediate_steps: bool = True,
        use_cold_sampling_for_last_step: bool = True,
        use_cold_sampling_for_init_of_ar_step: Optional[bool] = None,
        time_encoding: str = "discrete",
        refine_predictions: str | bool = False,
        refinement_rounds: int = 0,
        refine_intermediate_predictions: bool = False,
        prediction_timesteps: Optional[Sequence[float]] = None,
        enable_interpolator_dropout: Union[bool, str] = True,
        interpolator_use_ema: bool = False,
        enable_predict_last_dropout: bool = False,
        log_every_t: Union[str, int] = None,
        use_same_dropout_state_for_sampling: str = "random",  # or "fixed_global" or "fixed_per_iter"
        reconstruction2_detach_x_last=None,
        hack_for_imprecise_interpolation: bool = False,
        *args,
        **kwargs,
    ):
        use_cold_sampling_for_init_of_ar_step = (
            use_cold_sampling_for_init_of_ar_step
            if use_cold_sampling_for_init_of_ar_step is not None
            else use_cold_sampling_for_last_step
        )
        super().__init__(*args, **kwargs, sampling_schedule=sampling_schedule)
        sampling_schedule = None if sampling_schedule == "None" else sampling_schedule
        self.save_hyperparameters(ignore=["model"])
        self.num_timesteps = self.hparams.timesteps
        self.use_cold_sampling_for_init_of_ar_step = use_cold_sampling_for_init_of_ar_step

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

        # Add additional interpolation steps to the diffusion steps
        # we substract 2 because we don't want to use the interpolator in timesteps outside [1, num_timesteps-1]
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
        d_to_i_step = {d: self.diffusion_step_to_interpolation_step(d) for d in range(1, self.num_timesteps)}
        self.dynamical_steps = {d: i_n for d, i_n in d_to_i_step.items() if float(i_n).is_integer()}
        self.i_to_diffusion_step = {i_n: d for d, i_n in d_to_i_step.items()}
        self.artificial_interpolation_steps = {d: i_n for d, i_n in d_to_i_step.items() if not float(i_n).is_integer()}
        # check that float tensors and floats return the same value
        for d, i_n in d_to_i_step.items():
            i_n2 = float(self.diffusion_step_to_interpolation_step(torch.tensor(d, dtype=torch.float)))
            assert math.isclose(
                i_n, i_n2, abs_tol=4e-6
            ), f"float and tensor return different values for diffusion_step_to_interpolation_step({d}): {i_n} != {i_n2}"
        # note that self.dynamical_steps does not include t=0, which is always dynamical (but not an output!)
        if additional_interpolation_steps_factor > 0 or additional_interpolation_steps > 0:
            self.log_text.info(
                f"Added {self.additional_diffusion_steps} steps.. total diffusion num_timesteps={self.num_timesteps}. \n"
                # f'Mapping diffusion -> interpolation steps: {d_to_i_step}. \n'
                f"Diffusion -> Dynamical timesteps: {self.dynamical_steps}."
            )
        self.enable_interpolator_dropout = enable_interpolator_dropout
        raise_error_if_invalid_value(
            enable_interpolator_dropout,
            [True, False, "always", "except_dynamical_steps"],
            "enable_interpolator_dropout",
        )
        if enable_predict_last_dropout:
            self.log_text.info("Enabling dropout in forward prediction of last timestep during inference.")
        if self.hparams.interpolator_use_ema:
            self.log_text.info("Using EMA for the interpolator.")
        if refine_intermediate_predictions:
            self.log_text.info("Enabling refinement of intermediate predictions.")
        refine_predictions_options = [False, "all", "intermediate"]
        raise_error_if_invalid_value(refine_predictions, refine_predictions_options, "refine_predictions")
        if not refine_predictions:
            assert refinement_rounds == 0, "refinement_rounds must be 0 if refine_predictions is False"

        # which diffusion steps to take during sampling
        self.full_sampling_schedule = list(range(0, self.num_timesteps))
        self.sampling_schedule = sampling_schedule or self.full_sampling_schedule

    @property
    def diffusion_steps(self) -> List[int]:
        return list(range(0, self.num_timesteps))

    def diffusion_step_to_interpolation_step(self, diffusion_step: Union[int, Tensor]) -> Union[float, Tensor]:
        """
        Convert a diffusion step to an interpolation step
        Args:
            diffusion_step: the diffusion step  (in [0, num_timesteps-1])
        Returns:
            the interpolation step
        """
        # assert correct range
        if torch.is_tensor(diffusion_step):
            assert (0 <= diffusion_step).all() and (
                diffusion_step <= self.num_timesteps - 1
            ).all(), f"diffusion_step must be in [0, num_timesteps-1]=[0, {self.num_timesteps - 1}], but got {diffusion_step}"
        else:
            assert (
                0 <= diffusion_step <= self.num_timesteps - 1
            ), f"diffusion_step must be in [0, num_timesteps-1]=[0, {self.num_timesteps - 1}], but got {diffusion_step}"
        if self.hparams.schedule == "linear":
            # self.di_to_ti_add is 0 or 1
            # Self.additional_interpolation_steps_fac is 0 by default (no additional interpolation steps)
            i_n = (diffusion_step + self.di_to_ti_add) / (self.additional_interpolation_steps_fac + 1)
        elif self.hparams.schedule == "before_t1_only":
            # map d_N to h-1, d_N-1 to h-2, ..., d_n to 1, and d_n-1..d_1 uniformly to [0, 1)
            # e.g. if h=5, then d_5 -> 4, d_4 -> 3, d_3 -> 2, d_2 -> 1, d_1 -> 0.5
            # or                d_6 -> 4, d_5 -> 3, d_4 -> 2, d_3 -> 1, d_2 -> 0.66, d_1 -> 0.33
            # or                d_7 -> 4, d_6 -> 3, d_5 -> 2, d_4 -> 1, d_3 -> 0.75, d_2 -> 0.5, d_1 -> 0.25
            if torch.is_tensor(diffusion_step):
                i_n = torch.where(
                    diffusion_step >= self.additional_diffusion_steps + 1,
                    (diffusion_step - self.additional_diffusion_steps).float(),
                    diffusion_step / (self.additional_diffusion_steps + 1),
                )
            elif diffusion_step >= self.additional_diffusion_steps + 1:
                i_n = diffusion_step - self.additional_diffusion_steps
            else:
                i_n = diffusion_step / (self.additional_diffusion_steps + 1)
        elif self.hparams.schedule == "before_t1_then_linear":
            if torch.is_tensor(diffusion_step):
                i_n = torch.where(
                    diffusion_step >= self.additional_diffusion_steps_pre_t1 + 1,
                    1
                    + (diffusion_step - self.additional_diffusion_steps_pre_t1 - 1)
                    / (self.additional_interpolation_steps_fac + 1),
                    diffusion_step / (self.additional_diffusion_steps_pre_t1 + 1),
                )
            elif diffusion_step >= self.additional_diffusion_steps_pre_t1 + 1:
                i_n = 1 + (diffusion_step - self.additional_diffusion_steps_pre_t1 - 1) / (
                    self.additional_interpolation_steps_fac + 1
                )
            else:
                i_n = diffusion_step / (self.additional_diffusion_steps_pre_t1 + 1)
        else:
            raise ValueError(f"schedule=``{self.hparams.schedule}`` not supported.")

        return i_n

    def q_sample(self, *args, random_mode: str = "random", iteration: int = 0, **kwargs):
        random_mode = "random" if random_mode is False else random_mode  # legacy support
        with controlled_rng(random_mode, iteration=iteration):
            return self._q_sample(*args, **kwargs)

    def _q_sample(
        self,
        x0,
        x_end,
        t: Optional[Tensor],
        interpolation_time: Optional[Tensor] = None,
        batch_mask: Optional[Tensor] = None,
        is_artificial_step: bool = True,
        **kwargs,
    ) -> Tensor:
        # q_sample = using model in interpolation mode
        # just remember that x_end here refers to t=0 (the initial conditions)
        # and x_0 (terminology of diffusion models) refers to t=T, i.e. the last timestep
        ipol_handles = [self.interpolator] if hasattr(self, "interpolator") else [self]
        if hasattr(self, "interpolator_artificial_steps") and self.interpolator_artificial_steps is not None:
            ipol_handles.append(self.interpolator_artificial_steps)

        # Apply mask if necessary on batch dimension
        if batch_mask is not None:
            assert interpolation_time is None, "interpolation_time must be None if batch_mask is not None."
            assert torch.is_tensor(t), "t must be a tensor if batch_mask is not None."
            x0 = x0[batch_mask]
            x_end = x_end[batch_mask]
            t = t[batch_mask]
            kwargs = {k: v[batch_mask] if torch.is_tensor(v) else v for k, v in kwargs.items()}

        assert t is None or interpolation_time is None, "Either t or interpolation_time must be None."
        t = interpolation_time if t is None else self.diffusion_step_to_interpolation_step(t)  # .float()

        # Handle dynamical cond based on logic in src.experimental_types.interpolation.InterpolationExperiment
        dynamical_cond = kwargs.pop("dynamical_condition", None)
        if dynamical_cond is not None:
            kwargs["condition"] = ipol_handles[0].get_dynamical_condition(dynamical_cond, t)

        # Tensorfy t if it is a float/int
        if not torch.is_tensor(t):
            t = torch.full((x0.shape[0],), t, dtype=torch.float32, device=self.device)

        do_enable = (
            self.training
            or self.enable_interpolator_dropout in [True, "always"]
            or (self.enable_interpolator_dropout == "except_dynamical_steps" and is_artificial_step)
        )

        with ExitStack() as stack:
            # inference_dropout_scope of all handles (enable and disable) is managed by the ExitStack
            for ipol in ipol_handles:
                stack.enter_context(ipol.inference_dropout_scope(condition=do_enable))
                if self.hparams.interpolator_use_ema:
                    stack.enter_context(ipol.ema_scope(condition=True))

            x_ti = self._interpolate(initial_condition=x_end, x_last=x0, t=t, **kwargs)
        return x_ti

    @abstractmethod
    def _interpolate(
        self,
        initial_condition: Tensor,
        x_last: Tensor,
        t: Tensor,
        num_predictions: int = 1,
        **kwargs,
    ):
        """This is an internal method. Please use q_sample to access it."""
        raise NotImplementedError(f"``_interpolate`` must be implemented in {self.__class__.__name__}")

    def get_condition(
        self,
        initial_condition_cond: Optional[Tensor],
        x_last: Optional[Tensor],
        prediction_type: str,
        condition: Optional[Tensor] = None,
        shape: Sequence[int] = None,
    ) -> Tensor:
        if initial_condition_cond is not None and condition is not None:
            return torch.cat([initial_condition_cond, condition], dim=1)
        elif initial_condition_cond is not None and condition is not None:
            return torch.cat([initial_condition_cond, condition], dim=1)
        elif initial_condition_cond is not None:
            return initial_condition_cond
        elif condition is not None:
            return condition
        else:
            return None

    @property
    def valid_time_range_for_backbone_model(self) -> List[int]:
        diff_steps = list(range(0, self.num_timesteps))
        if self.hparams.time_encoding == "discrete":
            valid_time = diff_steps
        elif self.hparams.time_encoding == "continuous":
            valid_time = list(np.array(diff_steps) / self.num_timesteps)
        elif self.hparams.time_encoding == "dynamics":
            valid_time = [self.diffusion_step_to_interpolation_step(d) for d in diff_steps]
        else:
            raise ValueError(f"Invalid time_encoding: {self.hparams.time_encoding}")
        return valid_time

    def _predict_last_dynamics(self, condition: Tensor, x_t: Tensor, t: Tensor, **kwargs):
        if self.hparams.time_encoding == "discrete":
            time = t
        elif self.hparams.time_encoding == "continuous":
            time = t / self.num_timesteps
        elif self.hparams.time_encoding == "dynamics":
            time = self.diffusion_step_to_interpolation_step(t)
        else:
            raise ValueError(f"Invalid time_encoding: {self.hparams.time_encoding}")

        dropout_condition = self.training
        if not self.training:
            # If sampling, we may enable Monte Carlo dropout
            if self.hparams.enable_predict_last_dropout == "except_last_step":
                dropout_condition = bool((t < self.num_timesteps - 1).all())
            elif self.hparams.enable_predict_last_dropout:
                dropout_condition = True
        # enable dropout for prediction of last dynamics
        with self.model.inference_dropout_scope(condition=dropout_condition):
            x_last_pred = self.model.predict_forward(x_t, time=time, condition=condition, **kwargs)
        return x_last_pred

    def predict_x_last(
        self,
        initial_condition: Tensor,
        x_t: Tensor,
        t: Tensor,
        **kwargs,
    ):
        """Predict x_{t+h} given x_t"""
        if not torch.is_tensor(t):
            assert 0 <= t <= self.num_timesteps - 1, f"Invalid timestep: {t}. {self.num_timesteps=}"
            t = torch.full((initial_condition.shape[0],), t, dtype=torch.float32, device=self.device)
        else:
            assert (0 <= t).all() and (t <= self.num_timesteps - 1).all(), f"Invalid timestep: {t}"
        cond_type = self.hparams.forward_conditioning
        if cond_type == "data":
            forward_inputs = initial_condition
        elif cond_type == "none":
            forward_inputs = None
        elif cond_type == "noise":
            forward_inputs = torch.randn_like(initial_condition)
        elif cond_type == "data|noise":
            forward_inputs = torch.cat([initial_condition, torch.randn_like(initial_condition)], dim=1)
        elif "data+noise" in cond_type:
            # simply use factor t/T to scale the condition and factor (1-t/T) to scale the noise
            # this is the same as using a linear combination of the condition and noise
            tfactor = t / (self.num_timesteps - 1)  # shape: (b,)
            tfactor = tfactor.view(
                initial_condition.shape[0], *[1] * (initial_condition.ndim - 1)
            )  # shape: (b, 1, 1, 1)
            if cond_type == "data+noise-v1":
                # add noise to the data in a linear combination, s.t. the noise is more important at the beginning (t=0)
                # and less important at the end (t=T)
                forward_inputs = tfactor * initial_condition + (1 - tfactor) * torch.randn_like(initial_condition)
            elif cond_type == "data+noise-v2":
                forward_inputs = (1 - tfactor) * initial_condition + tfactor * torch.randn_like(initial_condition)
        elif cond_type == "t0noise":
            first_step = t == 0
            forward_inputs = initial_condition.clone()
            forward_inputs[first_step] = torch.randn_like(initial_condition[first_step])
        else:
            raise ValueError(f"Invalid forward conditioning type: {cond_type}")

        dynamic_cond = kwargs.pop("dynamical_condition", None)  # a window (=1) + horizon (=T) tensor
        if dynamic_cond is not None:
            assert (
                dynamic_cond.shape[1] == self.num_timesteps + 1
            ), f"{dynamic_cond.shape}[1] != {self.num_timesteps+1}. ({initial_condition.shape=}, {x_t.shape=})"
            if self.hparams.dynamic_cond_from_t == "0":
                dynamic_cond = self.slice_time(dynamic_cond, 0)  # take from initial conditions timestep
            elif self.hparams.dynamic_cond_from_t == "h":
                dynamic_cond = self.slice_time(dynamic_cond, -1)  # take from last timestep (to predict)
            elif self.hparams.dynamic_cond_from_t == "t":
                dynamic_cond = self.slice_time(dynamic_cond, t)  # take from input timestep
            else:
                raise ValueError(f"Invalid dynamic_cond_from_t: {self.hparams.dynamic_cond_from_t}")

        forward_inputs = self.get_condition(
            initial_condition_cond=forward_inputs,
            x_last=None,
            prediction_type="forward",
            shape=initial_condition.shape,
            condition=dynamic_cond,
        )
        x_last_pred = self._predict_last_dynamics(x_t=x_t, condition=forward_inputs, t=t, **kwargs)
        return x_last_pred

    def slice_time(self, x: Tensor, t: Union[int, Tensor]) -> Tensor:
        if torch.is_tensor(t):
            b = x.shape[0]
            return x[torch.arange(b), t]
        return x[:, t]

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
        if isinstance(schedule_name, str):
            base_schedule = [0] + list(self.dynamical_steps.keys())  # already included: + [self.num_timesteps - 1]
            artificial_interpolation_steps = list(self.artificial_interpolation_steps.keys())
            if "only_dynamics" in schedule_name:
                schedule = []  # only sample from base_schedule (added below)
                if "only_dynamics_plus_random" in schedule_name:
                    rng = np.random.default_rng(seed=7)
                    dsteps_to_choose_from = list(self.artificial_interpolation_steps.keys())
                    # parse schedule 'only_dynamics_plus_randomFLOAT' to get FLOAT
                    plus_random = float(schedule_name.replace("only_dynamics_plus_random", ""))
                    assert 0 < plus_random, f"Invalid sampling schedule: {schedule}, must end with number/float > 0"
                    if plus_random < 1:
                        # Add any dstep from base_schedule with probability plus_random
                        schedule = [dstep for dstep in base_schedule if rng.random() < plus_random]
                    else:
                        assert (
                            plus_random.is_integer()
                        ), f"If plus_random >= 1, it must be an integer, but got {plus_random}"
                        # Add plus_random additional steps to the front of the schedule
                        schedule = list(
                            rng.choice(
                                dsteps_to_choose_from,
                                size=int(plus_random),
                                replace=False,
                            )
                        )

                elif "only_dynamics_plus" in schedule_name:
                    # parse schedule 'only_dynamics_plusN' to get N
                    plus_n = int(schedule_name.replace("only_dynamics_plus", "").replace("_discrete", ""))
                    # Add N additional steps to the front of the schedule
                    schedule = list(np.linspace(0, base_schedule[1], plus_n + 1, endpoint=False))
                    if "_discrete" in schedule_name:  # floor the values
                        schedule = [int(np.floor(s)) for s in schedule]
                else:
                    assert "only_dynamics" == schedule_name, f"Invalid sampling schedule: {schedule}"

            elif schedule_name.startswith("every"):
                # parse schedule 'everyNth' to get N
                every_nth = schedule.replace("every", "").replace("th", "").replace("nd", "").replace("rd", "")
                every_nth = int(every_nth)
                assert 1 <= every_nth <= self.num_timesteps, f"Invalid sampling schedule: {schedule}"
                schedule = artificial_interpolation_steps[::every_nth]

            elif schedule.startswith("first"):
                # parse schedule 'firstN' to get N
                first_n = float(schedule.replace("first", "").replace("v2", ""))
                if first_n < 1:
                    assert 0 < first_n < 1, f"Invalid sampling schedule: {schedule}, must end with number/float > 0"
                    first_n = int(np.ceil(first_n * len(artificial_interpolation_steps)))
                    schedule = artificial_interpolation_steps[:first_n]
                    self.log_text.info(f"Using sampling schedule: {schedule_name} -> (first {first_n} steps)")
                else:
                    assert first_n.is_integer(), f"If first_n >= 1, it must be an integer, but got {first_n}"
                    assert 1 <= first_n <= self.num_timesteps, f"Invalid sampling schedule: {schedule}"
                    first_n = int(first_n)
                    if "v2" in schedule:
                        # sample first N steps, but restarting at each dynamic step
                        schedule = []
                        for d, dnext in zip(base_schedule[:-1], base_schedule[1:]):
                            schedule += list(range(d + 1, dnext))[:first_n]
                    else:
                        # Simple schedule: sample using first N steps
                        schedule = artificial_interpolation_steps[:first_n]

            elif schedule.startswith("times"):
                # schedules add extra diffusion steps to the full sampling schedule
                # if 'timesN' -> use N times *more* diffusion steps than the training schedule
                # if 'times_dynamics_stepsN' -> add N extra diffusion steps in-between the dynamic steps
                # that is, if diffusion steps = [0, 1, 2, 3]
                # and timesN = 2, then the sampling schedule will be [0, 0.5, 1, 1.5, 2, 2.5, 3]
                times_n = float(schedule.replace("times_dynamics_steps", "").replace("times", ""))
                assert times_n > 0, f"Invalid sampling schedule: {schedule}"
                schedule_to_interpolate = (
                    base_schedule[1:] if "dynamics_steps" in schedule else self.full_sampling_schedule
                )
                schedule = []
                for d, dnext in zip(schedule_to_interpolate[:-1], schedule_to_interpolate[1:]):
                    schedule += list(np.linspace(d, dnext, int(times_n), endpoint=False))
                schedule += self.full_sampling_schedule
            else:
                # other schedules
                schedules = {}
                if schedule not in schedules:
                    raise ValueError(
                        f"Invalid sampling schedule: ``{schedule}``. Choose from {list(schedules.keys())}"
                    )
                schedule = schedules[schedule]

            # Add dynamic steps to the schedule
            schedule += base_schedule
            # need to sort in ascending order and remove duplicates
            schedule = list(sorted(set(schedule)))

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
                f"Are you sure you don't want to sample at the last timestep? (current last timestep: {last})"
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

    def sample_loop(
        self,
        initial_condition,
        log_every_t: Optional[Union[str, int]] = None,
        num_predictions: int = None,
        verbose=True,
        **kwargs,
    ):
        # b = initial_condition.shape[0]
        log_every_t = log_every_t or self.hparams.log_every_t
        log_every_t = log_every_t if log_every_t != "auto" else 1
        sampling_schedule = self.sampling_schedule

        assert len(initial_condition.shape) == 4, f"condition.shape: {initial_condition.shape} (should be 4D)"
        x_s = initial_condition[:, -self.num_input_channels :]
        intermediates, xhat_th, dynamics_pred_step = dict(), None, 0
        last_i_n_plus_one = sampling_schedule[-1] + 1
        dynamics_pred_step_last = None
        is_cold_sampling = self.hparams.sampling_type in ["cold", "heun1", "heun2", "heun3"]
        for sampling_round in range(0, self.hparams.refinement_rounds + 1):
            desc = f"Refinement round {sampling_round}" if sampling_round > 0 else "Sampling"
            s_and_snext = zip(
                sampling_schedule,
                sampling_schedule[1:] + [last_i_n_plus_one],
                sampling_schedule[2:] + [last_i_n_plus_one, last_i_n_plus_one + 1],
            )
            progress_bar = tqdm(s_and_snext, desc=desc, total=len(sampling_schedule), leave=False)
            x_s = initial_condition
            for sampling_step, (s, s_next, s_nnext) in enumerate(progress_bar):
                is_first_step = s == 0
                is_last_step = s == self.num_timesteps - 1

                if sampling_round == 0 or (self.hparams.refine_predictions == "all" and not is_first_step):
                    # Forecast x_{t+h} using x_{s} as input
                    xhat_th = self.predict_x_last(
                        initial_condition=initial_condition,
                        x_t=x_s,
                        t=s,
                        **kwargs,
                    )
                else:
                    # If sampling_round > 0 and refine_predictions == "all", use the refined x0_hat from previous round
                    xhat_th = intermediates[f"t{dynamics_pred_step_last}_preds"]

                # Are we predicting dynamical time step or an artificial interpolation step?
                time_i_n = self.diffusion_step_to_interpolation_step(s_next) if not is_last_step else np.inf
                is_dynamics_pred = float(time_i_n).is_integer() or is_last_step
                q_sample_kwargs = dict(
                    x0=xhat_th,
                    x_end=initial_condition,
                    is_artificial_step=not is_dynamics_pred,
                    num_predictions=num_predictions if is_first_step else 1,
                    random_mode=self.hparams.use_same_dropout_state_for_sampling,
                    iteration=sampling_step,
                )
                if s_next <= self.num_timesteps - 1:
                    # D(x_s, s-1)
                    x_interpolated_s_next = self.q_sample(**q_sample_kwargs, t=s_next, **kwargs)
                else:
                    assert is_last_step, f"Invalid s_next: {s_next} (should be <= {self.num_timesteps - 1})"
                    x_interpolated_s_next = xhat_th  # for the last step, we use the final x0_hat prediction
                    if self.hparams.hack_for_imprecise_interpolation:
                        x_interpolated_s_next = torch.cat([initial_condition[:, :1], x_interpolated_s_next], dim=1)

                if is_cold_sampling:
                    if not self.hparams.use_cold_sampling_for_last_step and is_last_step:
                        if self.hparams.use_cold_sampling_for_init_of_ar_step:
                            x_interpolated_s = self.q_sample(**q_sample_kwargs, t=s, **kwargs)
                            ar_init = x_s + xhat_th - x_interpolated_s
                            if self.hparams.hack_for_imprecise_interpolation:
                                ar_init = ar_init[:, 1:]
                            intermediates["preds_autoregressive_init"] = ar_init
                        x_s = xhat_th
                    else:
                        # D(x_s, s)
                        x_interpolated_s = self.q_sample(**q_sample_kwargs, t=s, **kwargs) if s > 0 else x_s
                        # for s = 0, we have x_interpolated_s = x_s, so we just directly return x_s_degraded_next
                        d_i1 = x_interpolated_s_next - x_interpolated_s
                        if (
                            self.hparams.sampling_type == "cold"
                            or s_nnext > self.num_timesteps - 1
                            or ("heun" in self.hparams.sampling_type and is_first_step)
                        ):
                            x_s = x_s + d_i1
                        elif "heun" in self.hparams.sampling_type:
                            # Heun correction
                            xs_tmp = x_s + d_i1
                            x0_hat2 = self.predict_x_last(
                                initial_condition=initial_condition,
                                x_t=xs_tmp,
                                t=s_next,
                                **kwargs,
                            )
                            q_sample_kwargs["x0"] = x0_hat2

                            x_interpolated_s_nnext = self.q_sample(**q_sample_kwargs, t=s_nnext, **kwargs)
                            if self.hparams.sampling_type == "heun1":
                                d_i2 = x_interpolated_s_nnext - xs_tmp  # Seems like the best option
                            elif self.hparams.sampling_type == "heun2":
                                x_interpolated_s_next2 = self.q_sample(**q_sample_kwargs, t=s_next, **kwargs)
                                d_i2 = x_interpolated_s_nnext - x_interpolated_s_next2
                            elif self.hparams.sampling_type == "heun3":
                                d_i2 = x_interpolated_s_nnext - x_interpolated_s_next

                            x_s = x_s + d_i1 / 2 + d_i2 / 2

                elif self.hparams.sampling_type == "naive":
                    x_s = x_interpolated_s_next
                else:
                    raise ValueError(f"unknown sampling type {self.hparams.sampling_type}")

                dynamics_pred_step = int(time_i_n) if s < self.num_timesteps - 1 else dynamics_pred_step + 1
                if is_last_step:
                    dynamics_pred_step_last = dynamics_pred_step
                if is_dynamics_pred:
                    if self.hparams.use_cold_sampling_for_intermediate_steps or is_last_step:
                        preds_t = x_s
                    else:
                        assert not self.hparams.use_cold_sampling_for_intermediate_steps and not is_last_step
                        preds_t = x_interpolated_s_next
                    if self.hparams.hack_for_imprecise_interpolation:
                        preds_t = preds_t[:, 1:]
                    intermediates[f"t{dynamics_pred_step}_preds"] = preds_t  # preds
                    if log_every_t is not None:
                        intermediates[f"t{dynamics_pred_step}_preds2"] = x_interpolated_s_next

                s1, s2 = s, s  # s + 1, next_step  # s, next_step
                if log_every_t is not None:
                    intermediates[f"x_{s2}_dmodel"] = x_s  # preds
                    intermediates[f"intermediate_{s1}_x0hat"] = xhat_th
                    intermediates[f"xipol_{s2}_dmodel"] = x_interpolated_s_next
                    if self.hparams.sampling_type == "cold":
                        intermediates[f"xipol_{s1}_dmodel2"] = x_interpolated_s

        if self.hparams.refine_intermediate_predictions:
            # Use last prediction of x0 for final prediction of intermediate steps (not the last timestep!)
            q_sample_kwargs["x0"] = xhat_th
            q_sample_kwargs["is_artificial_step"] = False
            q_sample_kwargs["random_mode"] = "fixed_global"  #  use the same dropout mask for all steps
            _ = q_sample_kwargs.pop("iteration", None)
            dynamical_steps = self.hparams.prediction_timesteps or list(self.dynamical_steps.values())
            dynamical_steps = [i for i in dynamical_steps if i < self.num_timesteps]
            for i_n in dynamical_steps:
                i_n_for_str = int(i_n) if float(i_n).is_integer() else i_n
                assert (
                    not float(i_n).is_integer() or f"t{i_n_for_str}_preds" in intermediates
                ), f"t{i_n_for_str}_preds not in intermediates"
                intermediates[f"t{i_n_for_str}_preds"] = self.q_sample(
                    **q_sample_kwargs,
                    t=None,
                    interpolation_time=i_n,
                    **kwargs,
                )
                if self.hparams.hack_for_imprecise_interpolation:
                    intermediates[f"t{i_n_for_str}_preds"] = intermediates[f"t{i_n_for_str}_preds"][:, 1:]
        if last_i_n_plus_one < self.num_timesteps:
            return x_s, intermediates
        return xhat_th, intermediates

    @torch.inference_mode()
    def sample(self, initial_condition, num_samples=1, **kwargs):
        x_0, intermediates = self.sample_loop(initial_condition, **kwargs)
        return intermediates

    def predict_forward(self, *inputs, metadata: Any = None, **kwargs):
        assert len(inputs) == 1, "Only one input tensor is allowed for the forward pass"
        inital_condition = inputs[0]
        return self.sample(inital_condition, **kwargs)


# --------------------------------------------------------------------------------
# DYffusion with a pretrained interpolator
# --------------------------------------------------------------------------------


class DYffusion(BaseDYffusion):
    """
    DYffusion model with a pretrained interpolator
    Args:
        interpolator: the interpolator model
        lambda_reconstruction: the weight of the reconstruction loss
        lambda_reconstruction2: the weight of the reconstruction loss (using the predicted xt_last as feedback)
    """

    def __init__(
        self,
        interpolator: Optional[nn.Module] = None,
        interpolator_run_id: Optional[str] = None,
        interpolator_checkpoint_path: Optional[str] = None,
        interpolator_artificial_steps_run_id: Optional[str] = None,
        lambda_reconstruction: float = 1.0,
        lambda_reconstruction2: float = 0.0,
        lambda_reverse_diffusion: float = 0.0,
        lambda_consistency: float = 0.0,
        consistency_strategy: str = "ema",
        reconstruction2_detach_x_last: bool = False,
        penalize_reverse_diffusion_steps: str = "last",
        interpolator_local_checkpoint_path: Optional[Union[str, bool]] = True,  # if true, search in local path
        interpolator_overrides: Optional[List[str]] = None,  # a dot list, e.g. ["model.hidden_dims=128"]
        interpolator_wandb_ckpt_filename: Optional[str] = None,
        interpolator_wandb_kwargs: Optional[Dict[str, Any]] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters(ignore=["interpolator", "model"])
        self.name = self.name or "DYffusion (2stage)"
        assert penalize_reverse_diffusion_steps == "last", "only penalize the last step for now"
        # Load interpolator and its weights
        interpolator_wandb_kwargs = interpolator_wandb_kwargs or {}
        interpolator_wandb_kwargs["epoch"] = interpolator_wandb_kwargs.get("epoch", "best")
        if interpolator_wandb_ckpt_filename is not None:
            assert interpolator_wandb_kwargs.get("ckpt_filename") is None, "ckpt_filename already set"
            interpolator_wandb_kwargs["ckpt_filename"] = interpolator_wandb_ckpt_filename
        interpolator_overrides = list(interpolator_overrides) if interpolator_overrides is not None else []
        interpolator_overrides.append("model.verbose=False")
        self.interpolator: InterpolationExperiment = get_checkpoint_from_path_or_wandb(
            interpolator,
            model_checkpoint_path=interpolator_local_checkpoint_path,
            wandb_run_id=interpolator_run_id,
            reload_kwargs=interpolator_wandb_kwargs,
            model_overrides=interpolator_overrides,
        )
        # freeze the interpolator (and set to eval mode)
        freeze_model(self.interpolator)

        self.interpolator_window = self.interpolator.window
        self.interpolator_horizon = self.interpolator.true_horizon
        last_d_to_i_tstep = self.diffusion_step_to_interpolation_step(self.num_timesteps - 1)
        if self.interpolator_horizon != last_d_to_i_tstep + 1:
            # maybe: automatically set the num_timesteps to the interpolator_horizon
            raise ValueError(
                f"interpolator horizon {self.interpolator_horizon} must be equal to the "
                f"last interpolation step+1=i_N=i_{self.num_timesteps - 1}={last_d_to_i_tstep + 1}"
            )
        if interpolator_artificial_steps_run_id is not None:
            self.interpolator_artificial_steps = get_checkpoint_from_path_or_wandb(
                wandb_run_id=interpolator_artificial_steps_run_id, reload_kwargs=dict(epoch="best")
            )["model"]
            freeze_model(self.interpolator_artificial_steps)
            self.log_text.info(
                f" Will use a separate interpolator for the artificial steps: {interpolator_artificial_steps_run_id}"
            )
            assert self.interpolator_artificial_steps.hparams.get("parametric") in [
                None,
                False,
            ], "parametric interpolator2 not supported"
        else:
            self.interpolator_artificial_steps = None

        raise_error_if_invalid_value(
            consistency_strategy,
            ["ema", "net", "net-detach"],
            name="consistency_strategy",
        )

    def _interpolate(
        self,
        initial_condition: Tensor,
        x_last: Tensor,
        t: Tensor,
        num_predictions: int = 1,
        **kwargs,
    ):
        # interpolator networks uses time in [1, horizon-1]
        assert (0 < t).all() and (
            t < self.interpolator_horizon
        ).all(), f"interpolate time must be in (0, {self.interpolator_horizon}), got {t}"
        # select condition data to be consistent with the interpolator training data
        if self.hparams.hack_for_imprecise_interpolation:
            x_last = torch.cat([initial_condition[:, :1], x_last], dim=1)
        interpolator_inputs = torch.cat([initial_condition, x_last], dim=1)
        with torch.no_grad():
            interpolator_outputs = self.interpolator.predict_packed(interpolator_inputs, time=t, **kwargs)
        interpolator_outputs = interpolator_outputs["preds"]
        if self.hparams.hack_for_imprecise_interpolation:
            interpolator_outputs = torch.cat([initial_condition[:, :1], interpolator_outputs], dim=1)
        return interpolator_outputs

    def p_losses(self, input_dynamics: Tensor, xt_last: Tensor, verbose=False, **kwargs):
        r"""

        Args:
            input_dynamics: the initial condition data  (time = 0)
            xt_last: the start/target data  (time = horizon)
            t: the time step of the diffusion process
        """
        batch_size = input_dynamics.shape[0]
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=self.device, dtype=torch.long)
        criterion = self.criterion["preds"]

        # x_t is what multi-horizon exp passes as targets, and xt_last is the last timestep of the data dynamics
        # check that the time step is valid (between 0 and horizon-1)
        # assert torch.all(t >= 0) and torch.all(t <= self.num_timesteps-1), f'invalid time step {t}'
        lam1 = self.hparams.lambda_reconstruction
        lam2 = self.hparams.lambda_reconstruction2
        lam3 = self.hparams.lambda_reverse_diffusion
        lam_cy = self.hparams.lambda_consistency

        x_t = input_dynamics.clone()
        # since we do not need to interpolate xt_0, we can skip all batches where t=0
        t_nonzero = t > 0
        if t_nonzero.any():
            # sample one interpolation prediction
            x_interpolated = self._q_sample(
                x_end=input_dynamics,
                x0=xt_last,
                t=t,
                batch_mask=t_nonzero,
                num_predictions=1,
                **kwargs,
            )
            # Now, simply concatenate the inital_conditions for t=0 with the interpolated data for t>0
            x_t[t_nonzero] = x_interpolated.to(x_t.dtype)
            # assert torch.all(x_t[t == 0] == condition[t == 0]), f'x_t[t == 0] != condition[t == 0]'

        # Train the forward predictions (i.e. predict xt_last from xt_t)
        xt_last_pred = self.predict_x_last(initial_condition=input_dynamics, x_t=x_t, t=t, **kwargs)
        loss_forward = criterion(xt_last_pred, xt_last)

        t2 = t + 1  # t2 is the next time step, between 1 and T
        tnot_last = t2 <= self.num_timesteps - 1  # tnot_last is True for t < T
        calc_t2 = tnot_last.any()
        get_xh2_no_ema = calc_t2 and (lam2 > 0 or lam_cy > 0)
        if get_xh2_no_ema:
            # train the predictions using x0 = xlast = forward_pred(condition, t=0)
            # x_last_denoised2 = self.predict_x_last(condition=condition, x_t=condition, t=torch.zeros_like(t))
            # simulate the diffusion process for a single step, where the x_last=forward_pred(condition, t) prediction
            # is used to get the interpolated x_t+1 = interpolate(condition, x_last, t+1)
            x_interpolated2 = self._q_sample(
                x_end=input_dynamics,
                x0=xt_last_pred.detach() if self.hparams.reconstruction2_detach_x_last else xt_last_pred,
                t=t2,
                batch_mask=tnot_last,
                num_predictions=1,
                **kwargs,
            )
            kwargs = {k: v[tnot_last] if torch.is_tensor(v) else v for k, v in kwargs.items()}
            if self.hparams.consistency_strategy == "ema" and lam2 <= 0:
                pass  # x_last_pred2 is computed below using the EMA
            else:
                x_last_pred2 = self.predict_x_last(
                    initial_condition=input_dynamics[tnot_last], x_t=x_interpolated2, t=t2[tnot_last], **kwargs
                )
        if lam2 > 0 and calc_t2:
            loss_forward2 = criterion(x_last_pred2, xt_last[tnot_last])
        else:
            loss_forward2 = 0.0

        if lam_cy > 0 and calc_t2:
            # train the consistency predictions
            xh1 = xt_last_pred[tnot_last]
            if self.hparams.consistency_strategy == "net":
                xh2 = x_last_pred2
            elif self.hparams.consistency_strategy == "net-detach":
                xh2 = x_last_pred2.detach()
            elif self.hparams.consistency_strategy == "ema":
                dropout_state = torch.get_rng_state()
                with self.ema_scope(condition=True):
                    torch.set_rng_state(dropout_state)
                    xh2 = self.predict_x_last(
                        initial_condition=input_dynamics[tnot_last], x_t=x_interpolated2, t=t2[tnot_last], **kwargs
                    ).detach()
            else:
                raise ValueError(f"invalid consistency_strategy {self.hparams.consistency_strategy}")
            loss_consistency = criterion(xh1, xh2)
        else:
            loss_consistency = 0.0

        if lam3 > 0 and self.hparams.penalize_reverse_diffusion_steps == "last":
            # train the sampling predictions (i.e. predict x_start from x_end)
            x_last_from_start, _, _ = self.sample_loop(
                initial_condition=input_dynamics, log_every_t=None, num_predictions=1, **kwargs
            )
            loss_rev_diffusion = criterion(x_last_from_start, xt_last)
        else:
            loss_rev_diffusion = 0.0

        loss = lam1 * loss_forward + lam2 * loss_forward2 + lam_cy * loss_consistency + lam3 * loss_rev_diffusion

        log_prefix = "train" if self.training else "val"
        loss_dict = {
            "loss": loss,
            f"{log_prefix}/loss_forward": loss_forward,
            f"{log_prefix}/loss_forward2": loss_forward2,
        }
        return loss_dict
