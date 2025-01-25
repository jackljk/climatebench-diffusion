from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
from torch import Tensor

from src.experiment_types._base_experiment import BaseExperiment
from src.utilities.utils import (
    rrearrange,
)


class InterpolationExperiment(BaseExperiment):
    r"""Base class for all interpolation experiments."""

    def __init__(self, stack_window_to_channel_dim: bool = True, inference_val_every_n_epochs=None, **kwargs):
        super().__init__(**kwargs)
        if inference_val_every_n_epochs is not None:
            self.log_text.warning("``inference_val_every_n_epochs`` will be ignored for interpolation experiments.")
        # The following saves all the args that are passed to the constructor to self.hparams
        #   e.g. access them with self.hparams.hidden_dims
        self.save_hyperparameters(ignore=["model"])
        assert self.horizon >= 2, "horizon must be >=2 for interpolation experiments"
        if hasattr(self.model, "set_min_max_time"):
            self.model.set_min_max_time(min_time=self.horizon_range[0], max_time=self.horizon_range[-1])

    @property
    def horizon_range(self) -> List[int]:
        # h = horizon
        # We use timesteps  w-l+1, ..., w-1, w+h to predict timesteps w, ..., w+h-1
        # interpolate between step t=0 and t=horizon
        return list(np.arange(1, self.horizon))

    @property
    def true_horizon(self) -> int:
        return self.horizon

    @property
    def horizon_name(self) -> str:
        s = f"{self.true_horizon}h"
        return s

    @property
    def short_description(self) -> str:
        name = super().short_description
        name += f" (h={self.horizon_name})"
        return name

    @property
    def WANDB_LAST_SEP(self) -> str:
        return "/"  # /ipol/"

    @property
    def num_conditional_channels(self) -> int:
        """The number of channels that are used for conditioning as auxiliary inputs."""
        nc = super().num_conditional_channels
        factor = self.window + 0 + 0  # num inputs before target + num targets + num inputs after target
        return nc * factor

    def actual_num_input_channels(self, num_input_channels: int) -> int:
        if self.hparams.stack_window_to_channel_dim:
            return num_input_channels * self.window + num_input_channels
        return 2 * num_input_channels  # inputs and targets are concatenated

    def postprocess_inputs(self, inputs):
        inputs = self.pack_data(inputs, input_or_output="input")
        if self.hparams.stack_window_to_channel_dim:  # and inputs.shape[1] == self.window:
            inputs = rrearrange(inputs, "b window c lat lon -> b (window c) lat lon")
        return inputs

    @torch.inference_mode()
    def _evaluation_step(
        self,
        batch: Any,
        batch_idx: int,
        split: str,
        dataloader_idx: int = None,
        aggregators: Dict[str, Callable] = None,
        return_only_preds_and_targets: bool = False,
    ):
        no_aggregators = aggregators is None or len(aggregators.keys()) == 0
        main_data_raw = batch.pop("raw_dynamics")
        dynamics = batch["dynamics"]  # dynamics is a (b, t, c, h, w) tensor

        return_dict = dict()
        extra_kwargs = {}
        dynamical_cond = batch.pop("dynamical_condition", None)
        if dynamical_cond is not None:
            assert "condition" not in batch, "condition should not be in batch if dynamical_condition is present"
        inputs = self.get_evaluation_inputs(dynamics, split=split)
        for k, v in batch.items():
            if k != "dynamics":
                extra_kwargs[k] = self.get_ensemble_inputs(v, split=split, add_noise=False)

        for t_step in self.horizon_range:
            # dynamics[, self.window] is already the first target frame (t_step=1)
            target_time = self.window + t_step - 1
            time = torch.full((inputs.shape[0],), t_step, device=self.device, dtype=torch.long)
            if dynamical_cond is not None:
                extra_kwargs["condition"] = self.get_ensemble_inputs(
                    self.get_dynamical_condition(dynamical_cond, target_time), split=split, add_noise=False
                )
            results = self.predict(inputs, time=time, **extra_kwargs)
            preds = results["preds"]

            targets_tensor_t = main_data_raw[:, target_time, ...]
            targets = self.get_target_variants(targets_tensor_t, is_normalized=False)
            results["targets"] = targets
            results = {f"t{t_step}_{k}": v for k, v in results.items()}

            if return_only_preds_and_targets:
                return_dict[f"t{t_step}_preds"] = preds
                return_dict[f"t{t_step}_targets"] = targets
            else:
                return_dict = {**return_dict, **results}

            if no_aggregators:
                continue

            PREDS_NORMED_K = f"t{t_step}_preds_normed"
            PREDS_RAW_K = f"t{t_step}_preds"
            targets_normed = targets["targets_normed"] if targets is not None else None
            targets_raw = targets["targets"] if targets is not None else None
            aggregators[f"t{t_step}"].record_batch(
                target_data=targets_raw,
                gen_data=results[PREDS_RAW_K],
                target_data_norm=targets_normed,
                gen_data_norm=results[PREDS_NORMED_K],
            )

        return return_dict

    def get_dynamical_condition(
        self, dynamical_condition: Optional[Tensor], target_time: Union[int, Tensor]
    ) -> Tensor:
        if dynamical_condition is not None:
            if isinstance(target_time, int):
                return dynamical_condition[:, target_time, ...]
            else:
                return dynamical_condition[torch.arange(dynamical_condition.shape[0]), target_time.long(), ...]
        return None

    def get_inputs_from_dynamics(self, dynamics: Tensor, **kwargs) -> Tensor:
        """Get the inputs from the dynamics tensor.
        Since we are doing interpolation, this consists of the first window frames plus the last frame.
        """
        past_steps = dynamics[:, : self.window, ...]  # (b, window, c, lat, lon) at time 0
        last_step = dynamics[:, -1:, ...]  # (b, c, lat, lon) at time t=window+horizon
        past_steps = self.postprocess_inputs(past_steps)
        last_step = self.postprocess_inputs(last_step)
        inputs = torch.cat([past_steps, last_step], dim=1)  # (b, window*c + c, lat, lon)
        return inputs

    def get_evaluation_inputs(self, dynamics: Tensor, split: str, **kwargs) -> Tensor:
        inputs = self.get_inputs_from_dynamics(dynamics)
        inputs = self.get_ensemble_inputs(inputs, split)
        return inputs

    # --------------------------------- Training
    def get_loss(self, batch: Any, optimizer_idx: int = 0) -> Tensor:
        r"""Compute the loss for the given batch."""
        dynamics = batch["dynamics"]  # dynamics is a (b, t, c, h, w) tensor
        inputs = self.get_inputs_from_dynamics(dynamics)  # (b, c, h, w) at time 0
        b = dynamics.shape[0]

        possible_times = torch.tensor(self.horizon_range, device=self.device, dtype=torch.long)  # (h,)
        # take random choice of time
        t = possible_times[torch.randint(len(possible_times), (b,), device=self.device, dtype=torch.long)]  # (b,)
        target_time = self.window + t - 1
        # t = torch.randint(start_t, max_t, (b,), device=self.device, dtype=torch.long)  # (b,)
        targets = dynamics[torch.arange(b), target_time, ...]  # (b, c, h, w)
        targets = self.pack_data(targets, input_or_output="output")
        # We use timesteps  w-l+1, ..., w-1, w+h to predict timesteps w, ..., w+h-1
        # so t=0 corresponds to interpolating w, t=1 to w+1, ..., t=h-1 to w+h-1

        loss = self.model.get_loss(
            inputs=inputs,
            targets=targets,
            condition=self.get_dynamical_condition(batch.pop("dynamical_condition", None), target_time=target_time),
            time=t,
            **{k: v for k, v in batch.items() if k != "dynamics"},
        )  # function of BaseModel or BaseDiffusion classes
        return loss


class NextStepInterpolationExperiment(InterpolationExperiment):
    r"""Very similar to above, but instead of conditioning on x_0 and x_h to interpolate x_i,
    it conditions on x_{i-1} and x_{h} to interpolate x_i, where i = 1, ..., h-1."""

    def __init__(
        self, n_autoregressive_train_steps: int = 0, autoregressive_loss_weights: List[float] = None, **kwargs
    ):
        super().__init__(**kwargs)
        if autoregressive_loss_weights is None:
            autoregressive_loss_weights = [1.0] * n_autoregressive_train_steps
        else:
            assert len(autoregressive_loss_weights) == n_autoregressive_train_steps
        self.autoregressive_loss_weights = autoregressive_loss_weights

    def get_epoch_aggregators(self, split: str, dataloader_idx: int = None) -> dict:
        aggs = super().get_epoch_aggregators(split, dataloader_idx)
        # Add autoregressive aggregators
        aggs2 = super().get_epoch_aggregators(split, dataloader_idx)
        aggs2.pop("t1")  # remove the first timestep since it does not make sense to have autoregressive predictions
        aggs.update({f"{k}_ar": v for k, v in aggs2.items()})
        return aggs

    @torch.inference_mode()
    def _evaluation_step(
        self,
        batch: Any,
        batch_idx: int,
        split: str,
        dataloader_idx: int = None,
        aggregators: Dict[str, Callable] = None,
        return_only_preds_and_targets: bool = False,
    ):
        no_aggregators = aggregators is None or len(aggregators.keys()) == 0
        main_data_raw = batch.pop("raw")
        dynamics = batch["dynamics"]  # dynamics is a (b, t, c, h, w) tensor
        return_dict = dict()

        extra_kwargs = {}
        for k, v in batch.items():
            if k != "dynamics":
                extra_kwargs[k] = self.get_ensemble_inputs(v, split=split, add_noise=False)

        autoregressive_inputs = None
        last_step_ens = self.get_ensemble_inputs(dynamics[:, -1, ...], split=split, add_noise=False)
        last_step_ens = self.postprocess_inputs(last_step_ens)
        for t_step in self.horizon_range:
            # dynamics[, self.window] is already the first target frame (t_step=1)
            target_time = self.window + t_step - 1
            inputs = self.get_evaluation_inputs(dynamics, split=split, time=t_step)
            time = torch.full((inputs.shape[0],), t_step, device=self.device, dtype=torch.long)

            results = self.predict(inputs, time=time, **extra_kwargs)

            targets_tensor_t = main_data_raw[:, target_time, ...]
            targets = self.get_target_variants(targets_tensor_t, is_normalized=False)
            results["targets"] = targets

            if autoregressive_inputs is not None:
                autoregressive_inputs = rrearrange(autoregressive_inputs, "N B ... -> (N B) ...")
                inputs_ar = torch.cat([autoregressive_inputs, last_step_ens], dim=1)
                results_ar = self.predict(inputs_ar, time=time, num_predictions=1, **extra_kwargs)
                results["ar_preds"] = results_ar["preds"]
                results["ar_targets"] = targets

                aggregators[f"t{t_step}_ar"].record_batch(
                    target_data=targets["targets"],
                    gen_data=results_ar["preds"],
                    target_data_norm=targets["targets_normed"],
                    gen_data_norm=results_ar["preds_normed"],
                )

            # Retrieve preds and update autoregressive inputs
            autoregressive_inputs = preds = results["preds"]

            results = {f"t{t_step}_{k}": v for k, v in results.items()}
            if return_only_preds_and_targets:
                return_dict[f"t{t_step}_preds"] = preds
                return_dict[f"t{t_step}_targets"] = targets
                if autoregressive_inputs is not None:
                    return_dict[f"t{t_step}_ar_preds"] = results[f"t{t_step}_ar_preds"]
                    return_dict[f"t{t_step}_ar_targets"] = results[f"t{t_step}_ar_targets"]
            else:
                return_dict = {**return_dict, **results}

            if no_aggregators:
                continue

            PREDS_NORMED_K = f"t{t_step}_preds_normed"
            PREDS_RAW_K = f"t{t_step}_preds"
            targets_normed = targets["targets_normed"] if targets is not None else None
            targets_raw = targets["targets"] if targets is not None else None
            aggregators[f"t{t_step}"].record_batch(
                target_data=targets_raw,
                gen_data=results[PREDS_RAW_K],
                target_data_norm=targets_normed,
                gen_data_norm=results[PREDS_NORMED_K],
            )

        return return_dict

    def get_inputs_from_dynamics(self, dynamics: Tensor, time: Union[Tensor, int]) -> Tensor:
        """Get the inputs from the dynamics tensor.
        Since we are doing interpolation, this consists of the first window frames plus the last frame.
        """
        assert dynamics.shape[1] == self.window + self.horizon, "dynamics tensor must have shape (b, t, c, h, w)"
        input_time = self.window + time - 2
        if torch.is_tensor(time):
            past_steps = dynamics[torch.arange(dynamics.shape[0]), input_time, ...]  # (b, c, h, w) at time 0
        else:
            assert 1 <= time <= self.horizon - 1, f"time must be in [1, {self.horizon - 1}]"
            past_steps = dynamics[:, input_time, ...]
        past_steps = self.postprocess_inputs(past_steps)

        inputs = self.concat_future_conditioning(past_steps, dynamics)
        return inputs

    def concat_future_conditioning(self, past_inputs: Tensor, dynamics: Tensor) -> Tensor:
        last_step = dynamics[:, -1, ...]  # (b, c, lat, lon) at time t=window+horizon
        last_step = self.postprocess_inputs(last_step)
        inputs = torch.cat([past_inputs, last_step], dim=1)
        return inputs

    def get_evaluation_inputs(self, dynamics: Tensor, split: str, **kwargs) -> Tensor:
        inputs = self.get_inputs_from_dynamics(dynamics, **kwargs)
        inputs = self.get_ensemble_inputs(inputs, split)
        return inputs

    # --------------------------------- Training
    def get_loss(self, batch: Any, optimizer_idx: int = 0) -> Tensor:
        r"""Compute the loss for the given batch."""
        dynamics = batch["dynamics"]  # dynamics is a (b, t, c, h, w) tensor
        b = dynamics.shape[0]

        possible_times = torch.tensor(self.horizon_range, device=self.device, dtype=torch.long)  # (h,)
        # take random choice of time
        t = possible_times[torch.randint(len(possible_times), (b,), device=self.device, dtype=torch.long)]  # (b,)
        targets = dynamics[torch.arange(b), self.window + t - 1, ...]  # (b, c, h, w), zero-based indexing
        inputs = self.get_inputs_from_dynamics(dynamics, time=t)

        loss, preds = self.model.get_loss(
            inputs=inputs,
            targets=targets,
            time=t,
            **{k: v for k, v in batch.items() if k != "dynamics"},
            return_predictions=True,
        )  # function of BaseModel or BaseDiffusion classes
        if self.hparams.n_autoregressive_train_steps > 0:
            for ar_step in range(1, self.hparams.n_autoregressive_train_steps + 1):
                ar_time = t + ar_step
                ar_time_valid = (ar_time <= self.horizon - 1).long()
                ar_inputs = self.concat_future_conditioning(preds, dynamics)[ar_time_valid, ...]
                ar_targets = dynamics[ar_time_valid, self.window + ar_time - 1, ...]
                ar_loss, preds = self.model.get_loss(
                    inputs=ar_inputs,
                    targets=ar_targets,
                    time=ar_time,
                    return_predictions=True,
                    **{k: v for k, v in batch.items() if k != "dynamics"},
                )
                loss += self.autoregressive_loss_weights[ar_step - 1] * ar_loss
        return loss
