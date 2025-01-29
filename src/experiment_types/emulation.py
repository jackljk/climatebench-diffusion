from __future__ import annotations

from typing import Any, Callable, Dict, List

import torch
from torch import Tensor

from src.diffusion._base_diffusion import BaseDiffusion
from src.experiment_types._base_experiment import BaseExperiment
from src.utilities.utils import (
    rrearrange,
    torch_to_numpy,
)


class EmulationExperiment(BaseExperiment):
    def __init__(
        self,
        stack_window_to_channel_dim: bool = True,
        return_outputs_at_evaluation: str | bool = False,  # can be "all", "preds_only", True, False
        **kwargs,
    ):
        super().__init__(**kwargs)
        # The following saves all the args that are passed to the constructor to self.hparams
        #   e.g. access them with self.hparams.stack_window_to_channel_dim
        self.save_hyperparameters(ignore=["model"])

    def actual_num_input_channels(self, num_input_channels: int) -> int:
        # if we use the inputs as conditioning, and use an output-shaped input (e.g. for DDPM),
        # we need to use the output channels here!
        is_standard_diffusion = self.is_diffusion_model
        if is_standard_diffusion:
            return self.actual_num_output_channels(self.dims["output"])
        if self.hparams.stack_window_to_channel_dim:
            return num_input_channels * self.window
        return num_input_channels

    @property
    def target_key(self) -> str:
        return "targets"

    @property
    def main_data_keys(self) -> List[str]:
        return ["inputs", self.target_key]

    @property
    def main_data_keys_val(self) -> List[str]:
        return self.main_data_keys

    @torch.inference_mode()
    def _evaluation_step(
        self,
        batch: Any,
        batch_idx: int,
        split: str,
        dataloader_idx: int = None,
        return_outputs: bool | str = None,
        aggregators: Dict[str, Callable] = None,
    ):
        return_dict = dict()
        return_outputs = return_outputs or self.hparams.return_outputs_at_evaluation

        # pop metadata from batch since not needed for evaluation
        metadata = batch.pop("metadata", None)
        raw_targets = batch.pop("raw_targets", None)  # Unnormalized (raw scale) data, used to compute targets
        if isinstance(self.model, BaseDiffusion) and split == "val":
            # log validation loss
            loss = self.get_loss(batch)
            if isinstance(loss, dict):
                # add split/ prefix if not already there
                log_dict = {f"{split}/{k}" if not k.startswith(split) else k: float(v) for k, v in loss.items()}
            elif torch.is_tensor(loss):
                log_dict = {f"{split}/loss": float(loss)}
            self.log_dict(log_dict, on_step=False, on_epoch=True)

        _ = batch.pop("targets", None)  # Remove normalized targets if present, not needed any more
        # Get predictions
        targets = self.get_target_variants(raw_targets, is_normalized=False)
        inputs = self.transform_inputs(batch.pop("inputs"), split=split, ensemble=True)
        results = self.predict(inputs, **batch)

        # Return outputs and log metrics
        targets_raw, targets_normed = targets.pop("targets"), targets.pop("targets_normed")
        preds_raw, preds_normed = results.pop("preds"), results.pop("preds_normed")
        if return_outputs in [True, "all", "preds_only"]:
            return_dict["preds_normed"] = torch_to_numpy(preds_normed)
        if return_outputs in [True, "all"]:
            return_dict["targets_normed"] = torch_to_numpy(targets_normed)

        if return_outputs == "all":
            return_dict["targets"] = torch_to_numpy(targets)
            return_dict["preds"] = torch_to_numpy(preds_raw)
            # add remaining outputs
            return_dict.update({k: torch_to_numpy(v) for k, v in results.items()})

        # Compute metrics
        for agg_name, agg in aggregators.items():
            agg.record_batch(
                target_data=targets_raw,
                gen_data=preds_raw,
                target_data_norm=targets_normed,
                gen_data_norm=preds_normed,
                metadata=metadata,
            )
        return return_dict

    def transform_inputs(self, inputs: Tensor, ensemble: bool, **kwargs) -> Tensor:
        inputs = self.pack_data(inputs, input_or_output="input")
        if self.hparams.stack_window_to_channel_dim:  # and inputs.shape[1] == self.window:
            inputs = rrearrange(inputs, "b window c lat lon -> b (window c) lat lon")
        inputs = self.get_ensemble_inputs(inputs, **kwargs) if ensemble else inputs
        return inputs

    def get_loss(self, batch: Any) -> Tensor:
        r"""Compute the loss for the given batch."""
        split = "train" if self.training else "val"
        # Both inputs and targets are normalized
        inputs = self.transform_inputs(batch["inputs"], split=split, ensemble=False)
        targets = batch["targets"]
        # Uncomment below if targets is a dict of variable to variable target data
        targets = self.pack_data(targets, input_or_output="output")

        # Remove metadata from batch as not needed for loss computation
        batch.pop("metadata", None)

        extra_kwargs = {k: v for k, v in batch.items() if k not in ["inputs", "targets"]}
        loss = self.model.get_loss(inputs=inputs, targets=targets, **extra_kwargs)
        return loss
