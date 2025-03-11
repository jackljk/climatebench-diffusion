from typing import Mapping, Optional

import numpy as np
import torch

from src.ace_inference.core.data_loading.data_typing import VariableMetadata
from src.ace_inference.core.wandb import WandB


wandb = WandB.get_instance()


class SnapshotAggregator:
    """
    An aggregator that records the first sample of the last batch of data.
    > The way it works is that it gets called once per batch, but in the end (when using get_logs)
    it only returns information based on the last batch.
    """

    _captions = {
        "full-field": ("{name} one step full field for last sample; " "(left) generated and (right) target [{units}]"),
        "residual": (
            "{name} one step residual (prediction - previous time) for last sample; "
            "(left) generated and (right) target [{units}]"
        ),
        "error": ("{name} one step full field error (generated - target) " "for last sample [{units}]"),
    }

    def __init__(
        self,
        is_ensemble: bool,
        target_time: Optional[int] = None,
        metadata: Optional[Mapping[str, VariableMetadata]] = None,
    ):
        """
        Args:
            metadata: Mapping of variable names their metadata that will
                used in generating logged image captions.
        """
        self.is_ensemble = is_ensemble
        assert target_time is None or target_time > 0
        self.target_time = target_time  # account for 0-indexing not needed because initial condition is included
        self.target_time_in_batch = None
        if metadata is None:
            self._metadata: Mapping[str, VariableMetadata] = {}
        else:
            self._metadata = metadata

    @torch.no_grad()
    def record_batch(
        self,
        target_data: Mapping[str, torch.Tensor],
        gen_data: Mapping[str, torch.Tensor],
        target_data_norm: Mapping[str, torch.Tensor],
        gen_data_norm: Mapping[str, torch.Tensor],
        inputs_norm: Mapping[str, torch.Tensor] = None,
        loss=None,
        i_time_start: int = 0,
    ):
        data_steps = target_data_norm[list(target_data_norm.keys())[0]].shape[1]
        if self.target_time is not None:
            diff = self.target_time - i_time_start
            # target time needs to be in the batch (between i_time_start and i_time_start + data_steps)
            if diff < 0 or diff >= data_steps:
                return  # skip this batch, since it doesn't contain the target time
            else:
                self.target_time_in_batch = diff

        def to_cpu(x):
            return {k: v.cpu() for k, v in x.items()} if isinstance(x, dict) else x.cpu()

        self._target_data = to_cpu(target_data)
        self._gen_data = to_cpu(gen_data)
        self._target_data_norm = to_cpu(target_data_norm)
        self._gen_data_norm = to_cpu(gen_data_norm)
        self._inputs_norm = to_cpu(inputs_norm) if inputs_norm is not None else None
        if self.target_time is not None:
            assert (
                self.target_time_in_batch <= data_steps
            ), f"target_time={self.target_time}, time_in_batch={self.target_time_in_batch} is larger than the number of timesteps in the data={data_steps}!"

    @torch.no_grad()
    def get_logs(self, label: str):
        """
        Returns logs as can be reported to WandB.

        Args:
            label: Label to prepend to all log keys.
        """
        if self.target_time_in_batch is None and self.target_time is not None:
            return {}  # skip this batch, since it doesn't contain the target time
        image_logs = {}
        max_snapshots = 3
        for name in self._gen_data.keys():
            if name in self._gen_data_norm.keys():
                gen_data = self._gen_data_norm
                target_data = self._target_data_norm
            else:
                gen_data = self._gen_data
                target_data = self._target_data
            if self.is_ensemble:
                snapshots_pred = gen_data[name][:max_snapshots, 0]
            else:
                snapshots_pred = gen_data[name][0].unsqueeze(0)
            target_for_image = target_data[name][0]  # first sample in batch
            small_gap = torch.zeros((target_for_image.shape[-2], 2)).to(snapshots_pred.device, dtype=torch.float)
            gap = torch.zeros((target_for_image.shape[-2], 4)).to(
                snapshots_pred.device, dtype=torch.float
            )  # gap between images in wandb (so we can see them separately)
            input_for_image = (
                self._inputs_norm[name][0]
                if self._inputs_norm is not None and name in self._inputs_norm.keys()
                else None
            )
            # Select target time
            if self.target_time is not None:
                snapshots_pred = snapshots_pred[:, self.target_time_in_batch]
                target_for_image = target_for_image[self.target_time_in_batch]
                if input_for_image is not None:
                    input_for_image = input_for_image[self.target_time_in_batch]

            # Create image tensors
            image_error, image_full_field, image_residual = [], [], []
            for i in range(snapshots_pred.shape[0]):
                image_full_field += [snapshots_pred[i]]
                image_error += [snapshots_pred[i] - target_for_image]
                if input_for_image is not None:
                    image_residual += [snapshots_pred[i] - input_for_image]
                if i == snapshots_pred.shape[0] - 1:
                    image_full_field += [gap, target_for_image]
                    if input_for_image is not None:
                        image_residual += [gap, target_for_image - input_for_image]
                else:
                    image_full_field += [small_gap]
                    image_residual += [small_gap]
                    image_error += [small_gap]

            images = {}
            images["error"] = torch.cat(image_error, dim=1)
            images["full-field"] = torch.cat(image_full_field, dim=1)
            if input_for_image is not None:
                images["residual"] = torch.cat(image_residual, dim=1)

            for key, data in images.items():
                caption = self._get_caption(key, name, data)
                data = np.flip(data.cpu().numpy(), axis=-2)
                wandb_image = wandb.Image(data, caption=caption)
                image_logs[f"image-{key}/{name}"] = wandb_image

        image_logs = {f"{label}/{key}": image_logs[key] for key in image_logs}
        return image_logs

    def _get_caption(self, caption_key: str, name: str, data: torch.Tensor) -> str:
        if name in self._metadata:
            caption_name = self._metadata[name].long_name
            units = self._metadata[name].units
        else:
            caption_name, units = name, "unknown_units"
        caption = self._captions[caption_key].format(name=caption_name, units=units)
        caption += f" vmin={data.min():.4g}, vmax={data.max():.4g}."
        return caption
