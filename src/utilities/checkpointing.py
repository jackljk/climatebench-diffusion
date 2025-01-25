import os
import re
from typing import List, Optional

import numpy as np
import torch
from omegaconf import DictConfig, ListConfig

from src.utilities.utils import get_logger


log = get_logger(__name__)

try:
    torch.serialization.add_safe_globals([ListConfig])
except AttributeError:
    log.warning("torch.serialization.add_safe_globals([ListConfig]) not supported in this version of PyTorch")


def get_local_ckpt_path(
    config: DictConfig,
    wandb_run,  #: wandb.apis.public.Run,
    ckpt_filename: str = "last.ckpt",
    throw_error_if_local_not_found: bool = False,
) -> Optional[str]:
    potential_dirs = [
        config.ckpt_dir,
        os.path.join(config.work_dir.replace("-test", ""), "checkpoints"),
        os.path.join(os.getcwd(), "results", "checkpoints"),
    ]
    for callback_k in config.get("callbacks", {}).keys():
        if "checkpoint" in callback_k and config.callbacks[callback_k] is not None:
            if config.callbacks[callback_k].get("dirpath", None) is not None:
                potential_dirs.append(config.callbacks[callback_k].dirpath)

    for local_dir in potential_dirs:
        log.info(f"Checking {local_dir}. {os.path.exists(local_dir)=}")
        if not os.path.exists(local_dir):
            continue
        if wandb_run.id not in local_dir:
            local_dir = os.path.join(local_dir, wandb_run.id)
            if not os.path.exists(local_dir):
                continue
        ckpt_files = [f for f in os.listdir(local_dir) if f.endswith(".ckpt")]
        if ckpt_filename == "last.ckpt":
            ckpt_files = [f for f in ckpt_files if "last" in f]
            if len(ckpt_files) == 0:
                continue
            elif len(ckpt_files) == 1:
                latest_ckpt_file = ckpt_files[0]
            else:
                # Get their epoch numbers from inside the file
                # epochs = [torch.load(os.path.join(local_dir, f), weights_only=True)["epoch"] for f in ckpt_files]
                epochs = [torch.load(os.path.join(local_dir, f))["epoch"] for f in ckpt_files]
                # Find the ckpt file with the latest epoch
                latest_ckpt_file = ckpt_files[np.argmax(epochs)]
                log.info(
                    f"Found multiple last-v<V>.ckpt files. Using the one with the highest epoch: {latest_ckpt_file}. ckpt_to_epoch: {dict(zip(ckpt_files, epochs))}"
                )
            return os.path.join(local_dir, latest_ckpt_file)

        elif ckpt_filename in ["earliest_epoch", "latest_epoch", "earliest_epoch_any", "latest_epoch_any"]:
            # Find the earliest epoch ckpt file
            # Ckpt dir has files like:
            # - Kolmogorov-H32-ERDM-1.0t-edm-0.002-80.0sigma_128x128-Vl_epoch103_seed11.ckpt
            # - Kolmogorov-H32-ERDM-1.0t-edm-0.002-80.0sigma_128x128-Vl_epoch033_seed11.ckpt
            if ckpt_filename in ["earliest_epoch_any", "latest_epoch_any"]:
                ckpt_files = [f for f in ckpt_files if "epoch" in f]
            else:
                ckpt_files = [f for f in ckpt_files if "epoch" in f and "epochepoch=" not in f]
            if len(ckpt_files) == 0:
                continue

            # Function to extract the epoch number from the filename
            def get_epoch_number(filename):
                if "_any" in ckpt_filename:
                    filename = filename.replace("epochepoch=", "epoch")  # Fix for a bug in the filename
                match = re.search(r"_epoch(\d+)_", filename)
                return int(match.group(1))

            # Find the ckpt file with the earliest epoch
            min_or_max = min if ckpt_filename == "earliest_epoch" else max
            earliest_ckpt_file = min_or_max(ckpt_files, key=lambda f: get_epoch_number(f))
            log.info(f"For ckpt_filename={ckpt_filename}, found ckpt file: {earliest_ckpt_file} in {local_dir}")
            return os.path.join(local_dir, earliest_ckpt_file)

        ckpt_path = os.path.join(local_dir, ckpt_filename)
        if os.path.exists(ckpt_path):
            return ckpt_path
        else:
            log.warning(f"{local_dir} exists but could not find {ckpt_filename=}. Files in dir: {ckpt_files}.")
    if ckpt_filename in ["earliest_epoch", "latest_epoch", "earliest_epoch_any", "latest_epoch_any"]:
        raise NotImplementedError("Not implemented")
    if throw_error_if_local_not_found:
        raise FileNotFoundError(
            f"Could not find ckpt file {ckpt_filename} in any of the potential dirs: {potential_dirs}"
        )
    return None


def load_state_dict_and_analyze_weight_changes(
    model, state_dict, strict=False, exclude_keys: List[str] = None, num_examples=8
):
    """
    Analyzes weight changes after loading partial state dict.
    Args:
        model: PyTorch model
        state_dict: State dict to load
        strict: Whether to strictly enforce that the keys in state_dict match the keys returned by model.state_dict()
        exclude_keys: List of keys to exclude from the state_dict (won't be loaded!)
        num_examples: Number of first/last layers to show
    """
    exclude_keys = exclude_keys or []
    orig_state = {k: v.clone() for k, v in model.state_dict().items()}
    unloaded_keys = [k for k in state_dict.keys() if k not in orig_state or k in exclude_keys]
    if exclude_keys:
        assert not strict, "Cannot exclude keys when strict=True"
        state_dict = {k: v for k, v in state_dict.items() if k not in exclude_keys}
    model.load_state_dict(state_dict, strict=strict)

    changed = []
    unchanged = []
    total_params = changed_params = 0

    for key, new_val in model.state_dict().items():
        if key in orig_state:
            params = new_val.numel()
            total_params += params
            if not torch.equal(orig_state[key], new_val):
                num_changed = (orig_state[key] != new_val).sum().item()
                changed_params += num_changed
                changed.append(f"{key}: {num_changed}/{params} params ({num_changed / params * 100:.1f}%)")
            else:
                unchanged.append(key)

    # If (almost) fully changed, no unloaded keys then no need to print the analysis
    if round(changed_params / total_params * 100) == 100 and len(unloaded_keys) == 0:
        return

    output = [
        f"Total reloaded parameters: {changed_params:,}/{total_params:,} ({changed_params / total_params * 100:.1f}%)"
    ]

    if changed:
        output.append(f"Changed (i.e. reloaded) layers ({len(changed)} total; showing params changed/total):")
        output.extend(changed[:num_examples])
        remaining = len(changed) - 2 * num_examples
        if remaining > 0:
            output.append(f"... and {remaining} layers in between ...")
        if len(changed) > num_examples:
            output.extend(changed[-num_examples:])
    else:
        output.append("No layers were changed")

    if unchanged:
        output.append(f"\nUnchanged layers (layers without any change; {len(unchanged)} total):")
        output.extend(unchanged[:num_examples])
        remaining = len(unchanged) - 2 * num_examples
        if remaining > 0:
            output.append(f"... and {remaining} layers in between ...")
        if len(unchanged) > num_examples:
            output.extend(unchanged[-num_examples:])
    else:
        output.append("No layers were unchanged")

    if unloaded_keys:
        output.append(f"\nUnloaded keys (layers in the state dict but not in the model; {len(unloaded_keys)} total):")
        output.extend(unloaded_keys[:num_examples])
        remaining = len(unloaded_keys) - 2 * num_examples
        if remaining > 0:
            output.append(f"... and {remaining} keys in between ...")
        if len(unloaded_keys) > num_examples:
            output.extend(unloaded_keys[-num_examples:])
    else:
        output.append("No unloaded keys")

    log.info("\n".join(output))


def analyze_weight_changes_concise(model, state_dict, n_examples: int = 4):
    """Analyzes weight changes after loading partial state dict."""
    orig_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(state_dict, strict=False)

    changed = []
    unchanged = []
    total_params = changed_params = 0

    for key, new_val in model.state_dict().items():
        if key in orig_state:
            params = new_val.numel()
            total_params += params
            if not torch.equal(orig_state[key], new_val):
                num_changed = (orig_state[key] != new_val).sum().item()
                changed_params += num_changed
                changed.append(f"{key}: {num_changed}/{params} params ({num_changed / params * 100:.1f}%)")
            else:
                unchanged.append(key)

    log.info(
        f"Total changed parameters: {changed_params:,}/{total_params:,} ({changed_params / total_params * 100:.1f}%)"
    )
    log.info(
        "Changed layers (showing params changed/total): "
        + ";  ".join(changed[:n_examples])
        + (f"\n... and {len(changed) - n_examples} more layers" if len(changed) > n_examples else "")
    )
    log.info(
        f"Unchanged layers (Layers without any change; {len(unchanged)} total): "
        + ";  ".join(unchanged[:n_examples])
        + (f"\n... and {len(unchanged) - n_examples} more layers" if len(unchanged) > n_examples else "")
    )
