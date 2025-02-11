import os
import re
from typing import Any, Dict, List, Optional, Union

import hydra
import numpy as np
import pytorch_lightning
import torch
from omegaconf import DictConfig, ListConfig

import src.utilities.wandb_api as wandb_api
from src.utilities.utils import get_logger, rename_state_dict_keys_and_save


log = get_logger(__name__)

try:
    torch.serialization.add_safe_globals([ListConfig])
except AttributeError:
    log.warning("torch.serialization.add_safe_globals([ListConfig]) not supported in this version of PyTorch")


def get_lightning_module(config: DictConfig, **kwargs):
    r"""Get the ML model, a subclass of :class:`~src.experiment_types._base_experiment.BaseExperiment`, as defined by the key value pairs in ``config.model``.

    Args:
        config (DictConfig): A OmegaConf config (e.g. produced by hydra <config>.yaml file parsing)
        **kwargs: Any additional keyword arguments for the model class (overrides any key in config, if present)

    Returns:
        BaseExperiment:
            The lightning module that you can directly use to train with pytorch-lightning

    Examples:

    .. code-block:: python

        from src.utilities.config_utils import get_config_from_hydra_compose_overrides

        config_mlp = get_config_from_hydra_compose_overrides(overrides=['model=mlp'])
        mlp_model = get_model(config_mlp)

        # Get a prediction for a (B, S, C) shaped input
        random_mlp_input = torch.randn(1, 100, 5)
        random_prediction = mlp_model.predict(random_mlp_input)
    """
    model = hydra.utils.instantiate(
        config.module,
        model_config=config.model,
        datamodule_config=config.datamodule,
        diffusion_config=config.get("diffusion", default_value=None),
        _recursive_=False,
        **kwargs,
    )

    return model


def get_datamodule(config: DictConfig):
    r"""Get the datamodule, as defined by the key value pairs in ``config.datamodule``. A datamodule defines the data-loading logic as well as data related (hyper-)parameters like the batch size, number of workers, etc.

    Args:
        config (DictConfig): A OmegaConf config (e.g. produced by hydra <config>.yaml file parsing)

    Returns:
        Base_DataModule:
            A datamodule that you can directly use to train pytorch-lightning models

    Examples:

    .. code-block:: python

        from src.utilities.config_utils import get_config_from_hydra_compose_overrides

        cfg = get_config_from_hydra_compose_overrides(overrides=['datamodule=icosahedron', 'datamodule.order=5'])
        ico_dm = get_datamodule(cfg)
    """
    data_module = hydra.utils.instantiate(
        config.datamodule,
        _recursive_=False,
        model_config=config.model,
    )
    return data_module


def get_model_and_data(config: DictConfig):
    r"""Get the model and datamodule. This is a convenience function that wraps around :meth:`get_model` and :meth:`get_datamodule`.

    Args:
        config (DictConfig): A OmegaConf config (e.g. produced by hydra <config>.yaml file parsing)

    Returns:
        (BaseExperiment, Base_DataModule): A tuple of (module, datamodule), that you can directly use to train with pytorch-lightning

    Examples:

    .. code-block:: python

        from src.utilities.config_utils import get_config_from_hydra_compose_overrides

        cfg = get_config_from_hydra_compose_overrides(overrides=['datamodule=icosahedron', 'model=mlp'])
        mlp_model, icosahedron_data = get_model_and_data(cfg)

        # Use the data from datamodule (its ``train_dataloader()``), to train the model for 10 epochs
        trainer = pl.Trainer(max_epochs=10, devices=1)
        trainer.fit(model=model, datamodule=icosahedron_data)

    """
    data_module = get_datamodule(config)
    model = get_lightning_module(config)
    if config.module.get("torch_compile") == "module":
        log.info("Compiling LightningModule with torch.compile()...")
        model = torch.compile(model)
    return model, data_module


def reload_model_from_config_and_ckpt(
    config: DictConfig,
    model_path: str,
    device: Optional[torch.device] = None,
    also_datamodule: bool = True,
    also_ckpt: bool = False,
    model: pytorch_lightning.LightningModule = None,
    reload_strict: bool = False,
    exclude_state_dict_keys: List[str] = None,
    print_name: str = "",
) -> Dict[str, Any]:
    r"""Load a model as defined by ``config.model`` and reload its weights from ``model_path``.

    Args:
        config (DictConfig): The config to use to reload the model
        model_path (str): The path to the model checkpoint (its weights)
        device (torch.device): The device to load the model on. Defaults to 'cuda' if available, else 'cpu'.
        also_datamodule (bool): If True, also reload the datamodule from the config. Defaults to True.
        also_ckpt (bool): If True, also returns the checkpoint from ``model_path``. Defaults to False.
        model (LightningModule): If provided, the model to reload the weights into. If None, a new model is instantiated.
        reload_strict (bool): If True, the model weights are loaded strictly (i.e. all keys must match). Defaults to False.
        exclude_state_dict_keys (List[str]): A list of keys to exclude from the state_dict when loading the model. Defaults to None.
        print_name (str): A string to print when reloading the model. Defaults to "".

    Returns:
        BaseModel: The reloaded model if load_datamodule is ``False``, otherwise a tuple of (reloaded-model, datamodule)

    Examples:

    .. code-block:: python

        # If you used wandb to save the model, you can use the following to reload it
        from src.utilities.wandb_api import load_hydra_config_from_wandb

        run_path = ENTITY/PROJECT/RUN_ID   # wandb run id (you can find it on the wandb URL after runs/, e.g. 1f5ehvll)
        config = load_hydra_config_from_wandb(run_path, override_kwargs=['datamodule.num_workers=4', 'trainer.gpus=-1'])

        model, datamodule = reload_model_from_config_and_ckpt(config, model_path, load_datamodule=True)

        # Test the reloaded model
        trainer = hydra.utils.instantiate(config.trainer, _recursive_=False)
        trainer.test(model=model, datamodule=datamodule)

    """
    if model is None:
        model_provided = False
        model, data_module = get_model_and_data(config) if also_datamodule else (get_lightning_module(config), None)
    else:
        model_provided = True
        assert not also_datamodule, "If model is provided, also_datamodule must be False"
        data_module = None
    # Reload model
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_state = torch.load(model_path, map_location=device, weights_only=False)
    # rename weights (sometimes needed for backwards compatibility)
    state_dict = rename_state_dict_keys_and_save(model_state, model_path, model)
    # Reload weights
    state_dict = load_state_dict_and_analyze_weight_changes(
        model, state_dict, strict=reload_strict, exclude_keys=exclude_state_dict_keys
    )
    # Using the returned state_dict, ensures that
    # we don't pass on any excluded keys (e.g. for freezing all but those layers)

    to_return = {
        "model": model,
        "datamodule": data_module,
        "state_dict": state_dict,
        "epoch": model_state["epoch"],
        "global_step": model_state["global_step"],
        "wandb": model_state.get("wandb", None),
    }
    file_size = os.path.getsize(model_path)
    print_name = f"{print_name}: " if print_name else ""
    str_to_print = (
        f"Reloaded {print_name}{model_path}" + (" into provided model" if model_provided else "") + "."
        f" Epoch={model_state['epoch']}."
        f" Global_step={model_state['global_step']}."
        f" File size [in MB]: {file_size / 1e6:.2f}"
    )
    if model_state.get("wandb") is not None:
        str_to_print += f"\nRun ID: {model_state['wandb']['id']}\t Name: {model_state['wandb']['name']}"
    log.info(str_to_print)
    if also_ckpt:
        to_return["ckpt"] = model_state
    return to_return


def get_checkpoint_from_path_or_wandb(
    model_checkpoint: Optional[torch.nn.Module] = None,
    model_checkpoint_path: Optional[str] = None,
    wandb_run_id: Optional[str] = None,
    model_name: Optional[str] = "model",
    reload_kwargs: Optional[Dict[str, Any]] = None,
    model_overrides: Optional[List[str]] = None,
) -> torch.nn.Module:
    if model_checkpoint is not None:
        assert model_checkpoint_path is None, "must provide either model_checkpoint or model_checkpoint_path"
        assert wandb_run_id is None, "must provide either model_checkpoint or wandb_run_id"
        model = model_checkpoint
    # elif model_checkpoint_path is not None:
    #     raise NotImplementedError('Todo: implement loading from checkpoint path')
    #     assert wandb_run_path is None, 'must provide either model_checkpoint or wandb_run_path'
    #
    elif wandb_run_id is not None:
        # assert model_checkpoint_path is None, 'must provide either wandb_run_path or model_checkpoint_path'
        override_key_value = model_overrides or []
        override_key_value += ["module.verbose=False"]
        reload_kwargs = reload_kwargs or {}
        model = reload_checkpoint_from_wandb(
            run_id=wandb_run_id,
            also_datamodule=False,
            override_key_value=override_key_value,
            local_checkpoint_path=model_checkpoint_path,
            **reload_kwargs,
        )["model"]
    else:
        raise ValueError("Provide either model_checkpoint, model_checkpoint_path or wandb_run_id")
    return model


def reload_checkpoint_from_wandb(
    run_id: str,
    entity: str = None,
    project: str = None,
    ckpt_filename: Optional[str] = None,
    epoch: Union[str, int] = "best",
    override_key_value: List[str] = None,
    local_checkpoint_path: str = None,
    **reload_kwargs,
) -> dict:
    """
    Reload model checkpoint based on only the Wandb run ID

    Args:
        run_id (str): the wandb run ID (e.g. 2r0l33yc) corresponding to the model to-be-reloaded
        entity (str): the wandb entity corresponding to the model to-be-reloaded
        project (str): the project entity corresponding to the model to-be-reloaded
        ckpt_filename (str): the filename of the checkpoint to be reloaded (e.g. 'last.ckpt')
        epoch (str or int): If 'best', the reloaded model will be the best one stored, if 'last' the latest one stored),
                             if an int, the reloaded model will be the one save at that epoch (if it was saved, otherwise an error is thrown)
        override_key_value: each element is expected to have a "=" in it, like datamodule.num_workers=8
        local_checkpoint_path (str): If not None, the path to the local checkpoint to be reloaded.
    """
    entity, project = wandb_api.get_entity(entity), project or wandb_api.get_project_train()
    run_id = str(run_id).strip()
    run_path = f"{entity}/{project}/{run_id}"

    config = wandb_api.load_hydra_config_from_wandb(
        run_path, local_dir=local_checkpoint_path, override_key_value=override_key_value
    )
    if local_checkpoint_path is True:
        # Find local checkpoint
        assert ckpt_filename is not None, "If local_checkpoint_path is True, please specify ckpt_filename"
        local_checkpoint_path = get_local_ckpt_path(
            config,
            wandb_run=wandb_api.get_run_api(run_path=run_path),
            ckpt_filename=ckpt_filename,
            throw_error_if_local_not_found=False,
        )
    if isinstance(local_checkpoint_path, (str,)):
        ckpt_path = local_checkpoint_path  # os.path.abspath(local_checkpoint_path)
        if os.path.isdir(ckpt_path):
            if os.path.exists(os.path.join(ckpt_path, run_id, ckpt_filename)):
                ckpt_path = os.path.join(ckpt_path, run_id, ckpt_filename)
            else:
                ckpt_path = os.path.join(ckpt_path, ckpt_filename)
        assert os.path.isfile(ckpt_path), f"Could not find local ckpt {ckpt_path=} in {os.getcwd()}"
        log.info(f"Restoring model from local absolute path: {ckpt_path}")
    else:
        # Restore model from wandb cloud or S3 storage
        ckpt_path = wandb_api.restore_model_from_wandb_cloud(run_path, epoch=epoch, ckpt_filename=ckpt_filename)

    assert os.path.isfile(ckpt_path), f"Could not find {ckpt_path=} in {os.getcwd()}"
    assert str(config.logger.wandb.id) == str(run_id), f"{config.logger.wandb.id=} != {run_id=}."
    try:
        reloaded_model_data = reload_model_from_config_and_ckpt(config, ckpt_path, **reload_kwargs)
    except RuntimeError as e:
        rank = os.environ.get("RANK", None) or os.environ.get("LOCAL_RANK", 0)
        raise RuntimeError(
            f"[rank: {rank}] You may have changed the model code, making it incompatible with older model "
            f"versions. Tried to reload the model ckpt for run.id={run_id} from {ckpt_path}.\n"
            f"config.model={config.model}"
        ) from e
    if reloaded_model_data.get("wandb") is not None:
        if reloaded_model_data["wandb"].get("id") != run_id:
            raise ValueError(f"run_id={run_id} != state_dict['wandb']['id']={reloaded_model_data['wandb']['id']}")
    # config.trainer.resume_from_checkpoint = ckpt_path
    # os.remove(ckpt_path) if os.path.exists(ckpt_path) else None  # delete the downloaded ckpt
    return {**reloaded_model_data, "config": config, "ckpt_path": ckpt_path}


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
    if not isinstance(exclude_keys, list):
        exclude_keys = list(exclude_keys)
    # add _orig_mod. prefix for when using torch compile
    exclude_keys += [ek.replace("model.", "model._orig_mod.") for ek in exclude_keys]
    exclude_keys = set(exclude_keys)
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

    output_str = [
        f"Total reloaded parameters: {changed_params:,}/{total_params:,} ({changed_params / total_params * 100:.1f}%)"
    ]

    if changed:
        output_str.append(f"Changed (i.e. reloaded) layers ({len(changed)} total; showing params changed/total):")
        output_str.extend(changed[:num_examples])
        remaining = len(changed) - 2 * num_examples
        if remaining > 0:
            output_str.append(f"... and {remaining} layers in between ...")
        if len(changed) > num_examples:
            output_str.extend(changed[-num_examples:])
    else:
        output_str.append("No layers were changed")

    if unchanged:
        output_str.append(f"\nUnchanged layers (layers without any change; {len(unchanged)} total):")
        output_str.extend(unchanged[:num_examples])
        remaining = len(unchanged) - 2 * num_examples
        if remaining > 0:
            output_str.append(f"... and {remaining} layers in between ...")
        if len(unchanged) > num_examples:
            output_str.extend(unchanged[-num_examples:])
    else:
        output_str.append("No layers were unchanged")

    if unloaded_keys:
        output_str.append(
            f"\nUnloaded keys (layers in the state dict but not in the model; {len(unloaded_keys)} total):"
        )
        output_str.extend(unloaded_keys[:num_examples])
        remaining = len(unloaded_keys) - 2 * num_examples
        if remaining > 0:
            output_str.append(f"... and {remaining} keys in between ...")
        if len(unloaded_keys) > num_examples:
            output_str.extend(unloaded_keys[-num_examples:])
    else:
        output_str.append("No unloaded keys")

    log.info("\n".join(output_str))
    return state_dict


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
