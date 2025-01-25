from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Union

import hydra
import pytorch_lightning
import torch
from omegaconf import DictConfig

from src.datamodules.abstract_datamodule import BaseDataModule
from src.experiment_types._base_experiment import BaseExperiment
from src.utilities.checkpointing import load_state_dict_and_analyze_weight_changes
from src.utilities.utils import (
    get_logger,
    rename_state_dict_keys_and_save,
)


"""
In this file you can find helper functions to avoid model/data loading and reloading boilerplate code
"""

log = get_logger(__name__)


def get_lightning_module(config: DictConfig, **kwargs) -> BaseExperiment:
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


def get_datamodule(config: DictConfig) -> BaseDataModule:
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


def get_model_and_data(config: DictConfig) -> (BaseExperiment, BaseDataModule):
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
    # remove all keys with model.interpolator prefix
    # state_dict = {k: v for k, v in state_dict.items() if not k.startswith("model.interpolator")}
    load_state_dict_and_analyze_weight_changes(
        model, state_dict, strict=reload_strict, exclude_keys=exclude_state_dict_keys
    )

    to_return = {
        "model": model,
        "datamodule": data_module,
        "state_dict": state_dict,
        "epoch": model_state["epoch"],
        "global_step": model_state["global_step"],
        "wandb": model_state.get("wandb", None),
    }
    file_size = os.path.getsize(model_path)
    str_to_print = (
        f"Reloaded {model_path}" + (" into provided model" if model_provided else "") + "."
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
    import src.utilities.wandb_api as wandb_api

    entity, project = wandb_api.get_entity(entity), project or wandb_api.get_project_train()
    run_id = str(run_id).strip()
    run_path = f"{entity}/{project}/{run_id}"

    config = wandb_api.load_hydra_config_from_wandb(run_path, override_key_value=override_key_value)
    ckpt_path = wandb_api.restore_model_from_wandb_cloud(
        run_path,
        local_checkpoint_path=local_checkpoint_path,
        epoch=epoch,
        ckpt_filename=ckpt_filename,
        throw_error_if_local_not_found=False,
        config=config,
    )

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


def get_simple_trainer(**kwargs) -> pytorch_lightning.Trainer:
    devices = kwargs.get("devices", 1 if torch.cuda.is_available() else None)
    accelerator = kwargs.get("accelerator", "gpu" if torch.cuda.is_available() else None)
    return pytorch_lightning.Trainer(
        devices=devices,
        accelerator=accelerator,
        **kwargs,
    )


def run_inference(
    module: pytorch_lightning.LightningModule,
    datamodule: pytorch_lightning.LightningDataModule,
    trainer: pytorch_lightning.Trainer = None,
    trainer_kwargs: Dict[str, Any] = None,
):
    trainer = trainer or get_simple_trainer(**(trainer_kwargs or {}))
    results = trainer.predict(module, datamodule=datamodule)
    results = module._evaluation_get_preds(results, split="predict")
    if hasattr(datamodule, "numpy_results_to_xr_dataset"):
        results = datamodule.numpy_results_to_xr_dataset(results, split="predict")
    return results
