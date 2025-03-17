import argparse
import dataclasses
import os
import time
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union

import dacite
import torch
import tqdm.auto as tqdm
import yaml

import wandb
from src.ace_inference.ace.aggregator.inference.main import InferenceAggregator
from src.ace_inference.ace.aggregator.null import NullAggregator
from src.ace_inference.ace.data_loading.data_typing import GriddedData
from src.ace_inference.ace.data_loading.getters import get_inference_data
from src.ace_inference.ace.data_loading.inference import InferenceDataLoaderConfig
from src.ace_inference.ace.inference import gcs_utils, logging_utils
from src.ace_inference.ace.inference.data_writer.main import DataWriter, DataWriterConfig
from src.ace_inference.ace.inference.loop import run_dataset_inference, run_inference
from src.ace_inference.core.device import get_device
from src.ace_inference.core.dicts import to_flat_dict
from src.ace_inference.core.stepper import SingleModuleStepper
from src.ace_inference.core.stepper_multistep import MultiStepStepper
from src.ace_inference.core.wandb import WandB
from src.utilities.utils import get_logger
from src.utilities.wandb_api import restore_model_from_wandb_cloud


logging = get_logger(__name__)
device = get_device()


def load_stepper(
    checkpoint_file: str, overrides: Dict[str, Any] = None, area=None
) -> Union[SingleModuleStepper, MultiStepStepper]:
    checkpoint = torch.load(checkpoint_file, map_location=device, weights_only=False)
    # checkpoint['hyper_parameters'].pop('prediction_mode', None)
    # checkpoint["hyper_parameters"]["diffusion_config"]["_target_"] = checkpoint["hyper_parameters"]["diffusion_config"]["_target_"].replace("DYffusionMultiHorizonWithPretrainedInterpolator", "DYffusion")
    # checkpoint["hyper_parameters"]["diffusion_config"].pop("is_parametric", None)
    # checkpoint["hyper_parameters"]["diffusion_config"].pop("prediction_mode", None)
    # checkpoint["hyper_parameters"]["diffusion_config"].pop("use_mean_of_parametric_predictions", None)
    # checkpoint["hyper_parameters"]["model_config"]["dropout_filter"] = 0
    # print(f"{checkpoint['hyper_parameters']}")
    # torch.save(checkpoint, checkpoint_file)
    # torch.save(checkpoint, checkpoint_file.replace('.ckpt', '_cleaned.ckpt'))
    epoch = checkpoint["epoch"]
    if wandb.run is None:
        pass  # wandb is not being used
    elif "wandb" in checkpoint.keys():
        # wandb.run.group = checkpoint['wandb']['group']: NOT POSSIBLE
        try:
            wandb.run._run_obj.run_group = checkpoint["wandb"]["group"]  # older version of wandb
        except AttributeError:
            pass
        wandb.run.name = checkpoint["wandb"]["name"] + f"-{epoch}epoch"
        run_current = wandb.Api().run(wandb.run.path)
        run_current.group = checkpoint["wandb"]["group"]
        run_current.update()
    else:
        wandb.run.name = f"{wandb.run.name}-{epoch}epoch"

    ckpt_time = {k: checkpoint[k] for k in ["epoch", "step", "global_step"] if k in checkpoint.keys()}
    print(f"Checkpoint state from: {ckpt_time}")
    if wandb.run is not None:
        wandb.log(ckpt_time, step=0)
        wandb.run.summary.update(ckpt_time)

    # Check if it is Spherical DYffusion model
    if ("FV3GFS" in checkpoint_file and "seed" in checkpoint_file) or ".ckpt" in checkpoint_file:
        stepper = MultiStepStepper.from_state(checkpoint, load_optimizer=False, overrides=overrides)
    else:
        assert overrides is None, "Overrides not supported for non-DYffusion models. Please set it to None."
        stepper = SingleModuleStepper.from_state(checkpoint["stepper"], area)
    return stepper


@dataclasses.dataclass
class InferenceConfig:
    """
    Configuration for running inference.

    Attributes:
        experiment_dir: Directory to save results to.
        n_forward_steps: Number of steps to run the model forward for. Must be divisble
            by forward_steps_in_memory.
        checkpoint_path: Path to stepper checkpoint to load.
        logging: configuration for logging.
        validation_loader: Configuration for validation data.
        prediction_loader: Configuration for prediction data to evaluate. If given,
            model evaluation will not run, and instead predictions will be evaluated.
            Model checkpoint will still be used to determine inputs and outputs.
        log_video: Whether to log videos of the state evolution.
        log_extended_video: Whether to log wandb videos of the predictions with
            statistical metrics, only done if log_video is True.
        log_extended_video_netcdfs: Whether to log videos of the predictions with
            statistical metrics as netcdf files.
        log_zonal_mean_images: Whether to log zonal-mean images (hovmollers) with a
            time dimension.
        save_prediction_files: Whether to save the predictions as a netcdf file.
        save_raw_prediction_names: Names of variables to save in the predictions
             netcdf file. Ignored if save_prediction_files is False.
        forward_steps_in_memory: Number of forward steps to complete in memory
            at a time, will load one more step for initial condition.
        data_writer: Configuration for data writers.
        overrides: Overrides for the re-loaded module. E.g. change the sampling behavior. Should be a dict or dict of dicts
    """

    experiment_dir: str
    n_forward_steps: int
    checkpoint_path: str
    logging: logging_utils.LoggingConfig
    validation_loader: InferenceDataLoaderConfig
    prediction_loader: Optional[InferenceDataLoaderConfig] = None
    n_ensemble_members: int = 1
    wandb_run_path: Optional[str] = None
    log_video: bool = True
    log_extended_video: bool = False
    log_extended_video_netcdfs: Optional[bool] = None
    log_zonal_mean_images: bool = True
    save_prediction_files: Optional[bool] = None
    save_raw_prediction_names: Optional[Sequence[str]] = None
    forward_steps_in_memory: int = 1
    overrides: Optional[Dict[str, Any]] = None
    data_writer: DataWriterConfig = dataclasses.field(default_factory=lambda: DataWriterConfig())
    compute_metrics: bool = True

    def __post_init__(self):
        # if self.n_forward_steps % self.forward_steps_in_memory != 0:
        #     raise ValueError(
        #         "n_forward_steps must be divisible by steps_in_memory, "
        #         f"got {self.n_forward_steps} and {self.forward_steps_in_memory}"
        #     )
        deprecated_writer_attrs = {
            k: getattr(self, k)
            for k in [
                "log_extended_video_netcdfs",
                "save_prediction_files",
                "save_raw_prediction_names",
            ]
            if getattr(self, k) is not None
        }
        for k, v in deprecated_writer_attrs.items():
            warnings.warn(
                f"Inference configuration attribute `{k}` is deprecated. "
                f"Using its value `{v}`, but please use attribute `data_writer` "
                "instead."
            )
            setattr(self.data_writer, k, v)
        if (self.data_writer.time_coarsen is not None) and (
            self.forward_steps_in_memory % self.data_writer.time_coarsen.coarsen_factor != 0
        ):
            raise ValueError(
                "forward_steps_in_memory must be divisible by "
                f"time_coarsen.coarsen_factor. Got {self.forward_steps_in_memory} "
                f"and {self.data_writer.time_coarsen.coarsen_factor}."
            )

    def configure_logging(self, log_filename: str):
        self.logging.configure_logging(self.experiment_dir, log_filename)

    def configure_wandb(self, env_vars: Optional[dict] = None, **kwargs):
        config = to_flat_dict(dataclasses.asdict(self))
        if "environment" in config:
            logging.warning("Not recording env vars since 'environment' is in config.")
        elif env_vars is not None:
            config["environment"] = env_vars
        self.logging.configure_wandb(config=config, resume=False, **kwargs)

    def clean_wandb(self):
        self.logging.clean_wandb(self.experiment_dir)

    def load_stepper(self, **kwargs) -> Union[SingleModuleStepper, MultiStepStepper]:
        logging.info(f"Loading trained model checkpoint from {self.checkpoint_path}")
        checkpoint_file = self.checkpoint_path
        if os.path.exists(checkpoint_file) and self.wandb_run_path is None:
            print(f"Loading checkpoint from local path {checkpoint_file}")
            pass
        elif self.wandb_run_path is not None:
            checkpoint_file = restore_model_from_wandb_cloud(self.wandb_run_path, ckpt_filename=checkpoint_file)
            logging.info(f"Restored model ckpt ``{checkpoint_file}`` from wandb run path {self.wandb_run_path}.")
        else:
            from pathlib import Path

            # List directory contents of every directory in the path, starting from the end
            # until we find a directory that exists
            path = Path(checkpoint_file)
            while not path.exists():
                path = path.parent
            print(f"Found {path} to exist. Ls: {os.listdir(path)}")
        return load_stepper(checkpoint_file, overrides=self.overrides, **kwargs)

    def get_data_writer(self, data: GriddedData) -> DataWriter:
        return self.data_writer.build(
            experiment_dir=self.experiment_dir,
            n_samples=self.validation_loader.n_samples,
            n_timesteps=self.n_forward_steps + 1,
            metadata=data.metadata,
            coords=data.coords,
            n_ensemble_members=self.n_ensemble_members,
        )


def main(
    yaml_config: str,
):
    with open(yaml_config, "r") as f:
        data = yaml.safe_load(f)
    config = dacite.from_dict(
        data_class=InferenceConfig,
        data=data,
        config=dacite.Config(strict=True),
    )

    if os.path.exists(config.experiment_dir):
        # Append a timestamp to the experiment directory to avoid overwriting
        config.experiment_dir += f"-{time.strftime('%Y-%m-%d-%H-%M-%S')}"
    assert not os.path.exists(config.experiment_dir), f"Experiment directory {config.experiment_dir} already exists."
    if not os.path.isdir(config.experiment_dir):
        os.makedirs(config.experiment_dir)
    with open(os.path.join(config.experiment_dir, "config.yaml"), "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    config.configure_logging(log_filename="inference_out.log")
    env_vars = logging_utils.retrieve_env_vars()
    config.configure_wandb(env_vars=env_vars)
    gcs_utils.authenticate()

    torch.backends.cudnn.benchmark = True

    logging_utils.log_versions()
    _ = WandB.get_instance()  # wandb = WandB.get_instance()

    start_time = time.time()
    stepper = config.load_stepper()
    logging.info("Loading inference data")
    data_requirements = stepper.get_data_requirements(n_forward_steps=config.n_forward_steps)

    data = get_inference_data(
        config.validation_loader,
        config.forward_steps_in_memory,
        data_requirements,
    )

    eval_device = device  # 'cuda'  #'cpu' if config.n_ensemble_members > 1 else 'cuda'
    aggregator = (
        InferenceAggregator(
            data.area_weights.to(device),
            sigma_coordinates=data.sigma_coordinates,
            record_step_20=config.n_forward_steps >= 20,
            log_video=config.log_video,
            enable_extended_videos=config.log_extended_video,
            log_zonal_mean_images=config.log_zonal_mean_images,
            n_timesteps=config.n_forward_steps + 1,
            metadata=data.metadata,
            n_ensemble_members=config.n_ensemble_members,
            device=eval_device,
        )
        if config.compute_metrics
        else NullAggregator()
    )
    writer = config.get_data_writer(data) if config.compute_metrics else None

    logging.info("Starting inference")
    if config.prediction_loader is not None:
        prediction_data = get_inference_data(
            config.prediction_loader,
            config.forward_steps_in_memory,
            data_requirements,
        )

        timers = run_dataset_inference(
            aggregator=aggregator,
            normalizer=stepper.normalizer,
            prediction_data=prediction_data,
            target_data=data,
            n_forward_steps=config.n_forward_steps,
            forward_steps_in_memory=config.forward_steps_in_memory,
            writer=writer,
        )
    else:
        timers = run_inference(
            aggregator=aggregator,
            writer=writer,
            stepper=stepper,
            data=data,
            n_forward_steps=config.n_forward_steps,
            forward_steps_in_memory=config.forward_steps_in_memory,
            n_ensemble_members=config.n_ensemble_members,
            eval_device=eval_device,
        )

    duration = time.time() - start_time
    total_steps = config.n_forward_steps * config.validation_loader.n_samples
    total_steps_per_second = total_steps / duration
    logging.info(f"Inference duration: {duration:.2f} seconds")
    logging.info(f"Total steps per second: {total_steps_per_second:.2f} steps/second")

    step_logs = aggregator.get_inference_logs(label="inference")
    tqdm_bar = tqdm.tqdm(step_logs, desc="Logging inference results to wandb")
    wandb = WandB.get_instance()
    duration_logs = {
        "duration_seconds": duration,
        "time/inference": duration,
        "total_steps_per_second": total_steps_per_second,
    }
    wandb.log({**timers, **duration_logs}, step=0)
    for i, log in enumerate(tqdm_bar):
        log["timestep"] = i
        wandb.log(log, step=i)
        # wandb.log cannot be called more than "a few times per second"
        time.sleep(0.005)
    writer.flush()

    logging.info("Writing reduced metrics to disk in netcdf format.")
    aggregators_to_save = ["time_mean"]  # , "zonal_mean"]
    for name, ds in aggregator.get_datasets(aggregators_to_save).items():
        coords = {k: v for k, v in data.coords.items() if k in ds.dims}
        ds = ds.assign_coords(coords)
        ds.to_netcdf(Path(config.experiment_dir) / f"{name}_diagnostics.nc")

    # config.clean_wandb()
    return step_logs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("yaml_config", type=str)

    args = parser.parse_args()

    main(yaml_config=args.yaml_config)
