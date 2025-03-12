import dataclasses
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import dacite
import torch
from torch import nn
from tqdm.auto import tqdm

from src.ace_inference.core.aggregator.null import NullAggregator
from src.ace_inference.core.device import get_device
from src.ace_inference.core.distributed import Distributed
from src.ace_inference.core.normalizer import (
    NormalizationConfig,
    StandardNormalizer,
    # FromStateNormalizer,
)
from src.ace_inference.core.prescriber import NullPrescriber, Prescriber, PrescriberConfig
from src.ace_inference.core.stepper import SingleModuleStepper
from src.evaluation.aggregators.main import OneStepAggregator
from src.experiment_types.forecasting_multi_horizon import (
    AbstractMultiHorizonForecastingExperiment,
    infer_class_from_ckpt,
)
from src.utilities.packer import Packer
from src.utilities.utils import to_tensordict, update_dict_with_other

from .optimization import NullOptimization, Optimization
from .stepper import SteppedData, get_name_and_time_query_fn


@dataclasses.dataclass
class DataRequirements:
    names: List[str]
    # TODO: delete these when validation no longer needs them
    in_names: List[str]
    out_names: List[str]
    n_timesteps: int


@dataclasses.dataclass
class MultiStepStepperConfig:
    in_names: List[str]
    out_names: List[str]
    prescriber: Optional[PrescriberConfig] = None
    data_dir: Optional[str] = None
    data_dir_stats: Optional[str] = None

    def get_data_requirements(self, n_forward_steps: int) -> DataRequirements:
        return DataRequirements(
            names=self.all_names,
            in_names=self.in_names,
            out_names=self.out_names,
            n_timesteps=n_forward_steps + 1,
        )

    def get_stepper(
        self,
        shapes: Dict[str, Tuple[int, ...]],
        max_epochs: int,
    ):
        return MultiStepStepper(
            config=self,
        )

    def get_state(self):
        return dataclasses.asdict(self)

    @classmethod
    def from_state(cls, state) -> "MultiStepStepperConfig":
        return dacite.from_dict(data_class=cls, data=state, config=dacite.Config(strict=True))

    @property
    def all_names(self):
        if self.prescriber is not None:
            mask_name = [self.prescriber.mask_name]
        else:
            mask_name = []
        all_names = list(set(self.in_names).union(self.out_names).union(mask_name))
        return all_names

    @property
    def normalize_names(self):
        return list(set(self.in_names).union(self.out_names))


class MultiStepStepper(SingleModuleStepper):
    """
    Stepper class for a single pytorch module.
    """

    channel_axis = -3

    def __init__(
        self,
        config: MultiStepStepperConfig,
        module: AbstractMultiHorizonForecastingExperiment,
        data_shapes: Dict[str, Tuple[int, ...]],
        max_epochs: int,
    ):
        """
        Args:
            config: The configuration.
            data_shapes: The shapes of the data.
            max_epochs: The maximum number of epochs. Used when constructing
                certain learning rate schedulers, if applicable.
        """
        dist = Distributed.get_instance()
        # n_in_channels = len(config.in_names)
        # n_out_channels = len(config.out_names)
        if "forcing_names" not in config.__dict__:
            config.forcing_names = list(set(config.in_names).difference(config.out_names))
        self.init_packers(config.in_names, config.out_names, config.forcing_names)
        # self.in_packer = Packer(config.in_names, axis=self.channel_axis)
        # self.out_packer = Packer(config.out_names, axis=self.channel_axis)
        # in_packer.names = [x for x in in_names if x not in forcing_names]
        # forcings_packer = Packer(forcing_names, axis_pack=in_packer.axis_pack, axis_unpack=in_packer.axis_unpack)
        # self.forcings_packer = Packer(config.forcing_names, axis=self.channel_axis)
        data_dir_stats = config.data_dir_stats or config.data_dir
        path_mean = Path(data_dir_stats) / "centering.nc"
        path_std = Path(data_dir_stats) / "scaling.nc"
        alternative_data_dirs = [
            "/net/nfs/climate/salvar/data/fv3gfs-ensemble-ic0001-stats-residual-scaling-all-years-v2",
            "/full-model/data",
            "/pscratch/sd/s/salvarc/full-model/data",
            "/data/climate-model/fv3gfs",
        ]
        if not path_mean.exists():
            for alt_dir in alternative_data_dirs:
                path_mean = Path(alt_dir) / "centering.nc"
                path_std = Path(alt_dir) / "scaling.nc"
                if path_mean.exists():
                    break
        if not path_mean.exists():
            raise FileNotFoundError(
                f"Could not find centering and scaling files in {data_dir_stats} or alternative dirs {alternative_data_dirs}"
            )

        normalization_config = NormalizationConfig(global_means_path=path_mean, global_stds_path=path_std)
        self.normalizer = normalization_config.build(config.normalize_names)

        # self.normalizer = get_normalizer(path_mean, path_std, names=config.normalize_names)
        if config.prescriber is not None:
            self.prescriber = config.prescriber.build(config.in_names, config.out_names)
        else:
            self.prescriber = NullPrescriber()
        self.module = module.to(get_device())
        self.data_shapes = data_shapes
        self._config = config
        self._max_epochs = max_epochs
        self.optimization = NullOptimization()

        self._no_optimization = NullOptimization()
        self._is_distributed = dist.is_distributed()

    def run_on_batch(
        self,
        data: Dict[str, torch.Tensor],
        optimization: Union[Optimization, NullOptimization],
        n_forward_steps: int = 1,
        aggregator: Optional[OneStepAggregator] = None,
    ) -> Tuple[
        float,
        Dict[str, torch.Tensor],
        Dict[str, torch.Tensor],
        Dict[str, torch.Tensor],
    ]:
        """
        Step the model forward on a batch of data.

        Args:
            data: The batch data of shape [n_sample, n_timesteps, n_channels, n_x, n_y].
            optimization: The optimization class to use for updating the module.
                Use `NullOptimization` to disable training.
            n_forward_steps: The number of timesteps to run the model for.

        Returns:
            The loss, the generated data, the normalized generated data,
                and the normalized batch data.
        """
        if aggregator is None:
            non_none_aggregator: Union[OneStepAggregator, NullAggregator] = NullAggregator()
        else:
            non_none_aggregator = aggregator

        device = get_device()
        device_data = {name: value.to(device, dtype=torch.float) for name, value in data.items()}
        return run_on_batch_multistep(
            data=device_data,
            module=self.module,
            normalizer=self.normalizer,
            in_packer=self.in_packer,
            out_packer=self.out_packer,
            forcings_packer=self.forcings_packer,
            optimization=optimization,
            loss_obj=self.loss_obj,
            n_forward_steps=n_forward_steps,
            prescriber=self.prescriber,
            aggregator=non_none_aggregator,
        )

    def load_state(self, state, load_optimizer: bool = True):
        """
        Load the state of the stepper.

        Args:
            state: The state to load.
            load_optimizer: Whether to load the optimizer state.
        """
        hparams = state["hyper_parameters"]
        hparams_data = hparams["datamodule_config"]
        state_dict = state["state_dict"]
        # state_dict = {f"module.{k}": v for k, v in state_dict.items()}  # add module. to keys if using DummyWrapper
        # Reload weights
        try:
            self.module.load_state_dict(state_dict)
        except RuntimeError as e:
            raise RuntimeError(
                f"Error loading state_dict."
                f"\nHyperparameters: {hparams}\nData: {hparams_data}\nself.module={self.module}"
            ) from e

        if load_optimizer and "optimization" in state:
            self.optimization.load_state(state["optimization"])
        # in_names = hparams_data['in_names'] + hparams_data['forcing_names']
        self.init_packers(hparams_data["in_names"], hparams_data["out_names"], hparams_data["forcing_names"])
        # self.prescriber.load_state(None)

    def init_packers(self, in_names, out_names, forcing_names):
        in_names = [x for x in in_names if x not in forcing_names]
        self.in_packer = Packer(in_names, axis=self.channel_axis)
        self.out_packer = Packer(out_names, axis=self.channel_axis)
        self.forcings_packer = Packer(forcing_names, axis=self.channel_axis)

    @classmethod
    def from_state(cls, state, load_optimizer: bool = True, overrides: Dict[str, Any] = None) -> "MultiStepStepper":
        """
        Load the state of the stepper.

        Args:
            state: The state to load.
            load_optimizer: Whether to load the optimizer state.
            overrides: Key -> value pairs to override the module's hyperparameters (value can be a dict).

        Returns:
            The stepper.
        """
        overrides = overrides or {}
        module_class = infer_class_from_ckpt(ckpt_path=None, state=state)
        # print(state['hyper_parameters'].keys())
        actual_hparams, diff_to_default = update_dict_with_other(state["hyper_parameters"], overrides)
        module = module_class(**actual_hparams)
        # Print the differences between the default and actual hyperparameters
        if len(diff_to_default) > 0:
            print("---------------- Overriding the following hyperparameters:")
            print("|\t" + "\n\t".join(diff_to_default))
            print("----------------------------------------------------------")
            # Update the wandb config and save diff_to_default as notes
            import wandb

            try:
                from omegaconf import OmegaConf

                # Make config a omegaconf.DictConfig
                actual_hparams = OmegaConf.create(actual_hparams)
            except ImportError:
                pass
            if wandb.run is not None:
                # try:
                #     wandb.config.update(actual_hparams)
                # except TypeError as e:
                #     print(f"Error updating wandb config: {e}")
                #     print(f"actual_hparams: {actual_hparams}")
                #     print(f"diff_to_default: {diff_to_default}\nSkipping... updating hparams to wandb")
                wandb.run.notes = " ".join(diff_to_default)
                wandb.log({"wandb.notes": wandb.run.notes}, step=0)
        elif len(overrides) > 0:
            print(f"No differences were found between the default and actual hyperparameters. Overrides: {overrides}")

        data_config = state["hyper_parameters"]["datamodule_config"]
        # Salva's runs use separate in_names and forcing_names, for ACE data-loading we just combine them
        state["hyper_parameters"]["datamodule_config"]["in_names"] = (
            data_config["in_names"] + data_config["forcing_names"]
        )
        config = {}  # 'builder': None, 'optimization': None}
        for x in ["in_names", "out_names"]:
            config[x] = list(data_config[x])
        for y in ["data_dir", "data_dir_stats"]:
            config[y] = data_config[y]

        # Build prescriber back from saved config file
        prescriber_config = data_config["prescriber"]
        prescriber_config.pop("_target_")
        config["prescriber"] = PrescriberConfig(**prescriber_config)
        stepper = cls(
            config=MultiStepStepperConfig.from_state(config),
            module=module,
            data_shapes=None,  # state["data_shapes"],
            max_epochs=1000,  # training not supported yet
        )
        stepper.load_state(state, load_optimizer=load_optimizer)
        return stepper


def run_on_batch_multistep(
    data: Dict[str, torch.Tensor],
    module: AbstractMultiHorizonForecastingExperiment,
    normalizer: StandardNormalizer,
    in_packer: Packer,
    out_packer: Packer,
    forcings_packer: Packer,
    optimization: Union[Optimization, NullOptimization],
    loss_obj: nn.Module,
    prescriber: Union[Prescriber, NullPrescriber],
    aggregator: Union[OneStepAggregator, NullAggregator],
    n_forward_steps: int = 1,
) -> SteppedData:
    """
    Run the model on a batch of data.

    The module is assumed to require un-packed and normalized data (packing must be handled by the module),
    except the forcing data, which is assumed to be packed and normalized.

    Args:
        data: The denormalized batch data. The second dimension of each tensor
            should be the time dimension.
        module: The module to run.
        normalizer: The normalizer.
        in_packer: The packer for the input data.
        out_packer: The packer for the output data.
        optimization: The optimization object. If it is NullOptimization,
            then the model is not trained.
        loss_obj: The loss object.
        prescriber: Overwrite an output with target value in specified region.
        n_forward_steps: The number of timesteps to run the model for.

    Returns:
        The loss, the generated data, the normalized generated data,
            and the normalized batch data. The generated data contains
            the initial input data as its first timestep.
    """
    assert isinstance(prescriber, Prescriber), f"prescriber is not a Prescriber, but {type(prescriber)}"
    module_actual = module.module if hasattr(module, "module") else module
    horizon_training = module_actual.true_horizon
    in_names = in_packer.names.copy()
    # forcing_names = list(set(in_packer.names).difference(out_packer.names))
    # in_packer.names = [x for x in in_names if x not in forcing_names]
    # forcings_packer = Packer(forcing_names, axis_pack=in_packer.axis_pack, axis_unpack=in_packer.axis_unpack)
    # must be negative-indexed, so it works with or without a time dim
    channel_dim = -3
    time_dim = 1
    example_shape = data[list(data.keys())[0]].shape
    assert len(example_shape) == 4
    assert example_shape[1] == n_forward_steps + 1
    forcings_time_slice = slice(0, horizon_training + 1)  # e.g. 0:6 for h=6 training horizon
    full_data_norm = normalizer.normalize(data)
    get_input_data = get_name_and_time_query_fn(data, full_data_norm, time_dim)

    device = get_device()
    eval_device = "cpu"
    full_target_tensor_norm = out_packer.pack(full_data_norm, axis=channel_dim)
    loss = torch.tensor(0.0, device=device)
    metrics = {}
    input_data_norm = get_input_data(in_packer.names, time_index=0, norm_mode="norm")
    forcing_data_norm = get_input_data(forcings_packer.names, time_index=0, norm_mode="norm")
    is_imprecise = (
        hasattr(module_actual.model.hparams, "hack_for_imprecise_interpolation")
        and module_actual.model.hparams.hack_for_imprecise_interpolation
    )
    gen_data_norm = []
    optimization.set_mode(module)
    tqdm_bar = tqdm(range(1, n_forward_steps + 1), desc="Horizon")
    num_predictions = 1
    module_actual.num_predictions = num_predictions
    module_actual.num_predictions_in_mem = num_predictions
    for total_horizon in tqdm_bar:
        # We need to map from the total horizon to the horizon for training, e.g. if train horizon = 3:
        # total_horizon = 1 -> horizon_training = 1, total_horizon = 2 -> horizon_training = 2,
        # total_horizon = 3 -> horizon_training = 3, total_horizon = 4 -> horizon_training = 1
        # total_horizon = 5 -> horizon_training = 2, etc.
        horizon_rel = total_horizon % horizon_training
        if horizon_rel == 0:
            horizon_rel = horizon_training
        input_tensor_norm = in_packer.pack(input_data_norm, axis=channel_dim)  # Done inside module

        target_tensor_norm = full_target_tensor_norm.select(dim=time_dim, index=total_horizon)
        with optimization.autocast():
            batch = {
                # module_actual.main_data_key_val: to_tensordict(input_data_norm),  # uncomment
                "dynamics": input_tensor_norm.to(device),
                # "dynamical_condition": forcing_tensor_norm.to(device),
            }
            if is_imprecise:
                forcings_key = "static_condition"
            else:
                forcings_key = "dynamical_condition"
                # forcing_data_norm = get_input_data(forcings_packer.names, time_index=forcings_time_slice, norm_mode="norm")
                # print(f"{total_horizon=} {forcings_time_slice=}, {list(full_data_norm.values())[0].shape=}")
                if horizon_rel == 1:  # new prediction, need to check that all forcings are available
                    assert example_shape[1] + 1 >= forcings_time_slice.stop, f"{example_shape=} {forcings_time_slice=}"
                forcing_data_norm = {k: full_data_norm[k][:, forcings_time_slice] for k in forcings_packer.names}
            forcing_tensor_norm = forcings_packer.pack(forcing_data_norm, axis=channel_dim)
            batch[forcings_key] = forcing_tensor_norm.to(device)
            with module_actual.ema_scope():
                with module_actual.inference_dropout_scope():
                    results = module_actual.get_preds_at_t_for_batch(
                        batch,
                        horizon=horizon_rel,
                        split="predict",
                        ensemble=False,
                        is_autoregressive=total_horizon > horizon_training,
                        prepare_inputs=False,  # already done above
                        num_predictions=num_predictions,  # only one prediction at a time (one ensemble member)
                    )
            predictions_key = f"t{horizon_rel}_preds_normed"
            gen_tensor_norm = results[predictions_key]
            # gen_tensor_norm = out_packer.pack(results[predictions_key], axis=channel_dim)  #if unpacked inside module
            step_loss = loss_obj(gen_tensor_norm, target_tensor_norm.to(device))
            loss += step_loss
            metrics[f"loss_step_{total_horizon-1}"] = step_loss.detach()

        # Gen_norm will be used as input for the next AR step
        gen_norm = out_packer.unpack(gen_tensor_norm, axis=channel_dim)
        target_norm = out_packer.unpack(target_tensor_norm, axis=channel_dim)
        data_time = {k: v.select(dim=time_dim, index=total_horizon).to(device) for k, v in data.items()}
        gen_norm = prescriber(data_time, gen_norm, target_norm.to(device))
        gen_data_norm.append(gen_norm.to(eval_device))

        if "preds_autoregressive_init_normed" not in results:
            autoregressive_init_normed = gen_norm
        else:
            print("Using autoregressive_init_normed")
            autoregressive_init_normed = results["preds_autoregressive_init_normed"]
            autoregressive_init_normed = out_packer.unpack(autoregressive_init_normed, axis=channel_dim)
            autoregressive_init_normed = prescriber(data_time, autoregressive_init_normed, target_norm.to(device))
        # update input data with generated outputs, and forcings for missing outputs
        forcing_data_norm = get_input_data(forcings_packer.names, time_index=total_horizon, norm_mode="norm")
        forcings_time_slice = slice(forcings_time_slice.start + 1, forcings_time_slice.stop + 1)
        if is_imprecise:
            autoregressive_init_normed["HGTsfc"] = input_data_norm["HGTsfc"].to(device)
        # input_data_norm = {**forcing_data_norm, **gen_norm}

        # Autoregressive mode: update input data with generated outputs
        input_data_norm = autoregressive_init_normed
        del data_time

    optimization.step_weights(loss)
    # prepend the initial (pre-first-timestep) output data to the generated data
    initial = to_tensordict(get_input_data(out_packer.names, time_index=0, norm_mode="norm"), device=eval_device)
    gen_data_norm = [initial] + gen_data_norm
    # gen_data_norm_timeseries2 = torch.stack(gen_data_norm, dim=time_dim)
    gen_data_norm_timeseries = {}
    for name in out_packer.names:
        gen_data_norm_timeseries[name] = torch.stack([x[name] for x in gen_data_norm], dim=time_dim)
    gen_data = normalizer.denormalize(gen_data_norm_timeseries)

    # for name in out_packer.names:
    #     assert torch.allclose(
    #         gen_data_norm_timeseries[name], gen_data_norm_timeseries2[name]
    #     ), f'{name} not equal'
    metrics["loss"] = loss.detach()
    # shapes = set([v.shape for v in data.values()])
    # if len(shapes) > 1:
    #     d_to_s = {k: v.shape for k, v in data.items()}
    # raise ValueError(f"Shapes of data tensors are not the same: {shapes}. example_shape={example_shape}"
    #                  f"Data to shape: {d_to_s}")

    data = to_tensordict(data, device=eval_device)  # Not needed on GPU
    full_data_norm = to_tensordict(full_data_norm, device=eval_device)
    aggregator.update(
        float(loss),
        target_data=data,
        gen_data=gen_data,
        target_data_norm=full_data_norm,
        gen_data_norm=gen_data_norm_timeseries,
    )
    in_packer.names = in_names
    return SteppedData(
        metrics=metrics,
        gen_data=gen_data,
        target_data=data,
        gen_data_norm=gen_data_norm_timeseries,
        target_data_norm=full_data_norm,
    )
