import os

import hydra
import wandb
from omegaconf import DictConfig

from src.train import run_model

if "CONFIG_PATH" in os.environ:
    # Split config path and config name from config path (split by last '/')
    config_path, config_name = os.environ["CONFIG_PATH"].rsplit("/", 1)
else:
    config_path = "src/configs/"
    config_name = "main_config.yaml"


@hydra.main(config_path=config_path, config_name=config_name, version_base=None)
def main(config: DictConfig) -> float:
    """Run/train model based on the config file configs/main_config.yaml (and any command-line overrides)."""
    return run_model(config)


if __name__ == "__main__":
    if "WANDB_API_KEY" in os.environ:
        wandb.login(key=os.environ["WANDB_API_KEY"])

    os.environ["HYDRA_FULL_ERROR"] = "1"
    main()
