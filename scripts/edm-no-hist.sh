#!/bin/bash

cd "$(dirname "$0")"/..
# export name of file as environment variable (to know which script created the run)
export SCRIPT_NAME=$(basename "$0")

# "$@" makes sure to use any extra command line arguments supplied here with bash <script>.sh <args>
python run.py "$@" \
    experiment=climatebench_daily_edm \
    datamodule=climatebench_daily \
    trainer=default \
    trainer.devices=2 \
    trainer.max_epochs=100 \
    datamodule.data_dir=/data/normalized \
    datamodule.mean_over_ensemble='first' \
    datamodule.simulations=["ssp126","ssp370","ssp585"] \
    module.num_predictions=10 module.prediction_inputs_noise=0.0 \
    module.enable_inference_dropout=True \
    module.optimizer.lr=1e-4 module.optimizer.weight_decay=1e-4 \
    datamodule.window=1 \
    datamodule.batch_size=128 \
    datamodule.eval_batch_size=8 \
    datamodule.batch_size_per_gpu=16 \
    model.dim=128 \
    callbacks.early_stopping.patience=10 test_after_training=True \
    datamodule.DEBUG_dataset_size=null \
    suffix=edm_batch_128 \
    "$@"


# File to run a base EDM model
# Already has Historical data used during training (in config file)