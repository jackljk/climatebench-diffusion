#!/bin/bash

cd "$(dirname "$0")"/..
# export name of file as environment variable (to know which script created the run)
export SCRIPT_NAME=$(basename "$0")

# "$@" makes sure to use any extra command line arguments supplied here with bash <script>.sh <args>
python run.py "$@" \
    experiment=climatebench_daily_edm \
    datamodule=climatebench_daily \
    datamodule.data_dir=/data \
    datamodule.mean_over_ensemble='first' \
    datamodule.simulations=["ssp126","ssp370","ssp585"] \
    datamodule.window=1 \
    datamodule.batch_size=128 \
    datamodule.eval_batch_size=3 \
    datamodule.batch_size_per_gpu=4 \
    datamodule.simulations_anom_type="none" \
    datamodule.simulations_raw="all" \
    datamodule.normalization_type="standard" \
    datamodule.num_workers = 48 \
    datamodule.output_vars='pr' \
    trainer=ddp \
    trainer.devices=12 \
    trainer.max_epochs=100 \
    model.dim=128 \
    module.num_predictions=5 module.prediction_inputs_noise=0.0 \
    callbacks.early_stopping.patience=20 test_after_training=True \
    suffix=edm_no_hist_pr \
    "$@"
# File to run a base EDM model
# Already has Historical data used during training (in config file)
