#!/bin/bash

cd "$(dirname "$0")"/..
# export name of file as environment variable (to know which script created the run)
export SCRIPT_NAME=$(basename "$0")

# "$@" makes sure to use any extra command line arguments supplied here with bash <script>.sh <args>
python run.py "$@" \
    experiment=climatebench_daily_edm \
    datamodule.mean_over_ensemble='first' \
    datamodule.data_dir=/data \
    datamodule.window=1 \
    datamodule.batch_size=512 \
    datamodule.eval_batch_size=1 \
    datamodule.batch_size_per_gpu=2 \
    datamodule.num_workers=16 \
    datamodule.output_vars='pr' \
    'datamodule.simulations=[ssp126, ssp370, ssp585]' name_suffix="3xSSP-first-ens" \
    datamodule.simulations_anom_type="none" datamodule.simulations_raw="all" datamodule.normalization_type="standard" \
    datamodule.additional_vars=["rsdt"] trainer.num_sanity_val_steps=0 \
    module.num_predictions=3 module.prediction_inputs_noise=0.0 \
    module.enable_inference_dropout=False callbacks.early_stopping=null \
    module.use_ema=True module.ema_decay=0.9999 \
    module.optimizer.lr=4e-4 module.optimizer.weight_decay=0 \
    module.conv_padding_mode_global="circular_width_only" \
    scheduler@module.scheduler=linear_warmup_cosine module.scheduler.warmup_epochs=6 trainer.max_epochs=100 \
    module.monitor='val/crps/pr' \
    trainer.devices=8 \
    model=adm model.loss_function="wmse" \
    model.model_channels=192 model.dropout=0.1 \
    diffusion.loss_function="wmse" diffusion.P_mean=-0.5 diffusion.P_std=1.2 \
    diffusion.sigma_max_inf=400 diffusion.sigma_min=0.02 diffusion.num_steps=16 \
    datamodule.DEBUG_dataset_size=null \
    suffix="raw_stdized+rsdt-edm-adm_debug_test" \
    "$@"

# numworkers=16 uses about 10 cpus in cluster (upper bound) 