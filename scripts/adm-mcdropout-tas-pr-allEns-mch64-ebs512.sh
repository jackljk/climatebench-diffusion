#!/bin/bash

cd "$(dirname "$0")"/../..
# export name of file as environment variable (to know which script created the run)
export SCRIPT_NAME=$(basename "$0")

# "$@" makes sure to use any extra command line arguments supplied here with bash <script>.sh <args>
python run.py "$@" \
    experiment=climatebench_daily_adm \
    datamodule.mean_over_ensemble='all' \
    datamodule.window=1 \
    datamodule.batch_size=512 \
    datamodule.eval_batch_size=8 \
    'datamodule.simulations=[ssp126, ssp370, ssp585]' name_suffix="3xSSP-AllEns-Tas+Pr-pCW" \
    'datamodule.output_vars=[tas, pr]' \
    datamodule.simulations_anom_type="none" datamodule.simulations_raw="all" datamodule.normalization_type="standard_new" \
    datamodule.additional_vars=["rsdt"] trainer.num_sanity_val_steps=0 \
    module.num_predictions=5 module.prediction_inputs_noise=0.0 \
    module.enable_inference_dropout=True callbacks.early_stopping=null \
    module.use_ema=True module.ema_decay=0.9999 \
    module.optimizer.lr=4e-4 module.optimizer.weight_decay=0 \
    module.conv_padding_mode_global="circular_width_only" \
    scheduler@module.scheduler=linear_warmup_cosine module.scheduler.warmup_epochs=3 trainer.max_epochs=100 \
    model=adm model.loss_function="wmse" \
    model.model_channels=64 model.dropout=0.1 \
    datamodule.DEBUG_dataset_size=null \
    suffix="raw_stdized2-ebs512" \
    "$@"

#     datamodule.batch_size=256 is too low