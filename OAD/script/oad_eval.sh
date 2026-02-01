#!/bin/bash

python oad_gad.py  --train_datasets crosstask_wind_ls \
                                       --short_len 20 \
                                       --long_len 128 \
                                       --long_sr 1 \
                                       --visual_dim 1536 \
                                       --eval_strategy no \
                                       --prediction_loss_only False \
                                       --dataloader_num_workers 16 \
                                       --bf16 True \
                                       --tf32 True \
                                       --report_to tensorboard \
                                       --output_dir YOUR_CHECKPOINT_FOLDER \
                                       --criterion CE \
                                       --resume_from_checkpoint YOUR_CHECKPOINT_FOLDER  \
                                       --test True \
                                       --test_set test