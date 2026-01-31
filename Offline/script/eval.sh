#!/bin/bash

python generation.py \
       --live_version live1+ \
       --eval_datasets coin_step_test coin_next_test coin_task_test \
       --llm_pretrained meta-llama/Meta-Llama-3-8B-Instruct \
       --per_device_train_batch_size 1 \
       --per_device_eval_batch_size 1 \
       --prediction_loss_only False \
       --dataloader_num_workers 16 \
       --bf16 True \
       --tf32 True \
       --report_to tensorboard \
       --output_dir YOUR_CHECKPOINT_FOLDER \
       --resume_from_checkpoint YOUR_CHECKPOINT_FOLDER

python discriminative.py \
       --live_version live1+ \
       --eval_datasets coin_step_test coin_next_test coin_task_test \
       --llm_pretrained meta-llama/Meta-Llama-3-8B-Instruct \
       --per_device_train_batch_size 1 \
       --per_device_eval_batch_size 1 \
       --prediction_loss_only False \
       --dataloader_num_workers 16 \
       --bf16 True \
       --tf32 True \
       --report_to tensorboard \
       --output_dir YOUR_CHECKPOINT_FOLDER \
       --resume_from_checkpoint YOUR_CHECKPOINT_FOLDER

python gad.py \
       --live_version live1+ \
       --eval_datasets coin_step_test coin_next_test coin_task_test \
       --llm_pretrained meta-llama/Meta-Llama-3-8B-Instruct \
       --per_device_train_batch_size 1 \
       --per_device_eval_batch_size 1 \
       --prediction_loss_only False \
       --dataloader_num_workers 16 \
       --bf16 True \
       --tf32 True \
       --report_to tensorboard \
       --output_dir YOUR_CHECKPOINT_FOLDER \
       --resume_from_checkpoint YOUR_CHECKPOINT_FOLDER
