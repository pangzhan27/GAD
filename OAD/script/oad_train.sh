#!/bin/bash

# THUMOS14
python oad_gad.py  --train_datasets thumos_wind_ls \
                                        --short_len 32 \
                                        --long_len 128 \
                                        --long_sr 1 \
                                        --stride 32 \
                                        --visual_dim 2048 \
                                        --num_train_epochs 50 \
                                        --group_by_length False \
                                        --group_by_video_num 0 \
                                        --per_device_train_batch_size 16 \
                                        --per_device_eval_batch_size 1 \
                                        --gradient_accumulation_steps 2 \
                                        --gradient_checkpointing False \
                                        --eval_strategy no \
                                        --prediction_loss_only False \
                                        --save_strategy steps \
                                        --save_steps 8150 \
                                        --learning_rate 0.0001 \
                                        --optim adamw_torch \
                                        --lr_scheduler_type cosine \
                                        --warmup_ratio 0.1 \
                                        --logging_steps 10 \
                                        --dataloader_num_workers 16 \
                                        --bf16 True \
                                        --tf32 True \
                                        --report_to tensorboard \
                                        --output_dir outputs/thumos/th-lam-gad_128-32-s32-ag2_CE-1r-4_bz16_w1.0 \
                                        --criterion CE \
                                        --stream_loss_weight 1.0


# CrossTask
python oad_gad.py   --train_datasets crosstask_wind_ls \
                                        --short_len 20 \
                                        --long_len 128 \
                                        --long_sr 1 \
                                        --stride 20 \
                                        --visual_dim 1536 \
                                        --num_train_epochs 50 \
                                        --per_device_train_batch_size 16 \
                                        --per_device_eval_batch_size 1 \
                                        --gradient_accumulation_steps 2 \
                                        --group_by_length False \
                                        --group_by_video_num 0 \
                                        --gradient_checkpointing False \
                                        --eval_strategy no \
                                        --prediction_loss_only False \
                                        --save_strategy steps \
                                        --save_steps 40000 \
                                        --learning_rate 0.0001 \
                                        --optim adamw_torch \
                                        --lr_scheduler_type cosine \
                                        --warmup_ratio 0.1 \
                                        --logging_steps 10 \
                                        --dataloader_num_workers 16 \
                                        --bf16 True \
                                        --tf32 True \
                                        --report_to tensorboard \
                                        --output_dir outputs/crosstask/ct-lam-gad_128-20-s20-ag2_CE-1r-4_bz16_w0.2 \
                                        --criterion CE \
                                        --stream_loss_weight 0.2


# For EK100
python oad_gad.py   --train_datasets ek100_wind_ls \
                                        --short_len 20 \
                                        --long_len 128 \
                                        --long_sr 1 \
                                        --stride 20 \
                                        --visual_dim 1024 \
                                        --num_train_epochs 50 \
                                        --per_device_train_batch_size 16 \
                                        --per_device_eval_batch_size 1 \
                                        --gradient_accumulation_steps 2 \
                                        --group_by_length False \
                                        --group_by_video_num 0 \
                                        --gradient_checkpointing False \
                                        --eval_strategy no \
                                        --prediction_loss_only False \
                                        --save_strategy steps \
                                        --save_steps 80000 \
                                        --learning_rate 0.0001 \
                                        --optim adamw_torch \
                                        --lr_scheduler_type cosine \
                                        --warmup_ratio 0.1 \
                                        --logging_steps 10 \
                                        --dataloader_num_workers 16 \
                                        --bf16 True \
                                        --tf32 True \
                                        --report_to tensorboard \
                                        --output_dir outputs/ek100/ek-lam-gad_128-20-s20-ag2_CE-1r-4_bz16_w0.5 \
                                        --criterion CE \
                                        --stream_loss_weight 0.5 \

# For Ego4DGoalStep
python oad_gad.py  --train_datasets ego4dgoal_wind_ls \
                                       --short_len 16 \
                                       --long_len 128 \
                                       --long_sr 1 \
                                       --stride 16 \
                                       --visual_dim 1536 \
                                       --num_train_epochs 20 \
                                       --per_device_train_batch_size 16 \
                                       --per_device_eval_batch_size 1 \
                                       --gradient_accumulation_steps 2 \
                                       --group_by_length False \
                                       --group_by_video_num 0 \
                                       --gradient_checkpointing False \
                                       --eval_strategy no \
                                       --prediction_loss_only False \
                                       --save_strategy steps \
                                       --save_steps 30000 \
                                       --learning_rate 0.0001 \
                                       --optim adamw_torch \
                                       --lr_scheduler_type cosine \
                                       --warmup_ratio 0.1 \
                                       --logging_steps 10 \
                                       --dataloader_num_workers 16 \
                                       --bf16 True \
                                       --tf32 True \
                                       --report_to tensorboard \
                                       --output_dir outputs/ego4dgoal/edg-lam-gad_128-16-s16-ag2_CE-1r-4_bz16_w0.5 \
                                       --criterion CE \
                                       --stream_loss_weight 0.5