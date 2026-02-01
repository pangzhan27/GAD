### 1. Preprocess Video Features


#### Download/Extract Features
- You can download the extracted features for THUMOS14 and EK100 from [`Testra`](https://github.com/zhaoyue-zephyrus/TeSTra).

- For CrossTask and Ego4DGoalStep, we use DINOV2 for feature extraction. Please refer to [`CMert`](https://github.com/pangzhan27/CMeRT/issues/4) for more details.


###  2. Training and Evaluation

- Use the scripts to perform training/evaluation under [scripts/](scripts/)

- *Training.* Select the method, **Generative, Discriminative or GAD**, by using the corresponding file names. Note that different models require different LoRA components to be fine-tuned, which are specified in each configuration file. 
```
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
```
- *Evaluation.* Assign the path to your finetuned model to replace ```YOUR_CHECKPOINT_FOLDER```

```
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

```
