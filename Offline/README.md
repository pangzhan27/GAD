### 1. Preprocess Video Frames for CION 


### Download Videos
Download the videos using the YouTube IDs listed in coin.json (e.g. via `yt-dlp`) and place them under `coin/videos/`.

#### Install ffmpeg

PyTorch source will make ffmpeg installed, but it is an old version and usually make very low quality preprocessing. Please install newest ffmpeg following:
```sh
wget https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz
tar xvf ffmpeg-release-amd64-static.tar.xz
rm ffmpeg-release-amd64-static.tar.xz
mv ffmpeg-7.0.2-amd64-static ffmpeg
```


#### Sample video frames to 2 FPS and max resolution 384 (with zero padding)

```
python preprocess.ffmpeg --frame_fps 2 --frame_resolution 384 --num_tasks 16 --video_dir datasets/coin/videos
```

- The results will be saved in a new folder with '{fps}fps_{resolution}' suffix. ```datasets/coin/videos -> datasets/coin/videos_2fps_384```.

#### Encode sampled 2fps_384 video frames

```
python preprocess.encode --vision_pretrained google/siglip-large-patch16-384 --video_dir datasets/coin/videos_2fps_384
```

- The results will be saved in a new folder with '{embed_mark}_{model}' suffix.  ```datasets/coin/videos_2fps_384 -> datasets/coin/videos_2fps_384_1+3x3_google--siglip-large-patch16-384```.

###  2. Training and Evaluation

- Use the scripts to perform training/evaluation under [scripts/](scripts/)

- *Training.* Specify the pretrained model using ```--llm_pretrained```, and select the method, **Generative, Discriminative or our GAD**, by using the corresponding file names. Note that different models require different LoRA components to be fine-tuned, which are specified in each configuration file. We trained our model using 2 GPUs with ```--gradient_accumulation_steps``` set to 8, which is equivalent to performing one update per 16 samples. The action step, next-action prediction, and task classification are jointly trained.
```
torchrun --nproc_per_node=2 --standalone generative.py \
     --live_version live1+ \
     --train_datasets coin_step_train coin_next_train coin_task_train \
     --eval_datasets coin_step_test coin_next_test coin_task_test \
     --llm_pretrained meta-llama/Meta-Llama-3-8B-Instruct \
     --num_train_epochs 5 \
     --per_device_train_batch_size 1 \
     --per_device_eval_batch_size 1 \
     --gradient_accumulation_steps 8 \
     --gradient_checkpointing True \
     --eval_strategy no \
     --prediction_loss_only False \
     --save_strategy epoch \
     --save_steps 1 \
     --learning_rate 0.0001 \
     --optim adamw_torch \
     --lr_scheduler_type cosine \
     --warmup_ratio 0.05 \
     --logging_steps 10 \
     --dataloader_num_workers 16 \
     --bf16 True \
     --tf32 True \
     --report_to tensorboard \
     --output_dir outputs/coin_benchmarks/live1+_generation_8B
```
- *Evaluation.* Assign the path to your finetuned model to replace ```YOUR_CHECKPOINT_FOLDER```

```
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

```
