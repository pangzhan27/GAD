from functools import partial
import submitit, transformers
from dataclasses import dataclass, field
from transformers import TrainingArguments
import tqdm, os, subprocess,  pathlib, submitit

@dataclass
class LiveTrainingArguments(TrainingArguments):
    live_version: str = 'live1+'
    system_prompt: str = (
        "A multimodal AI assistant is helping users with some activities."
        " Below is their conversation, interleaved with the list of video frames received by the assistant."
    )
    train_datasets: list[str] = None
    eval_datasets: list[str] = None
    stream_loss_weight: float = 1.0
    llm_pretrained: str = 'meta-llama/Llama-3.2-1B-Instruct' #'meta-llama/Meta-Llama-3-8B-Instruct'#
    vision_pretrained: str = 'google/siglip-large-patch16-384'
    lora_modules: str = "model.*(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)|lm_head$"
    lora_r: int = 128
    lora_alpha: int = 256
    finetune_modules: list[str] = field(default_factory=lambda: ['connector'])
    frame_fps: int = 2 # for training. inference can be 10
    frame_token_cls: bool = None
    frame_token_pooled: list[int] = None
    frame_resolution: int = 384
    frame_token_interval: str  = None
    frame_token_interval_threshold: float = 0.0
    augmentation: bool = False
    attn_implementation: str = 'flash_attention_2'
    output_dir: str = 'outputs/debug'

@dataclass
class LiveOnePlusTrainingArguments(LiveTrainingArguments):
    live_version: str = 'live1+'
    frame_token_cls: bool = True
    frame_token_pooled: list[int] = field(default_factory=lambda: [3,3])
    frame_num_tokens: int = 10 # 1+3x3
    embed_mark: str = '2fps_384_1+3x3'
    frame_token_interval: str = ','
    max_num_frames: int = 1200 # 10min, 2fps, 1200 frames

@dataclass
class LiveOnePlusEncodingArguments(LiveOnePlusTrainingArguments):
    num_nodes: int = 1
    num_tasks: int = 16
    video_dir: str = 'datasets/ego4d/v2/full_scale'
    slurm_partition: str = None


def ffmpeg_once(src_path: str, dst_path: str, *, fps: int = None, resolution: int = None, pad: str = '#000000', mode='bicubic'):
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    command = [
        'ffmpeg',
        '-y',
        '-sws_flags', mode,
        '-i', src_path,
        '-an',
        '-threads', '10',
    ]
    if fps is not None:
        command += ['-r', str(fps)]
    if resolution is not None:
        command += ['-vf', f"scale='if(gt(iw\\,ih)\\,{resolution}\\,-2)':'if(gt(iw\\,ih)\\,-2\\,{resolution})',pad={resolution}:{resolution}:(ow-iw)/2:(oh-ih)/2:color='{pad}'"]
    command += [dst_path]
    subprocess.run(command, check=True)

def distributed_ffmpeg(*, src_root: str, fps: int = None, resolution: int = None, pad: str = '#000000', mode='bicubic'):
    import submitit
    env = submitit.JobEnvironment()
    src_root = src_root.rstrip('/')
    pather = pathlib.Path(src_root)
    src_paths = [str(path) for path in pather.rglob('*') if path.is_file() and str(path).endswith('.mp4')]
    dst_root = src_root
    if fps is not None:
        dst_root += f'_{fps}fps'
    if resolution is not None:
        assert (pad is not None)
        dst_root += f'_{resolution}'
    for i, src_path in tqdm.tqdm(enumerate(src_paths), desc=f'{src_root} -> {dst_root}'):
        if i % env.num_tasks != env.global_rank:
            continue
        dst_path = src_path.replace(src_root, dst_root)
        ffmpeg_once(src_path, dst_path, fps=fps, resolution=resolution, pad=pad, mode=mode)


if __name__ == "__main__":
    args, = transformers.HfArgumentParser(LiveOnePlusEncodingArguments).parse_args_into_dataclasses()
    executor = submitit.AutoExecutor(folder=f"outputs/preprocess/", cluster='local' if args.num_nodes == 1 else 'slurm')
    task = partial(distributed_ffmpeg, src_root=args.video_dir, resolution=args.frame_resolution, fps=args.frame_fps)
    executor.update_parameters(
        tasks_per_node=args.num_tasks,
        nodes=args.num_nodes,
        slurm_partition=args.slurm_partition,
        cpus_per_task=10,
        mem_gb=240,
        slurm_time='24:00:00',
        timeout_min=600,
    )
    job = executor.submit(task)
    job.results()
