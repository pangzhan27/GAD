import submitit, functools, transformers
from dataclasses import asdict
from dataclasses import dataclass, field
from transformers import TrainingArguments
import tqdm, os, torchvision
import math, torch
from functools import partial
from torch import nn, Tensor
from torchvision.transforms.functional import normalize
from transformers import AutoModel
from transformers.utils.constants import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD


from transformers import PretrainedConfig

class LiveConfigMixin(PretrainedConfig):
    def __init__(self, *, vision_pretrained: str = None,
        frame_resolution: int = None, frame_token_cls: bool = None, frame_token_pooled: list[int] = None, frame_num_tokens: int = None,
        v_placeholder: str = '<v>', frame_token_interval: str = None, v_placeholder_id: int = None, frame_token_interval_id: int = None,
        stream_loss_weight: float = 1.0, vision_hidden_size=1024, **kwargs
    ):
        super().__init__(**kwargs)
        self.vision_pretrained = vision_pretrained
        self.frame_resolution = frame_resolution
        self.frame_token_cls = frame_token_cls
        self.frame_token_pooled = frame_token_pooled
        self.frame_num_tokens = frame_num_tokens
        self.vision_hidden_size = vision_hidden_size
        self.stream_loss_weight = stream_loss_weight
        self.v_placeholder = v_placeholder
        self.frame_token_interval = frame_token_interval
        self.v_placeholder_id = v_placeholder_id
        self.frame_token_interval_id = frame_token_interval_id


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
    num_gpus: int = 8
    video_dir: str = 'datasets/ego4d/v2/full_scale_2fps_384'
    slurm_partition: str = None


##################################################################################
def _siglip_vision_encode(vision_model: nn.Module, frames: Tensor, frame_token_cls: bool, frame_token_pooled: tuple,
    mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5], rescale_factor=0.00392156862745098, **kwargs):
    frames = normalize(frames * rescale_factor, mean=mean, std=std)
    with torch.cuda.amp.autocast():
        vision_outputs = vision_model(frames)
        last_hidden_state = vision_outputs.last_hidden_state
        if frame_token_pooled:
            s = int(math.sqrt(last_hidden_state.shape[1]))
            spatial_tokens = torch.nn.functional.adaptive_avg_pool2d(
                last_hidden_state.reshape(
                    last_hidden_state.shape[0], s, s, last_hidden_state.shape[-1]
                ).permute(0, 3, 1, 2),
                frame_token_pooled
            ).flatten(2, 3).permute(0, 2, 1)
            if not frame_token_cls:
                return spatial_tokens
        if frame_token_cls:
            cls_token = vision_outputs.pooler_output[:, None]
            if not frame_token_pooled:
                return cls_token
    return torch.cat([cls_token, spatial_tokens], dim=1)

def _clip_vision_encode(vision_model: nn.Module, frames: Tensor, frame_token_cls: bool, frame_token_pooled: tuple,
    mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD, rescale_factor=0.00392156862745098, **kwargs):
    frames = normalize(frames * rescale_factor, mean=mean, std=std)
    with torch.cuda.amp.autocast():
        vision_outputs = vision_model(frames)
        last_hidden_state = vision_outputs.last_hidden_state
        if frame_token_pooled:
            s = int(math.sqrt(last_hidden_state.shape[1]))
            spatial_tokens = torch.nn.functional.adaptive_avg_pool2d(
                last_hidden_state[:,1:].reshape(
                    last_hidden_state.shape[0], s, s, last_hidden_state.shape[-1]
                ).permute(0, 3, 1, 2),
                frame_token_pooled
            ).flatten(2, 3).permute(0, 2, 1)
            if not frame_token_cls:
                return spatial_tokens
        if frame_token_cls:
            cls_token = last_hidden_state[:,0]
            if not frame_token_pooled:
                return cls_token
    return torch.cat([cls_token, spatial_tokens], dim=1)

def build_live_vision(config: LiveConfigMixin):
    model = AutoModel.from_pretrained(config.vision_pretrained).vision_model
    if 'google/siglip-large-patch16-384' == config.vision_pretrained:
        return model, partial(_siglip_vision_encode, frame_token_cls=config.frame_token_cls, frame_token_pooled=config.frame_token_pooled)
    elif 'laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90k' == config.vision_pretrained or 'openai/clip-vit-large-patch14-336' == config.vision_pretrained:
        return model, partial(_clip_vision_encode, config)
    else:
        raise ValueError(f'Unverified vision_pretrained: {config.vision_pretrained}')



def distributed_encode(*, src_root: str, vision_pretrained: str, vision_encode: callable, batch_size: int, embed_mark: str, save_bf16: bool = False, **kwargs):
    env = submitit.JobEnvironment()
    src_root = src_root.rstrip('/')
    model = AutoModel.from_pretrained(vision_pretrained, device_map=f'cuda:{env.local_rank}').vision_model
    model.eval()
    dst_root = f"{src_root}_{embed_mark.split('_')[-1]}_{vision_pretrained.replace('/', '--')}"
    os.makedirs(dst_root, exist_ok=True)
    for i, file in tqdm.tqdm(enumerate(os.listdir(src_root)), desc=f'{src_root} -> {dst_root}'):
        if i % env.num_tasks != env.global_rank:
            continue
        frame_path = os.path.join(src_root, file)
        save_path = os.path.splitext(frame_path)[0] + '.pt'
        save_path = save_path.replace(src_root, dst_root)
        frames = torchvision.io.read_video(frame_path, pts_unit='sec', output_format='TCHW')[0]
        with torch.no_grad():
            frames = torch.cat([vision_encode(model, batch.to(f'cuda:{env.local_rank}')).cpu() for batch in frames.split(batch_size)])
        if save_bf16:
            frames = frames.to(torch.bfloat16)
        torch.save(frames, save_path)



if __name__ == "__main__":
    args, = transformers.HfArgumentParser(LiveOnePlusEncodingArguments).parse_args_into_dataclasses()
    vision_config = LiveConfigMixin(**asdict(args))
    _, vision_encode = build_live_vision(vision_config)
    task = functools.partial(
        distributed_encode, src_root=args.video_dir,
        vision_pretrained=args.vision_pretrained,
        embed_mark=args.embed_mark,
        vision_encode=vision_encode,
        batch_size=256, save_bf16=True
    )
    executor = submitit.AutoExecutor(folder=f"outputs/preprocess/", cluster='local' if args.num_nodes == 1 else 'slurm')
    executor.update_parameters(
        tasks_per_node=args.num_gpus,
        nodes=args.num_nodes,
        gpus_per_node=args.num_gpus,
        cpus_per_task=10,
        slurm_partition=args.slurm_partition,
        mem_gb=240,
        slurm_time='24:00:00',
        timeout_min=600,
    )
    job = executor.submit(task)
    job.results()