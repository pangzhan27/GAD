from dataclasses import asdict
from transformers import HfArgumentParser
from dataclasses import dataclass, field
from transformers import TrainingArguments
from transformers import LlamaForCausalLM, AutoModelForCausalLM
from transformers.activations import GELUActivation
from transformers import LlamaConfig
from transformers import PretrainedConfig, EvalPrediction
import math, torch, sys
from functools import partial
from torch import nn, Tensor
from torchvision.transforms.functional import normalize
from transformers import AutoModel
from transformers.utils.constants import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import AutoTokenizer
from torch.utils.data import ConcatDataset
from transformers import Trainer, PreTrainedTokenizer
from transformers.trainer_pt_utils import LabelSmoother
import os, json, tqdm, random
import numpy as np
import Levenshtein

from transformers.utils import logging
logger = logging.get_logger(__name__)

####################################################### args ######################@@@@@@@@@@@#######################
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
    llm_pretrained: str = 'meta-llama/Meta-Llama-3-8B-Instruct'#
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
class LiveOneTrainingArguments(LiveTrainingArguments):
    live_version: str = 'live1'
    frame_token_cls: bool = True
    frame_num_tokens: int = 1
    frame_token_interval: str  = ''
    embed_mark: str = '2fps_384_1+3x3' #'2fps_384_1'
    max_num_frames: int = 7200 # 1h, 2fps, 7200 frames

@dataclass
class LiveOnePlusTrainingArguments(LiveTrainingArguments):
    live_version: str = 'live1+'
    frame_token_cls: bool = True
    frame_token_pooled: list[int] = field(default_factory=lambda: [3,3])
    frame_num_tokens: int = 10 # 1+3x3
    embed_mark: str = '2fps_384_1+3x3'
    frame_token_interval: str = ','
    max_num_frames: int = 1200 # 10min, 2fps, 1200 frames

def get_args_class(live_version: str):
    if live_version == 'live1':
        return LiveOneTrainingArguments
    elif live_version == 'live1+':
        return LiveOnePlusTrainingArguments
    raise NotImplementedError

def parse_args() -> LiveTrainingArguments:
    args, = HfArgumentParser(LiveTrainingArguments).parse_args_into_dataclasses()
    args, = HfArgumentParser(get_args_class(args.live_version)).parse_args_into_dataclasses()
    return args

################################################### vision encoder ###################################################
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

####################################################### configure ######################################################

class LiveLlamaConfig(LlamaConfig, LiveConfigMixin):
    pass

####################################################### model ######################################################
class LiveMixin(AutoModelForCausalLM):
    def set_vision_inside(self):
        logger.warning_once("!!! Set vision encoder in the model, only recommended for on in-the-wild inference. "
            "Please dont call this for efficient training & evaluation. Instead, do visual feature pre-extraction.")
        self.vision_encoder, self.vision_encode = build_live_vision(self.config)

    def unset_vision_inside(self):
        del self.vision_encoder
        del self.vision_encode

    def visual_embed(self, frames: torch.Tensor):
        if hasattr(self, 'vision_encode'):
            with torch.cuda.amp.autocast():
                frames = self.vision_encode(self.vision_encoder, frames)
            frames = frames.to(self.dtype)
        frames = self.connector(frames)
        return frames.view(-1, frames.shape[-1])

    def joint_embed(
        self,
        input_ids: torch.Tensor = None,
        frames: torch.Tensor = None,
    ):
        if frames is None:
            return self.get_input_embeddings()(input_ids)
        if input_ids is None:
            return self.visual_embed(frames)
        inputs_embeds = self.get_input_embeddings()(input_ids.clamp(max=self.vocab_size-1))
        v_mask = input_ids == self.config.v_placeholder_id
        if v_mask.any():
            inputs_embeds[v_mask] = self.visual_embed(frames)
        return inputs_embeds

    @torch.no_grad()
    def stream_evaluate(
        self,
        input_ids: torch.LongTensor,
        labels: torch.LongTensor,
        frames: torch.Tensor,
        ignore_token_id: int = -100,
        frame_token_interval_threshold: float = 0.0,
        **kwargs
    ):
        # 0. evaluation only supports batch_size = 1
        assert input_ids.size(0) == labels.size(0) == 1
        input_id, label = input_ids[0], labels[0]
        device = input_id.device
        zero = torch.tensor(0, dtype=torch.int, device=device)
        one = torch.tensor(1, dtype=torch.int, device=device)

        # 1. prepare multi-turn start and stop
        turn_stops = ((input_id == self.config.eos_token_id).nonzero() + 1)[:,0].tolist()
        turn_starts = [0] + turn_stops[:-1]
        num_turns = len(turn_starts)

        # 2. forward the full input_ids and labels, get tokenwise logits and losses
        outputs = self.forward(input_ids=input_ids, frames=frames, return_dict=True, use_cache=True)
        logit, past_key_values = outputs.logits[0], outputs.past_key_values

        # 3. compute metrics for each turn
        v_placeholder_id = self.config.v_placeholder_id
        use_interval = self.config.frame_token_interval_id is not None
        frame_token_interval_id = self.config.frame_token_interval_id if use_interval else self.config.eos_token_id
        frame_num_tokens = self.config.frame_token_cls
        if self.config.frame_token_pooled:
            frame_num_tokens += self.config.frame_token_pooled[0] * self.config.frame_token_pooled[1]
        past_num_frames = 0
        lm_ppls, frame_diffs, fluencies, lm_correctness = [], [], [], []
        for r, (turn_start, turn_stop) in enumerate(zip(turn_starts, turn_stops)):
            ## 3.1. we only have two losses: stream loss on frame tokens, and lm loss. prepare corresponding mask according two losses
            turn_label = label[turn_start:turn_stop]
            turn_learn_mask = turn_label != ignore_token_id
            if not turn_learn_mask.any():
                continue
            turn_logit = logit[turn_start:turn_stop]
            turn_input_id = input_id[turn_start:turn_stop]
            turn_v_mask = turn_input_id == v_placeholder_id
            turn_num_frames = turn_v_mask.sum() // frame_num_tokens
            turn_stream_mask = turn_v_mask & turn_learn_mask
            turn_lm_mask = turn_learn_mask & ~turn_stream_mask

            ## 3.2 ppl, offline metric
            if turn_lm_mask.any():
                turn_lm_masked_logit, turn_lm_masked_label = turn_logit[turn_lm_mask], turn_label[turn_lm_mask]
                lm_ppl = torch.nn.functional.cross_entropy(turn_lm_masked_logit, turn_lm_masked_label).exp()
                lm_ppls.append(lm_ppl)
                turn_lm_masked_wrong_mask = turn_lm_masked_logit.argmax(dim=-1) != turn_lm_masked_label
                if turn_lm_masked_wrong_mask.any():
                    num_lm_correct_tokens = turn_lm_masked_wrong_mask.nonzero()[0,0]
                else:
                    num_lm_correct_tokens = (~turn_lm_masked_wrong_mask).sum()
                lm_correctness.append(num_lm_correct_tokens / turn_lm_masked_label.numel())

            ## 3.3. frame_diff (will be casted to time_diff in compute_metrics)
            if turn_stream_mask.any():
                ## 3.3.1: reply before (at) turn_num_frames
                turn_score = turn_logit.softmax(dim=-1)
                turn_stream_masked_score = turn_score[turn_stream_mask]
                if frame_token_interval_threshold > 0:
                    lower_threshold_mask = turn_stream_masked_score[:, frame_token_interval_id] < frame_token_interval_threshold
                    turn_stream_masked_score[lower_threshold_mask] = 0
                turn_stream_masked_pred_mask = turn_stream_masked_score.argmax(dim=-1) != frame_token_interval_id
                if turn_stream_masked_pred_mask.any():
                    frame_diff = turn_stream_mask.sum() - turn_stream_masked_pred_mask.nonzero()[0,0] - 1
                else:
                    ## 3.3.2: the most complex part,reply after turn_num_frames. we assume the 'assistant: ...' not exists
                    turn_last_stream_idx = turn_stream_mask.nonzero()[-1,0]
                    past_key_values_before_assistant = self.trim_past_key_values(past_key_values, 0, turn_start + turn_last_stream_idx + 1)
                    if r == num_turns - 1: # no future frame. we assume the model should receive a signal when streaming ends (e.g. close button).
                        frame_diff = zero
                    else:
                        next_turn_num_frames = (input_id[turn_starts[r+1]:turn_stops[r+1]] == v_placeholder_id).sum() // frame_num_tokens
                        to_append_num_frames = min(next_turn_num_frames, turn_num_frames - 1) # avoid bias. current as center, two equal left/right side
                        if to_append_num_frames == 0:
                            frame_diff = zero
                        else:
                            to_append_frames = frames[past_num_frames+turn_num_frames:past_num_frames+turn_num_frames+to_append_num_frames]
                            frame_placeholder = [v_placeholder_id] * frame_num_tokens
                            if use_interval:
                                frame_placeholder = [frame_token_interval_id] + frame_placeholder
                            to_append_input_id = torch.tensor(frame_placeholder * to_append_num_frames, dtype=torch.long, device=device)
                            to_append_logit = self.forward(
                                input_ids=to_append_input_id[None],
                                past_key_values=past_key_values_before_assistant,
                                frames=to_append_frames,
                                return_dict=True, use_cache=True
                            ).logits[0]
                            # we only use the last idx of each frame
                            idxs = torch.arange(len(frame_placeholder)-1, len(to_append_input_id), len(frame_placeholder), device=device)
                            to_append_score = to_append_logit[idxs].softmax(dim=-1)
                            if frame_token_interval_threshold > 0:
                                lower_threshold_mask = to_append_score[:, frame_token_interval_id] < frame_token_interval_threshold
                                to_append_score[lower_threshold_mask] = 0
                            to_append_score_pred_mask = to_append_score.argmax(dim=-1) != frame_token_interval_id
                            if to_append_score_pred_mask.any():
                                frame_diff = -(to_append_score_pred_mask.nonzero()[0,0] + 1)
                            else:
                                frame_diff = -to_append_num_frames
                frame_diffs.append(frame_diff.abs())

            ## 2.6 fluency
            if turn_lm_mask.any() and turn_stream_mask.any():
                num_learn_v_tokens = turn_stream_mask.sum()
                num_learn_valid_tokens = turn_lm_masked_label.numel() + num_learn_v_tokens
                if frame_diff == 0:
                    fluency = (num_learn_v_tokens + num_lm_correct_tokens) / num_learn_valid_tokens
                elif frame_diff > 0:
                    fluency = (num_learn_v_tokens - frame_diff) / num_learn_valid_tokens
                else:
                    fluency = (num_learn_v_tokens - 1) / num_learn_valid_tokens
                fluencies.append(fluency)
            ## 2.7 next turn
            past_num_frames += turn_num_frames
        lm_ppl = torch.stack(lm_ppls).mean() if lm_ppls else one
        frame_diff = torch.stack(frame_diffs).float().mean() if frame_diffs else zero
        fluency = torch.stack(fluencies).float().mean() if fluencies else one
        lm_correctness = torch.stack(lm_correctness).float().mean() if lm_correctness else one
        return torch.stack([lm_ppl, frame_diff, fluency, lm_correctness])

    def trim_past_key_values(self, past_key_values, start, stop):
        return [[past_keys[:,:,start:stop], past_values[:,:,start:stop]] for past_keys, past_values in past_key_values]


class LiveLlamaForCausalLM(LlamaForCausalLM, LiveMixin):
    config_class = LiveLlamaConfig
    _keys_to_ignore_on_load_missing = ['vision_encoder', 'connector']

    def __init__(self, config: LiveLlamaConfig):
        super().__init__(config)
        self.connector = torch.nn.Sequential(
            torch.nn.Linear(config.vision_hidden_size, config.hidden_size, bias=True),
            GELUActivation(config.hidden_size),
            torch.nn.Linear(config.hidden_size, config.hidden_size, bias=True),
        )

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            frames: torch.FloatTensor = None,
            attention_mask: torch.Tensor = None,
            position_ids: torch.LongTensor = None,
            past_key_values: list[torch.FloatTensor] = None,
            inputs_embeds: torch.FloatTensor = None,
            labels: torch.LongTensor = None,
            use_cache: bool = None,
            output_attentions: bool = None,
            output_hidden_states: bool = None,
            return_dict: bool = None,
            cache_position: torch.LongTensor = None,
            **kwargs,
    ):
        if inputs_embeds is None:
            inputs_embeds = self.joint_embed(input_ids, frames)
        outputs = super().forward(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            # labels
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        loss = None
        if labels is not None:
            logits = outputs[0]
            v_mask = input_ids.flatten(0, 1) == self.config.v_placeholder_id
            weight = v_mask * self.config.stream_loss_weight + ~v_mask
            loss = nn.functional.cross_entropy(logits.flatten(0, 1), labels.flatten(), reduction='none') * weight
            loss = loss.sum() / (labels >= 0).sum()

        if not return_dict:
            return (loss,) + outputs[1:] if loss is not None else outputs

        outputs.loss = loss
        return outputs

    def generate_after_embed(self, input_ids, frames, **kwargs):
        return super().generate(inputs_embeds=self.joint_embed(input_ids, frames), **kwargs)


####################################################### tokenizer ######################################################
def get_stream_placeholder_len(num_frames: int, model_config: LiveConfigMixin) -> str:
    return num_frames * model_config.frame_num_tokens * len(model_config.v_placeholder) + len(model_config.frame_token_interval) * (num_frames - 1)

def get_stream_placeholder_jinja2(model_config: LiveConfigMixin) -> str:
    return f"'{model_config.frame_token_interval}'.join([{model_config.frame_num_tokens} * '{model_config.v_placeholder}'] * message['num_frames'])"

def get_stream_learn_ranges(num_frames: int, model_config: LiveConfigMixin) -> torch.Tensor:
    len_frame_placeholder_with_interval = model_config.frame_num_tokens * len(model_config.v_placeholder) + len(model_config.frame_token_interval)
    intermediate_interval_idxs = torch.arange(
        len_frame_placeholder_with_interval,
        len_frame_placeholder_with_interval * num_frames + 1,
        len_frame_placeholder_with_interval
    ) - len(model_config.frame_token_interval)
    len_learn = len(model_config.frame_token_interval) if model_config.frame_token_interval else len(model_config.v_placeholder)
    learn_ranges = torch.stack([
        intermediate_interval_idxs,
        intermediate_interval_idxs + len_learn
    ], dim=1)
    return learn_ranges

def chat_template(self, stream_placeholder_jinja2: str):
    """
    system prompt
    [<v>,<v>,<v>]
    User: ...
    Assistant: ...</s>
    [<v>,<v>]
    Assistant: ...</s>
    User: ...
    Assistant: ...</s>
    """
    template = (
        "{% if messages[0]['role'] == 'system' %}"
        "{{ bos_token + messages[0]['content'] + '\n' }}" # system
        "{% set messages = messages[1:] %}"
        "{% endif %}"
        "{% for message in messages %}"
        "{% if message['role'] == 'user' %}"
        "{% if add_stream_query_prompt %}"
        "{{ ']\nUser: ' + message['content'] }}"
        "{% else %}"
        "{{ '\nUser: ' + message['content'] }}"
        "{% endif %}"
        "{% elif message['role'] == 'assistant' %}"
        "{{ '\nAssistant: '  + message['content'] + eos_token }}"
        "{% elif message['role'] == 'stream' and message['num_frames'] > 0: %}"
        "{{ '\n[' + STREAM_PLACEHOLDER + ']' }}"
        "{% endif %}"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
        "{{ '\nAssistant:' }}"
        "{% elif add_stream_prompt %}"
        "{{ '\n[' }}"
        "{% elif add_stream_generation_prompt %}"
        "{{ ']\nAssistant:' }}"
        "{% endif %}"
    )
    template = template.replace('STREAM_PLACEHOLDER', stream_placeholder_jinja2)
    return template

def chat_template_transition(tokenizer):
    return {
        (None, 'system'): tokenizer.bos_token,
        ('system', 'user'): '\n\nUser: ',
        ('system', 'stream'): '\n\n[',
        ('user', 'assistant'): '\nAssistant: ',
        ('user', 'stream'): '\n[',
        ('user', 'user'): '\nUser: ',
        ('assistant', 'user'): f'{tokenizer.eos_token}\nUser: ',
        ('assistant', 'stream'): f'{tokenizer.eos_token}\n[',
        ('stream', 'user'): ']\nUser: ',
        ('stream', 'assistant'): ']\nAssistant: ',
        'assistant': 'Assistant: ',
        'eos_token': tokenizer.eos_token,
    }

def chat_template_offsets(tokenizer):
    return {k:len(v) for k, v in chat_template_transition(tokenizer).items()}

def get_learn_ranges(conversation: list[dict], *, chat_template_offsets: dict[tuple, int], model_config: LiveConfigMixin):
    offset = 0
    learn_ranges = []
    last_role = None
    for message in conversation:
        role = message['role']
        offset += chat_template_offsets[(last_role, role)]
        last_role = role
        if role == 'stream':
            if message.get('learn', False):
                ranges = get_stream_learn_ranges(message['num_frames'], model_config) + offset
                # the last one has ]\n, should also consider \n
                ranges[-1, 1] += 1
                if not isinstance(message['learn'], bool):
                    ranges = ranges[:message['learn']]
                learn_ranges.extend([range(r[0], r[1]) for r in ranges])
            offset += get_stream_placeholder_len(message['num_frames'], model_config)
        else:
            if role == 'assistant':
                if message.get('learn', False):
                    learn_ranges.append(range(offset - chat_template_offsets['assistant'], offset + len(message['content']) + chat_template_offsets['eos_token']))
            offset += len(message['content'])
    return learn_ranges


def build_live_tokenizer_and_update_config(llm_pretrained: str, model_config: LiveConfigMixin) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(llm_pretrained, use_fast=True, padding_side='left')
    tokenizer.add_special_tokens({'additional_special_tokens': [model_config.v_placeholder]})
    v_placeholder_id = len(tokenizer) - 1
    if model_config.frame_token_interval:
        frame_token_interval_id = tokenizer.convert_tokens_to_ids(model_config.frame_token_interval)
    else:
        frame_token_interval_id = None
    tokenizer.pad_token = tokenizer.eos_token
    model_config.update(dict(v_placeholder_id=v_placeholder_id, frame_token_interval_id=frame_token_interval_id, eos_token_id=tokenizer.eos_token_id))
    tokenizer.chat_template = chat_template(tokenizer, get_stream_placeholder_jinja2(model_config))
    tokenizer.get_learn_ranges = partial(get_learn_ranges, chat_template_offsets=chat_template_offsets(tokenizer), model_config=model_config)
    return tokenizer

####################################################### tokenizer + model ######################################################
def build_live(*, is_training: bool, config_class: type, model_class: type, llm_pretrained: str = None,
    finetune_modules: list[str] = None, lora_modules: str = None, lora_r: int = None, lora_alpha: int = None, set_vision_inside: bool = False,
    resume_from_checkpoint: str = '', attn_implementation: str = 'flash_attention_2', torch_dtype: str | torch.dtype = 'auto', **kwargs):

    a = config_class.from_pretrained(llm_pretrained, **kwargs)
    b = config_class()
    c = config_class.from_pretrained(llm_pretrained)
    model = model_class.from_pretrained(llm_pretrained, config=config_class.from_pretrained(llm_pretrained, **kwargs), torch_dtype=torch_dtype, attn_implementation=attn_implementation)
    tokenizer = build_live_tokenizer_and_update_config(llm_pretrained, model.config)
    if is_training:
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_modules,
            lora_dropout=0.05,
            task_type="CAUSAL_LM",
            modules_to_save=finetune_modules,
            inference_mode=False,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    else:
        if resume_from_checkpoint:
            model = PeftModel.from_pretrained(model, resume_from_checkpoint, is_trainable=False)
        else:
            logger.warning(f'!!! Fail to load checkpoint: {resume_from_checkpoint}. Return a new initialized model.')
        if set_vision_inside:
            model.set_vision_inside()
        model.requires_grad_(False)
    return model, tokenizer


def build_model_and_tokenizer(**kwargs):
    return build_live(config_class=LiveLlamaConfig, model_class=LiveLlamaForCausalLM, **kwargs)

####################################################### data ######################################################
def ceil_time_by_fps(time: float, fps: int, min_time: float, max_time: float):
    return min(max(math.ceil(time * fps) / fps, min_time), max_time)

def rand_bool():
    return bool(random.getrandbits(1))

class DictWithTo(dict):
    def to(self, *args, **kwargs):
        return self

class StreamMixIn(torch.utils.data.Dataset):
    def __init__(self, is_training: bool, system_prompt: str, augmentation: bool, max_num_frames: int,
                 tokenizer: PreTrainedTokenizer, **kwargs):
        super().__init__()
        self.is_training = is_training
        self.system_prompt = system_prompt
        self.augmentation = augmentation
        self.tokenizer = tokenizer
        self.max_num_frames = max_num_frames
        assert system_prompt is not None, 'Please add a system prompt'

    # NOTE: this augmentation is to reduce the text dependency
    def augment(self, conversation):
        if not self.augmentation or not self.is_training:
            return conversation
        assistant_messages = [(i, message) for i, message in enumerate(conversation) if
                              message['role'] == 'assistant' and message.get('learn', False)]
        if len(assistant_messages) <= 1:
            return conversation
        i, assistant_message_i = random.choice(
            assistant_messages[:-1])  # do not choose the last one, since its meaningless to dependency
        real_content = assistant_message_i['content']
        fake_contents = list(
            set(message['content'] for _, message in assistant_messages if message['content'] != real_content)) + [
                            ''] + [None]
        fake_content = random.choice(fake_contents)
        fake_message_i = {'role': 'assistant', 'content': fake_content,
                          'learn': False} if fake_content is not None else None
        if rand_bool():  # fix the wrong content at the next frame
            # case1: ... fake_message, frame, real_message, stream - 1 ...
            if fake_message_i is not None and conversation[i + 1]['role'] == 'stream' and conversation[i + 1][
                'num_frames'] > 1:
                conversation = conversation[:i] + [
                    fake_message_i,
                    {'role': 'stream', 'num_frames': 1, 'learn': True},
                    {'role': 'assistant', 'content': f'(Sorry, the last response is wrong) {real_content}',
                     'learn': True},
                    {'role': 'stream', 'num_frames': conversation[i + 1]['num_frames'] - 1, 'learn': True}
                ] + conversation[i + 2:]
            # case2: ... stream + 1, real_message, stream -1, ...
            elif fake_message_i is None and conversation[i - 1]['role'] == 'stream' and conversation[i + 1][
                'role'] == 'stream' and conversation[i + 1]['num_frames'] > 1:
                conversation = conversation[:i - 1] + [
                    {'role': 'stream', 'num_frames': conversation[i - 1]['num_frames'] + 1,
                     'learn': conversation[i - 1]['num_frames'] - 1},
                    {'role': 'assistant', 'content': real_content, 'learn': True},
                    {'role': 'stream', 'num_frames': conversation[i + 1]['num_frames'] - 1, 'learn': True}
                ] + conversation[i + 2:]
        else:  # not fix
            # case3: ... fake_message, stream (unlearn) / message ...
            if fake_message_i is not None:
                if conversation[i + 1]['role'] == 'stream':
                    conversation = conversation[:i] + [
                        fake_message_i,
                        {'role': 'stream', 'num_frames': conversation[i + 1]['num_frames'], 'learn': False},
                    ] + conversation[i + 2:]
                else:
                    conversation = conversation[:i] + [fake_message_i] + conversation[i + 1:]
            # case4: ... stream (learn-1), stream (unlearn) / message ...
            else:
                if conversation[i - 1]['role'] == 'stream':
                    if conversation[i + 1]['role'] != 'stream':
                        conversation = conversation[:i - 1] + [
                            {'role': 'stream', 'num_frames': conversation[i - 1]['num_frames'],
                             'learn': conversation[i - 1]['num_frames'] - 1},
                        ] + conversation[i + 1:]
                    else:
                        conversation = conversation[:i - 1] + [
                            {'role': 'stream',
                             'num_frames': conversation[i - 1]['num_frames'] + conversation[i + 1]['num_frames'],
                             'learn': conversation[i - 1]['num_frames'] - 1},
                        ] + conversation[i + 2:]
                else:
                    if conversation[i + 1]['role'] == 'stream':
                        conversation = conversation[:i] + [
                            {'role': 'stream', 'num_frames': conversation[i + 1]['num_frames'], 'learn': False},
                        ] + conversation[i + 2:]
                    else:
                        conversation = conversation[:i] + conversation[i + 1:]
        return conversation

    def max_frames_clip(self, conversation: list[dict], load_ranges: dict[str, range], max_num_frames: int):
        cum_num_frames = 0
        for i, message in enumerate(conversation):
            if message['role'] == 'stream':
                if cum_num_frames + message['num_frames'] > max_num_frames:
                    conversation = conversation[:i]
                    load_ranges = {path: range(ranger.start, ranger.start + cum_num_frames) for path, ranger in
                                   load_ranges.items()}
                    break
                cum_num_frames += message['num_frames']
        return conversation, load_ranges

    def __getitem__(self, *, conversation: list[dict], load_ranges: dict[str, range] | torch.Tensor = None,
                    add_generation_prompt=False, **kwargs):
        # 1. load visual encoding
        if isinstance(load_ranges, torch.Tensor):
            frames = load_ranges
        elif load_ranges is not None:
            conversation, load_ranges = self.max_frames_clip(conversation, load_ranges, self.max_num_frames)
            frames = torch.cat([torch.load(path, weights_only=True)[ranger] for path, ranger in load_ranges.items()])
        else:
            frames = torch.tensor([])
        # 2. prepare texts
        if self.augmentation:
            conversation = self.augment(conversation)
        conversation = [{"role": "system", "content": self.system_prompt}] + conversation
        text = self.tokenizer.apply_chat_template(conversation, tokenize=False,
                                                  add_generation_prompt=add_generation_prompt)
        # 3. learn ranges
        learn_ranges = self.tokenizer.get_learn_ranges(conversation) if not add_generation_prompt else []
        # print(text)
        # for i in range(len(learn_ranges)):
        #     print(i, text[learn_ranges[i].start: learn_ranges[i].stop])
        return text, frames, learn_ranges

class COIN:
    root = 'datasets/coin'
    video_root = os.path.join(root, 'videos')
    anno_root = os.path.join(root, 'annotations')

    def __init__(self, split: str, vision_pretrained: str, embed_mark: str, frame_fps: int, **kwargs):
        super().__init__(**kwargs)
        self.embed_dir = f"{self.video_root}_{embed_mark}_{vision_pretrained.replace('/', '--')}"
        self.frame_fps = frame_fps
        self.metadata = self.get_metadata()
        annos = json.load(open(os.path.join(self.root, 'coin.json')))['database']
        assert split in ['train', 'test']
        self._annos = [{
            'video_uid': video_uid,
            'task': COIN._clean_task(anno['class']),
            'start': anno['start'],
            'end': anno['end'],
            'steps': [dict(
                start=step['segment'][0],
                end=step['segment'][1],
                text=COIN._clean_step(step['label']),
            ) for step in anno['annotation']],
        } for video_uid, anno in annos.items() if (split in anno['subset'].lower()) and (video_uid in self.metadata)]
        self.task_categories = list(set([v['task'].lower() for v in self._annos]))
        self.step_categories = list(set([step['text'].lower() for steps in self._annos for step in steps['steps']]))
        self.annos: list[dict]

    def get_metadata(self, ):
        metadata_path = f'{self.embed_dir}_metadata.json'
        if os.path.exists(metadata_path):
            print(f'load {metadata_path}...')
            metadata = json.load(open(metadata_path))
        else:
            metadata = {}
            for file in tqdm.tqdm(os.listdir(self.embed_dir), desc=f'prepare {metadata_path}...'):
                path = os.path.join(self.embed_dir, file)
                duration = (len(torch.load(path)) - 1) / self.frame_fps
                key = os.path.splitext(os.path.basename(path))[0]
                metadata[key] = {'duration': duration, 'path': path}
            json.dump(metadata, open(metadata_path, 'w'), indent=4)
        return metadata

    @staticmethod
    def _clean_step(step):
        replaces = {
            'process (crop, fold) paper': 'crop and fold paper',
            'try to press gun head, spray residual old grease': 'try to press gun head to spray residual old grease'
        }
        return replaces.get(step, step)

    # PutOnHair -> put on hair
    @staticmethod
    def _clean_task(text):
        result = ''
        for char in text:
            if char.isupper():
                result += ' ' + char.lower()
            else:
                result += char
        result = result.replace(' t v', ' TV')
        result = result.replace(' c d', ' CD')
        result = result.replace('s i m', 'SIM')
        result = result.replace('n b a', 'NBA')
        result = result.replace('s s d', 'SSD')
        result = result.replace('r j45', 'RJ45')
        return result.strip()

    def __len__(self):
        return len(self.annos)

class COINBenchmark(COIN, StreamMixIn):
    evaluation_kwargs = DictWithTo(evaluator='generate_after_embed', max_new_tokens=512, do_sample=False, use_cache=True, temperature=1.0, top_p=1.0)

    @staticmethod
    def fuzzy_match(text, choices):
        return min([(Levenshtein.distance(text, choice), choice) for choice in choices])[1]

    def compute_metrics(self, eval_predictions: EvalPrediction, tokenizer: PreTrainedTokenizer, **kwargs):
        batch_pred_tensor, sample_idxs = eval_predictions.predictions, eval_predictions.label_ids
        batch_pred_tensor[batch_pred_tensor < 0] = tokenizer.bos_token_id # not use clamp(min=0), since 0 is ! in Llama-3 tokenizer and may affect matching
        predictions = tokenizer.batch_decode(batch_pred_tensor, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        correct = 0
        for prediction, label in zip(predictions, self.labels[sample_idxs]): # should be self.labels[sample_idx] to get the correct order
            prediction = prediction.lower().rstrip('.')
            if prediction == label or self.fuzzy_match(prediction, self.categories) == label:
                correct += 1
        return dict(accuracy=correct / len(predictions) * 100) # * 100

    def __getitem__(self, index):
        anno = self.annos[index]
        conversation = anno['conversation'] if self.is_training else anno['conversation'][:-1] # if not training, do not include the assistant message
        return *super().__getitem__(conversation=conversation, load_ranges=anno['load_ranges'], add_generation_prompt=not self.is_training), index, self.evaluation_kwargs

class COINStep(COINBenchmark):
    user_message = {
        "role": "user",
        "content": 'What is the action in the video? Format your answer concisely. No extra text output.'
    }
    def __init__(self, *, split: str, frame_fps: int, is_training: bool, **kwargs):
        super().__init__(split=split, frame_fps=frame_fps, is_training=is_training, **kwargs)
        self.is_training = is_training
        self.frame_fps = frame_fps
        self.annos, self.labels = [], []
        for anno in self._annos:
            video_uid = anno['video_uid']
            duration = self.metadata[video_uid]['duration']
            steps = anno['steps']
            for i in range(len(steps)):
                response = steps[i]['text'].capitalize() + '.'
                self.labels.append(steps[i]['text'].lower())
                start_time = ceil_time_by_fps(steps[i]['start'], frame_fps, min_time=0, max_time=duration)
                end_time = ceil_time_by_fps(steps[i]['end'], frame_fps, min_time=0, max_time=duration)
                start_frame = int(start_time * frame_fps)
                end_frame = int(end_time * frame_fps) + 1
                conversation = [
                    COINStep.user_message,
                    {"role": "stream", 'num_frames': end_frame - start_frame, 'learn': True},
                    {"role": "assistant", "content": response, 'learn': True}
                ]
                self.annos.append({
                    'conversation': conversation,
                    'load_ranges': {self.metadata[video_uid]['path']: range(start_frame, end_frame)}
                })
        self.labels = np.array(self.labels) # for fast indexing
        self.categories = self.step_categories

def build_coin_step_train(**kwargs):
    return COINStep(split='train', **kwargs)

def build_coin_step_test(**kwargs):
    return COINStep(split='test', **kwargs)

class COINNext(COINBenchmark):
    user_message = {
        "role": "user",
        "content": 'What is the next action for the video? Format your answer concisely. No extra text output.'
    }
    def __init__(self, *, split: str, frame_fps: int, is_training: bool, **kwargs):
        super().__init__(split=split, frame_fps=frame_fps, is_training=is_training, **kwargs)
        self.is_training = is_training
        self.frame_fps = frame_fps
        self.annos, self.labels = [], []
        for anno in self._annos:
            video_uid = anno['video_uid']
            duration = self.metadata[video_uid]['duration']
            steps = anno['steps']
            for i in range(len(steps) - 1):
                response = steps[i+1]['text'].capitalize() + '.'
                self.labels.append(steps[i+1]['text'].lower())
                start_time = ceil_time_by_fps(steps[i]['start'], frame_fps, min_time=0, max_time=duration)
                end_time = ceil_time_by_fps(steps[i]['end'], frame_fps, min_time=0, max_time=duration)
                start_frame = int(start_time * frame_fps)
                end_frame = int(end_time * frame_fps) + 1
                conversation = [
                    COINNext.user_message,
                    {"role": "stream", 'num_frames': end_frame - start_frame, 'learn': True},
                    {"role": "assistant", "content": response, 'learn': True}
                ]
                self.annos.append({
                    'conversation': conversation,
                    'load_ranges': {self.metadata[video_uid]['path']: range(start_frame, end_frame)}
                })
        self.labels = np.array(self.labels) # for fast indexing
        self.categories = self.step_categories

def build_coin_next_train(**kwargs):
    return COINNext(split='train', **kwargs)

def build_coin_next_test(**kwargs):
    return COINNext(split='test', **kwargs)

class COINTask(COINBenchmark):
    user_message = {
        "role": "user",
        "content": 'What is the overall activity in the video? Format your answer concisely. No extra text output.'
    }
    def __init__(self, *, split: str, frame_fps: int, is_training: bool, **kwargs):
        super().__init__(split=split, frame_fps=frame_fps, is_training=is_training, **kwargs)
        self.is_training = is_training
        self.frame_fps = frame_fps
        self.annos, self.labels = [], []
        for anno in self._annos:
            video_uid = anno['video_uid']
            duration = self.metadata[video_uid]['duration']
            response = anno['task'].capitalize() + '.'
            self.labels.append(anno['task'].lower())
            start_time = ceil_time_by_fps(anno['start'], frame_fps, min_time=0, max_time=duration)
            end_time = ceil_time_by_fps(anno['end'], frame_fps, min_time=0, max_time=duration)
            start_frame = int(start_time * frame_fps)
            end_frame = int(end_time * frame_fps) + 1
            conversation = [
                COINTask.user_message,
                {"role": "stream", 'num_frames': end_frame - start_frame, 'learn': True},
                {"role": "assistant", "content": response, 'learn': True}
            ]
            self.annos.append({
                'conversation': conversation,
                'load_ranges': {self.metadata[video_uid]['path']: range(start_frame, end_frame)}
            })
        self.labels = np.array(self.labels) # for fast indexing
        self.categories = self.task_categories

def build_coin_task_train(**kwargs):
    return COINTask(split='train', **kwargs)

def build_coin_task_test(**kwargs):
    return COINTask(split='test', **kwargs)

####################################################### dataloader ######################################################
def data_collator(batch: list[list], *, tokenizer: PreTrainedTokenizer, **kwargs):
    batch = list(zip(*batch))
    batch_text, batch_frames, batch_learn_ranges, batch_sample_idx, batch_evaluation_kwargs = batch
    # print(batch_text)
    batch = tokenizer(batch_text, return_offsets_mapping=True, add_special_tokens=False, return_tensors="pt", padding=True)
    batch_labels = torch.full_like(batch.input_ids, LabelSmoother.ignore_index, dtype=torch.long)
    for text, labels, input_ids, offset_mapping, learn_range in zip(
        batch_text, batch_labels, batch.input_ids, batch.offset_mapping, batch_learn_ranges
    ):
        for learn_r in learn_range:
            start = torch.nonzero(offset_mapping[:,0] == learn_r.start).item()
            if offset_mapping[:,0][-1] >= learn_r.stop:
                stop = torch.nonzero(offset_mapping[:,0] == learn_r.stop).item()
            else: # the last eos token
                stop = len(input_ids)
            labels[start-1:stop-1] = input_ids[start:stop]
            # NOTE: input_ids may out of boundary of len(tokenizer) - 1. (1 is the added vision placeholder)
            # this is because some frames has v_placeholder_id target. so replace it with eos token.
            labels[labels >= len(tokenizer) - 1] = tokenizer.eos_token_id
    batch['labels'] = batch_labels
    batch.pop('offset_mapping')
    batch['frames'] = torch.cat(batch_frames)
    batch['sample_idxs'] = torch.tensor(batch_sample_idx)
    if batch_evaluation_kwargs[0]:
        batch['evaluation_kwargs'] = batch_evaluation_kwargs[0] # evaluation only supports bs = 1, so its okay
    return batch

def get_data_collator(**kwargs):
    return partial(data_collator, **kwargs)

def get_compute_metrics_dict( dataset_dict: dict,  **kwargs):
    if not dataset_dict:
        return None
    # add eval_ since transformers default metrics prefix is eval
    return {k: partial(v.compute_metrics, **kwargs) for k, v in dataset_dict.items()}

def _build_list_datasets(datasets: list, is_training: bool, **kwargs):
    # tokenizer will be changed (e.g., add tokens) during this process
    datasets = [
        globals()[f"build_{dataset}"](
            is_training=is_training,
            **kwargs
        ) for dataset in datasets
    ]
    return datasets

def build_concat_train_dataset(train_datasets: list, is_training=True, **kwargs):
    if train_datasets is None or len(train_datasets) == 0:
        return None
    return ConcatDataset(_build_list_datasets(datasets=train_datasets, is_training=is_training, **kwargs))

def build_eval_dataset_dict(eval_datasets: list, is_training=False, **kwargs):
    if eval_datasets is None or len(eval_datasets) == 0:
        return None
    list_datasets = _build_list_datasets(datasets=eval_datasets, is_training=is_training, **kwargs)
    return {name:dataset for name, dataset in zip(eval_datasets, list_datasets)}

####################################################### trainer ######################################################
class TrainerWithGenToEval(Trainer):
    def prediction_step(
        self,
        model: torch.nn.Module,
        inputs: dict,
        prediction_loss_only: bool,
        ignore_keys: list[str] = None,
    ):
        with torch.no_grad(), self.compute_loss_context_manager():
            inputs = self._prepare_inputs(inputs)
            if prediction_loss_only:
                loss = self.compute_loss(model, inputs, return_outputs=False)
                return (loss, None, None)
            sample_idxs = inputs.pop('sample_idxs')
            evaluation_kwargs = inputs.pop('evaluation_kwargs')
            evaluator = evaluation_kwargs.pop('evaluator')
            output_ids = getattr(model, evaluator)(**inputs, **evaluation_kwargs, pad_token_id=self.tokenizer.pad_token_id, eos_token_id=self.tokenizer.eos_token_id)
            return (None, output_ids.reshape(1, -1), sample_idxs)

def train():
    args = parse_args()
    #args.use_cpu = True
    model, tokenizer = build_model_and_tokenizer(is_training=True, **asdict(args))
    train_dataset = build_concat_train_dataset(tokenizer=tokenizer, **asdict(args))
    eval_dataset_dict = build_eval_dataset_dict(tokenizer=tokenizer, **asdict(args))
    data_collator = get_data_collator(tokenizer=tokenizer, **asdict(args))
    compute_metrics_dict = get_compute_metrics_dict(dataset_dict=eval_dataset_dict, tokenizer=tokenizer, **asdict(args))

    args.gradient_checkpointing_kwargs = {'use_reentrant': False}
    trainer = TrainerWithGenToEval(
        model=model, tokenizer=tokenizer,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset_dict,
        data_collator=data_collator,
        compute_metrics=compute_metrics_dict,
    )
    trainer.train()
    trainer.save_model()

    if eval_dataset_dict is not None:
        metrics = {}
        for eval_dataset_name, eval_dataset in eval_dataset_dict.items():
            trainer.compute_metrics = compute_metrics_dict[eval_dataset_name]
            metrics.update(
                trainer.evaluate(
                    eval_dataset=eval_dataset,
                    metric_key_prefix=f"eval_{eval_dataset_name}",
                )
            )
        print(metrics)

def evaluate():
    args = parse_args()
    model, tokenizer = build_model_and_tokenizer(is_training=False, **asdict(args))
    eval_dataset_dict = build_eval_dataset_dict(tokenizer=tokenizer, model_config=model.config, **asdict(args))
    data_collator = get_data_collator(tokenizer=tokenizer, model_config=model.config, **asdict(args))
    compute_metrics_dict = get_compute_metrics_dict(dataset_dict=eval_dataset_dict, tokenizer=tokenizer, **asdict(args))

    trainer = TrainerWithGenToEval(
        model=model, tokenizer=tokenizer,
        args=args,
        eval_dataset=eval_dataset_dict,
        data_collator=data_collator,
        compute_metrics=compute_metrics_dict,
    )

    metrics = {}
    for eval_dataset_name, eval_dataset in eval_dataset_dict.items():
        trainer.compute_metrics = compute_metrics_dict[eval_dataset_name]
        dataset_metrics = trainer.evaluate(
            eval_dataset=eval_dataset,
            metric_key_prefix=f"eval_{eval_dataset_name}",
        )
        metrics.update(dataset_metrics)
    print(metrics)

    final_results = {'Checkpoints': args.resume_from_checkpoint}
    tasks = ['step_test_accuracy', 'next_test_accuracy', 'task_test_accuracy']
    for t in tasks:
        task_name = ' ' + t.replace('_test_accuracy', '')
        for metric in metrics.keys():
            if t in metric:
                final_results[task_name] = metrics[metric]
                break
        if task_name not in final_results:
            final_results[task_name] = -1

    import pandas as pd
    df = pd.DataFrame([final_results])

    head = True
    if os.path.exists('coin_results.csv'):
        head = False
    df.to_csv('coin_results.csv', mode='a', index=False, header=head, float_format='%.1f')


if __name__ == "__main__":
    argument = sys.argv[1:]
    if '--resume_from_checkpoint' in argument:
        evaluate()
    else:
        train()
