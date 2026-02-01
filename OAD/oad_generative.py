#
from transformers import AutoTokenizer, HfArgumentParser, PreTrainedTokenizer, AutoModelForCausalLM, TrainerCallback, \
    Cache
from transformers import Trainer, TrainingArguments, LlamaConfig, EvalPrediction
from transformers.trainer_pt_utils import LabelSmoother
from dataclasses import asdict
from functools import partial
import torch, collections
from torch import nn
from transformers.utils import logging
from transformers.activations import GELUActivation
from peft import LoraConfig, get_peft_model, PeftModel
import numpy as np
from matplotlib import pyplot as plt
import os, json, tqdm, time
from torch.utils.data import ConcatDataset
import Levenshtein
from dataclasses import dataclass, field
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
import random
import sys
sys.path.append('..')

os.environ["TOKENIZERS_PARALLELISM"] = "false"
logger = logging.get_logger(__name__)

from src.metrics import thumos_results_new, crosstask_results_new, ek100_results_new, ego4dgoal_results_new
from src.utils import DictWithTo, get_labels_start_end_time, uniform_sampler
from src.trainer import TrainerWithTensorBoard, LengthGroupedSampler, VideoGroupedSampler
from src.cls_llm import LlamaForCausalLM


####################################### Arguments
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
    llm_pretrained: str = 'meta-llama/Llama-3.2-1B-Instruct'  # 'meta-llama/Meta-Llama-3-8B-Instruct'#
    vision_pretrained: str = 'google/siglip-large-patch16-384'
    lora_modules: str = "model.*(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)|lm_head$"
    lora_r: int = 128
    lora_alpha: int = 256
    finetune_modules: list[str] = field(default_factory=lambda: ['connector'])
    frame_fps: int = 2  # for training. inference can be 10
    frame_token_cls: bool = None
    frame_token_pooled: list[int] = None
    frame_resolution: int = 384
    frame_token_interval: str = None
    frame_token_interval_threshold: float = 0.0
    augmentation: bool = False
    attn_implementation: str = 'flash_attention_2'
    output_dir: str = 'outputs/debug'
    text2visual_connector = '\n' #'\nStream:'
    visual2text_connector = '\nAssistant: '


@dataclass
class LiveOneTHUMOSTrainingArguments(LiveTrainingArguments):
    live_version: str = 'last'
    frame_token_cls: bool = True
    frame_num_tokens: int = 1
    frame_token_interval: str = ''
    silence_token: str = ' background'
    embed_mark: str = 'kinetics'
    max_num_frames: int = 7200  # 1h, 2fps, 7200 frames
    train_datasets: list[str] = field(default_factory=lambda: ['thumos_train'])
    eval_datasets: list[str] = field(default_factory=lambda: ['thumos_test'])
    stream_loss_weight: float = 1.0
    llm_pretrained: str = 'meta-llama/Llama-3.2-1B-Instruct'  # 'meta-llama/Meta-Llama-3-8B-Instruct'#
    vision_pretrained: str = 'resnet50'
    flow_pretrained: str = 'bninception'
    lora_modules: str = "model.*(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)|lm_head$"
    lora_r: int = 128
    lora_alpha: int = 256
    lora_dropout: float = 0.05
    finetune_modules: list[str] = field(default_factory=lambda: ['connector'])
    frame_fps: int = 4  # for training. inference can be 10
    augmentation: bool = False
    attn_implementation: str = 'sdpa'  # 'flash_attention_2'
    criterion: str = 'CE'
    short_len: int = 20  # 8 * 4
    short_sr: int = 1
    long_len: int = 128  # 128 * 4
    long_sr: int = 1  # 4
    visual_dim: int = 1536  # 2048
    imbalance_ratio: float = -1.0
    group_by_length: bool = False
    group_by_video_num: int = 0
    max_response_length: int = 8
    stride: int = 32
    test: bool = False
    test_set: str = 'train'
    test_no_cache: bool = False
    silence_threshold: float = 0.0
    head_dropout: float = 0.0
    post_process: bool = False
    post_threshold: float = 0.5


####################################### configure
class LiveLlamaConfig(LlamaConfig):
    def __init__(self, *, vision_pretrained: str = None,
                 frame_resolution: int = None, frame_token_cls: bool = None, frame_token_pooled: list[int] = None,
                 frame_num_tokens: int = None,
                 v_placeholder: str = '<v>', frame_token_interval: str = None, v_placeholder_id: int = None,
                 frame_token_interval_id: int = None, silence_token: str = None,
                 criterion: str = 'CE', stream_loss_weight: float = 1.0, vision_hidden_size=1024, **kwargs
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
        self.silence_token = silence_token
        self.v_placeholder_id = v_placeholder_id
        self.frame_token_interval_id = frame_token_interval_id
        self.criterion = criterion


##################################### data collator
def data_collator(batch: list[list], *, tokenizer: PreTrainedTokenizer, **kwargs):
    batch = list(zip(*batch))
    batch_text, batch_frames, batch_learn_ranges, batch_sample_idx, batch_evaluation_kwargs = batch
    batch = tokenizer(batch_text, return_offsets_mapping=True, add_special_tokens=False, return_tensors="pt",
                      padding=True)
    # for j in batch['input_ids'][0, :]:
    #     print(j, repr(tokenizer.decode(j)))
    batch_labels = torch.full_like(batch.input_ids, LabelSmoother.ignore_index, dtype=torch.long)
    for text, labels, input_ids, offset_mapping, learn_range in zip(
            batch_text, batch_labels, batch.input_ids, batch.offset_mapping, batch_learn_ranges
    ):
        for learn_r in learn_range:
            assert torch.max(offset_mapping[:, 0] == learn_r.start).int() > 0
            start = torch.nonzero(offset_mapping[:, 0] == learn_r.start).item()
            if offset_mapping[:, 0][-1] >= learn_r.stop:
                stop = torch.nonzero(offset_mapping[:, 0] == learn_r.stop).item()
            else:  # the last eos token
                stop = len(input_ids)
            labels[start - 1:stop - 1] = input_ids[start:stop]
            # NOTE: input_ids may out of boundary of len(tokenizer) - 1. (1 is the added vision placeholder)
            # this is because some frames has v_placeholder_id target. so replace it with eos token.
            labels[labels >= len(tokenizer) - 1] = tokenizer.eos_token_id

    batch['labels'] = batch_labels
    batch.pop('offset_mapping')
    num_frames = [len(frames) for frames in batch_frames]
    #print(num_frames)
    max_num_frames = max(num_frames)
    to_stack_frames = []
    for frames in batch_frames:
        if len(frames) < max_num_frames:
            frames = torch.nn.functional.pad(frames, (0, 0, 0, max_num_frames-len(frames)), "constant", 0)
        to_stack_frames.append(frames)
    batch['rgb_frames'] = torch.stack(to_stack_frames)
    batch['sample_idxs'] = torch.tensor(batch_sample_idx)
    batch['num_frames'] = torch.tensor(num_frames)
    if batch_evaluation_kwargs[0]:
        batch['evaluation_kwargs'] = batch_evaluation_kwargs[0]  # evaluation only supports bs = 1, so its okay
    return batch


def get_data_collator(**kwargs):
    return partial(data_collator, **kwargs)


class DataShuffleCallback(TrainerCallback):
    """
    Trigger re-computing subset for dataset Examples-proportional mixing, see `dataset::ProportionMixingDataset`

    A hack that modifies the train dataset, pointed by Trainer's dataloader
    """

    def __init__(self, ):
        pass

    def on_epoch_begin(self, args: TrainingArguments, state, control, train_dataloader, **kwargs):

        # if state.epoch > 0:
        #     time.sleep(60)
        #     train_dataloader.dataset.shuffle()
        #     if isinstance(train_dataloader.batch_sampler.sampler, LengthGroupedSampler) or \
        #             isinstance(train_dataloader.batch_sampler.sampler, VideoGroupedSampler):
        #         train_dataloader.batch_sampler.sampler.update_length(train_dataloader.dataset)
        if state.epoch > 0:
            if args.local_rank == 0:
                train_dataloader.dataset.shuffle() #.batch_sampler
                if isinstance(train_dataloader.sampler, LengthGroupedSampler) or \
                        isinstance(train_dataloader.batch_sampler.sampler, VideoGroupedSampler):
                    train_dataloader.sampler.update_length(train_dataloader.dataset)
            else:
                time.sleep(60)



##################################### dataset
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
        pass

    def compute_metrics(self, eval_predictions: EvalPrediction, tokenizer: PreTrainedTokenizer, **kwargs):
        pass


class THUMOS:
    root = f'../datasets/thumos14'
    video_root = os.path.join(root, '')
    anno_root = os.path.join(root, 'target_perframe')
    with open(f'{root}/data_info.json', 'r') as f:
        data_info = json.load(f)['THUMOS']

    def __init__(self, split: str, vision_pretrained: str, flow_pretrained: str, embed_mark: str, **kwargs):
        super().__init__(**kwargs)
        self.embed_dir = f"{self.video_root}/rgb_{embed_mark}_{vision_pretrained}"
        self.flow_dir = f"{self.video_root}/flow_{embed_mark}_{flow_pretrained}"
        self.frame_fps = self.data_info['fps']
        self.metadata = self.get_metadata()
        self.annodata = self.get_anno()
        assert split in ['train', 'test']
        self.session_list = self.data_info['train_session_set']  # ['video_validation_0000190'] #
        if split == 'test':
            self.session_list = self.data_info['test_session_set']  # ['video_test_0000004']  #
        self._annos = [{
            'video_uid': video_uid,
            'steps': [dict(
                start=step['start'],
                end=step['end'],
                text=THUMOS._clean_step(step['text']),
            ) for step in anno['steps']],
        } for video_uid, anno in self.annodata.items() if
            (video_uid in self.session_list) and (video_uid in self.metadata)]
        self.step_categories = list(
            set([THUMOS._clean_step(step['text']) for steps in self._annos for step in steps['steps']]))
        self.annos: list[dict]

    def get_anno(self, ):
        annodata_path = f'{self.root}/annodata.json'
        if os.path.exists(annodata_path):
            print(f'load {annodata_path}...')
            annodata = json.load(open(annodata_path))
        else:
            annodata = {}
            CLASS_NAMES = self.data_info['class_names']
            for file in os.listdir(self.anno_root):
                path = os.path.join(self.anno_root, file)
                key = os.path.splitext(os.path.basename(path))[0].split('.npy')[0]
                labels = np.load(path)
                steps = []
                for i in range(1, len(CLASS_NAMES) - 1):
                    l, s, e = get_labels_start_end_time(labels[:, i], bg_class=[0])
                    for j in range(len(l)):
                        steps.append({'start': s[j] / self.data_info['fps'], 'end': e[j] / self.data_info['fps'],
                                      'text': CLASS_NAMES[i]})

                steps = sorted(steps, key=lambda x: x['start'])
                annodata[key] = {'video_uid': key, 'steps': steps}
            json.dump(annodata, open(annodata_path, 'w'), indent=4)
        return annodata

    def get_metadata(self, ):
        metadata_path = f'{self.root}/metadata.json'
        if os.path.exists(metadata_path):
            print(f'load {metadata_path}...')
            metadata = json.load(open(metadata_path))
        else:
            metadata = {}
            for file in tqdm.tqdm(os.listdir(self.embed_dir), desc=f'prepare {metadata_path}...'):
                path = os.path.join(self.embed_dir, file)
                flow_path = os.path.join(self.flow_dir, file)
                duration = (len(np.load(path)) - 1) / self.frame_fps
                key = os.path.splitext(os.path.basename(path))[0]
                key = key.split('.npy')[0]
                metadata[key] = {'duration': duration, 'rgb_path': path, 'flow_path': flow_path}
            json.dump(metadata, open(metadata_path, 'w'), indent=4)
        return metadata

    # PutOnHair -> put on hair
    @staticmethod
    def _clean_step(text):
        result = ''
        for char in text:
            if char.isupper():
                result += ' ' + char.lower()
            else:
                result += char
        return result.strip()

    def get_framelabel(self, ):
        classnames = self.data_info['class_names']
        frame_anno = {}
        for file in os.listdir(self.anno_root):
            path = os.path.join(self.anno_root, file)
            key = os.path.splitext(os.path.basename(path))[0].split('.npy')[0]
            labels = np.load(path)
            gt = np.argmax(labels, axis=-1)
            frame_anno[key] = [THUMOS._clean_step(classnames[j]) for j in gt]
        return frame_anno

    def __len__(self):
        return len(self.annos)


class THUMOWind_ls(THUMOS, StreamMixIn):
    evaluation_kwargs = DictWithTo(evaluator='generate_after_embed', max_new_tokens=512, do_sample=False,
                                   use_cache=True, temperature=1.0, top_p=1.0)
    sys_message = { "role": "system",  "content": 'What is the action in the last frame?'}

    def __init__(self, *, split: str, is_training: bool, short_len: int, short_sr: int, long_len: int, long_sr: int,
                 stride: int, imbalance_ratio: float, **kwargs):
        super().__init__(split=split, is_training=is_training, **kwargs)
        self.is_training = is_training
        self.framelabel = self.get_framelabel()
        self.short_length = short_len
        self.long_length = long_len
        self.short_sample_rate = short_sr
        self.long_sample_rate = long_sr
        self.stride = stride
        self.imbalance_ratio = imbalance_ratio
        self._init_dataset()
        self.total_num = len(self.annos)
        self.categories = self.step_categories

    def _init_dataset(self):
        self.count = 0
        self.annos, self.labels = [], []
        self.anno_dict, index_action = {}, []
        for anno in self._annos:
            video_uid, steps = anno['video_uid'], self.framelabel[anno['video_uid']]
            seed = np.random.randint(self.short_length) if self.is_training else 0
            for work_end in range(seed, len(steps), self.stride):
                work_end = max(1, work_end)
                work_start = max(0, work_end - self.short_length)

                # decide the short-term
                sub_steps = steps[work_start:work_end:self.short_sample_rate]
                work_indices = np.arange(work_start, work_end).clip(0)
                work_indices = work_indices[::self.short_sample_rate]

                # decide the long-term context
                long_start, long_end = work_start - self.long_length, work_start - 1
                long_indices = uniform_sampler(long_start, long_end, self.long_sample_rate).clip(0)

                final_indices = np.concatenate((long_indices, work_indices))
                response = sub_steps[-1].lower()
                if response == 'background':
                    conversation = [THUMOWind_ls.sys_message,
                                    {"role": "stream", 'num_frames': len(final_indices), 'learn': False},
                                    {"role": "silence", "content": self.tokenizer.silence_token}]
                else:
                    conversation = [THUMOWind_ls.sys_message,
                                    {"role": "stream", 'num_frames': len(final_indices), 'learn': False},
                                    {"role": "assistant", "content": response, 'learn': True}]

                #############
                text = self.tokenizer.apply_chat_template(conversation, tokenize=False,  add_generation_prompt=False)
                input_ids = self.tokenizer(text, return_offsets_mapping=True, add_special_tokens=False, return_tensors="pt", padding=False)
                length = len(input_ids.input_ids[0])
                self.annos.append({
                    'conversation': conversation,
                    'load_ranges': ['../' +  self.metadata[video_uid]['rgb_path'], final_indices],
                    'input_length': length,
                    'text': text,
                    'video': video_uid,
                    'index': work_indices[-1]
                })
                ###############
                self.labels.append(response)
                if response not in self.anno_dict:
                    self.anno_dict[response] = 0
                self.anno_dict[response] += 1
        frequency = sorted([self.anno_dict[k] for k in self.anno_dict])[::-1]
        assert frequency[0] == self.anno_dict['background']
        print("Intial imbalance ratio: background/most foreground = {}, background/least foreground = {}".format(
            frequency[0] / frequency[1], frequency[0] / frequency[-1]))
        if self.imbalance_ratio > 0:
            # resample for background
            index = np.where(np.array(self.labels) == 'background')[0]
            # TODO: random sampling or uniform sampling
            sampled_list = random.sample(index.tolist(), int(self.imbalance_ratio * frequency[1]))
            orign_list = np.where(np.array(self.labels) != 'background')[0]
            sampled_index = sorted(orign_list.tolist() + list(sampled_list))
            self.annos = [self.annos[k] for k in sampled_index]
            self.labels = [self.labels[k] for k in sampled_index]
            assert np.sum(np.array(self.labels) == 'background') == len(sampled_list)
            print(
                "After resampling, imbalance ratio: background/most foreground = {}, background/least foreground = {}".format(
                    len(sampled_list) / frequency[1], len(sampled_list) / frequency[-1]))

    def trunc_pad_anno(self, ):
        if len(self.annos) > self.total_num:
            self.annos = self.annos[:self.total_num]
            self.labels = self.labels[:self.total_num]
        if len(self.annos) < self.total_num:
            diff = self.total_num - len(self.annos)
            self.annos = self.annos + self.annos[:diff]
            self.labels = self.labels + self.labels[:diff]

    def __getitem__(self, index):
        self.count += 1
        anno = self.annos[index]
        conversation, load_ranges = anno['conversation'], anno['load_ranges']
        rgb, ranger = load_ranges
        rgb_frames = torch.cat([torch.as_tensor(np.load(rgb, mmap_mode='r').astype(np.float32))[ranger]])
        # conversation = [{"role": "system", "content": self.system_prompt}] + conversation
        text = self.tokenizer.apply_chat_template(conversation, tokenize=False,
                                                  add_generation_prompt=not self.is_training)
        # print(text)
        # 3. learn ranges
        learn_ranges = self.tokenizer.get_learn_ranges(conversation) if self.is_training else []
        # for i in range(len(learn_ranges)):
        #     print(i, repr(text[learn_ranges[i].start: learn_ranges[i].stop]))
        return text, rgb_frames, learn_ranges, index, self.evaluation_kwargs

    def shuffle(self):
        self._init_dataset()
        self.trunc_pad_anno()
        print('[shuffle the data]')


def build_thumos_wind_ls(split='train', **kwargs):
    return THUMOWind_ls(split=split, **kwargs)


class CrossTask:
    root = '../datasets/CrossTask'
    video_root = os.path.join(root, '')
    anno_root = os.path.join(root, 'target_perframe')
    DISCARD = ['MG8Frk8xCnM', 'bo355kAfADM', 'iGi3Wsx9MZU', '7LqVD40qcq4', 'zspSsRGjLTw', 'fkJVun3NveM',
               'NYpQ6GJCBVI', 'i4xTz_OwlSQ', 'igKDiI3s5As', 'rq-nNfUrgNc', 'cugUlvHB430', 'vB16Dbjz8oo',
               'rEOuV_NMcu8', 'JDVujgjtkPI', 'Z3ylY7Xv3XE', '4YoKs6cmjps', 'Gy3MURE_2Jw', '9rPmr8dPX70',
               'PZh3Jkel3Ds', 'Q_g8VZ-m720', 'QNdlI7Fpc6M', 'E5HA-ZsDdMg']

    with open(f'{root}/data_info.json', 'r') as f:
        data_info = json.load(f)['CrossTask']

    def __init__(self, split: str, vision_pretrained: str, flow_pretrained: str, embed_mark: str, **kwargs):
        super().__init__(**kwargs)
        # self.embed_dir = f"{self.video_root}/{embed_mark}_{vision_pretrained}"
        self.embed_dir = f"{self.video_root}/vitg14_rgb_fps1_h224_w224"
        self.frame_fps = self.data_info['fps']
        self.metadata = self.get_metadata()
        self.annodata = self.get_anno()
        assert split in ['train', 'test']
        self.session_list = self.data_info['train_session_set']  # ['video_validation_0000190'] #
        if split == 'test':
            self.session_list = self.data_info['test_session_set']  # ['video_test_0000004']  #
        self._annos = [{
            'video_uid': video_uid,
            'steps': [dict(
                start=step['start'],
                end=step['end'],
                text=THUMOS._clean_step(step['text']),
            ) for step in anno['steps']],
        } for video_uid, anno in self.annodata.items() if
            (video_uid in self.session_list) and (video_uid in self.metadata)]
        self.step_categories = list(
            set([CrossTask._clean_step(step['text']) for steps in self._annos for step in steps['steps']]))
        self.annos: list[dict]

    def get_anno(self, ):
        annodata_path = f'{self.root}/annodata.json'
        if os.path.exists(annodata_path):
            print(f'load {annodata_path}...')
            annodata = json.load(open(annodata_path))
        else:
            annodata = {}
            CLASS_NAMES = self.data_info['class_names']
            for file in os.listdir(self.anno_root):
                path = os.path.join(self.anno_root, file)
                key = os.path.splitext(os.path.basename(path))[0].split('.npy')[0]
                labels = np.load(path)
                steps = []
                for i in range(1, len(CLASS_NAMES)):
                    l, s, e = get_labels_start_end_time(labels[:, i], bg_class=[0])
                    for j in range(len(l)):
                        steps.append({'start': s[j] / self.data_info['fps'], 'end': e[j] / self.data_info['fps'],
                                      'text': CLASS_NAMES[i]})

                steps = sorted(steps, key=lambda x: x['start'])
                annodata[key] = {'video_uid': key, 'steps': steps}
            json.dump(annodata, open(annodata_path, 'w'), indent=4)
        return annodata

    def get_metadata(self, ):
        metadata_path = f'{self.root}/metadata.json'
        if os.path.exists(metadata_path):
            print(f'load {metadata_path}...')
            metadata = json.load(open(metadata_path))
        else:
            metadata = {}
            for file in tqdm.tqdm(os.listdir(self.embed_dir), desc=f'prepare {metadata_path}...'):
                path = os.path.join(self.embed_dir, file)
                duration = (len(np.load(path)) - 1) / self.frame_fps
                key = os.path.splitext(os.path.basename(path))[0]
                key = key.split('.npy')[0]
                metadata[key] = {'duration': duration, 'rgb_path': path}
            json.dump(metadata, open(metadata_path, 'w'), indent=4)
        return metadata

    # PutOnHair -> put on hair
    @staticmethod
    def _clean_step(text):
        if text == 'BackGround':
            return 'background'
        result = ''
        for char in text:
            if char.isupper():
                result += ' ' + char.lower()
            else:
                result += char
        return result.strip()

    def get_framelabel(self, ):
        classnames = self.data_info['class_names']
        frame_anno = {}
        for file in os.listdir(self.anno_root):
            path = os.path.join(self.anno_root, file)
            key = os.path.splitext(os.path.basename(path))[0].split('.npy')[0]
            labels = np.load(path)
            gt = np.argmax(labels, axis=-1)
            frame_anno[key] = [CrossTask._clean_step(classnames[j]) for j in gt]
        return frame_anno

    def __len__(self):
        return len(self.annos)

class CrossTaskWind_ls(CrossTask, StreamMixIn):
    evaluation_kwargs = DictWithTo(evaluator='generate_after_embed', max_new_tokens=512, do_sample=False,
                                   use_cache=True, temperature=1.0, top_p=1.0)
    sys_message = { "role": "system",  "content": 'What is the action in the last frame?'}

    def __init__(self, *, split: str, is_training: bool, short_len: int, short_sr: int, long_len: int, long_sr: int,
                 stride: int, imbalance_ratio: float, **kwargs):
        super().__init__(split=split, is_training=is_training, **kwargs)
        self.is_training = is_training
        self.framelabel = self.get_framelabel()
        self.short_length = short_len
        self.long_length = long_len
        self.short_sample_rate = short_sr
        self.long_sample_rate = long_sr
        self.stride = stride
        self.imbalance_ratio = imbalance_ratio
        self._init_dataset()
        self.total_num = len(self.annos)
        self.categories = self.step_categories

    def _init_dataset(self):
        self.count = 0
        self.annos, self.labels = [], []
        self.anno_dict, index_action = {}, []
        for anno in self._annos:
            video_uid, steps = anno['video_uid'], self.framelabel[anno['video_uid']]
            seed = np.random.randint(self.short_length) if self.is_training else 0
            for work_end in range(seed, len(steps), self.stride):
                work_end = max(1, work_end)
                work_start = max(0, work_end - self.short_length)

                # decide the short-term
                sub_steps = steps[work_start:work_end:self.short_sample_rate]
                work_indices = np.arange(work_start, work_end).clip(0)
                work_indices = work_indices[::self.short_sample_rate]

                # decide the long-term context
                long_start, long_end = work_start - self.long_length, work_start - 1
                long_indices = uniform_sampler(long_start, long_end, self.long_sample_rate).clip(0)

                final_indices = np.concatenate((long_indices, work_indices))
                response = sub_steps[-1].lower()
                if response == 'background':
                    conversation = [CrossTaskWind_ls.sys_message,
                                    {"role": "stream", 'num_frames': len(final_indices), 'learn': False},
                                    {"role": "silence", "content": self.tokenizer.silence_token}]
                else:
                    conversation = [CrossTaskWind_ls.sys_message,
                                    {"role": "stream", 'num_frames': len(final_indices), 'learn': False},
                                    {"role": "assistant", "content": response, 'learn': True}]

                #############
                text = self.tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False)
                input_ids = self.tokenizer(text, return_offsets_mapping=True, add_special_tokens=False, return_tensors="pt",
                                      padding=False)
                length = len(input_ids.input_ids[0])
                self.annos.append({
                    'conversation': conversation,
                    'load_ranges': ['../' +  self.metadata[video_uid]['rgb_path'], final_indices],
                    'input_length': length,
                    'text': text,
                    'video': video_uid,
                    'index': work_indices[-1]
                })
                ###############
                self.labels.append(response)
                if response not in self.anno_dict:
                    self.anno_dict[response] = 0
                self.anno_dict[response] += 1

        frequency = sorted([self.anno_dict[k] for k in self.anno_dict])[::-1]
        # assert len(self.anno_dict) == len(self.step_categories) + 1 , "stride is too large that leads to missing categories."
        assert frequency[0] == self.anno_dict['background']
        print("Intial imbalance ratio: background/most foreground = {}, background/least foreground = {}".format(
            frequency[0] / frequency[1], frequency[0] / frequency[-1]))
        if self.imbalance_ratio > 0:
            # resample for background
            index = np.where(np.array(self.labels) == 'background')[0]
            # TODO: random sampling or uniform sampling
            sampled_list = random.sample(index.tolist(), int(self.imbalance_ratio * frequency[1]))
            orign_list = np.where(np.array(self.labels) != 'background')[0]
            sampled_index = sorted(orign_list.tolist() + list(sampled_list))
            self.annos = [self.annos[k] for k in sampled_index]
            self.labels = [self.labels[k] for k in sampled_index]
            assert np.sum(np.array(self.labels) == 'background') == len(sampled_list)
            print(
                "After resampling, imbalance ratio: background/most foreground = {}, background/least foreground = {}".format(
                    len(sampled_list) / frequency[1], len(sampled_list) / frequency[-1]))

    def trunc_pad_anno(self, ):
        if len(self.annos) > self.total_num:
            self.annos = self.annos[:self.total_num]
            self.labels = self.labels[:self.total_num]
        if len(self.annos) < self.total_num:
            diff = self.total_num - len(self.annos)
            self.annos = self.annos + self.annos[:diff]
            self.labels = self.labels + self.labels[:diff]

    def __getitem__(self, index):
        self.count += 1
        anno = self.annos[index]
        conversation, load_ranges = anno['conversation'], anno['load_ranges']
        rgb, ranger = load_ranges
        rgb_frames = torch.cat([torch.as_tensor(np.load(rgb, mmap_mode='r').astype(np.float32))[ranger]])
        #conversation = [{"role": "system", "content": self.system_prompt}] + conversation
        text = self.tokenizer.apply_chat_template(conversation, tokenize=False,
                                                  add_generation_prompt=not self.is_training)
        # print(text)
        # 3. learn ranges
        learn_ranges = self.tokenizer.get_learn_ranges(conversation) if self.is_training else []
        # for i in range(len(learn_ranges)):
        #     print(i, repr(text[learn_ranges[i].start: learn_ranges[i].stop]))

        return text, rgb_frames, learn_ranges, index, self.evaluation_kwargs

    def shuffle(self):
        self._init_dataset()
        self.trunc_pad_anno()
        print('[shuffle the data]')


def build_crosstask_wind_ls(split='train', **kwargs):
    return CrossTaskWind_ls(split=split, **kwargs)

class EK100:
    root = '../datasets/EK100'
    video_root = os.path.join(root, '')
    anno_root = os.path.join(root, 'target_perframe')

    with open(f'../datasets/CrossTask/data_info.json', 'r') as f:
        data_info = json.load(f)['EK100']

    def __init__(self, split: str, vision_pretrained: str, flow_pretrained: str, embed_mark: str, **kwargs):
        super().__init__(**kwargs)
        # self.embed_dir = f"{self.video_root}/{embed_mark}_{vision_pretrained}"
        self.embed_dir = f"{self.video_root}/rgb_kinetics_bninception"
        self.frame_fps = self.data_info['fps']
        self.metadata = self.get_metadata()
        self.annodata = self.get_anno()
        assert split in ['train', 'test']
        self.session_list = self.data_info['train_session_set'] # ['video_validation_0000190'] #
        if split == 'test':
            self.session_list = self.data_info['test_session_set']  # ['video_test_0000004']  #
        self._annos = [{
            'video_uid': video_uid,
            'steps': [dict(
                start=step['start'],
                end=step['end'],
                text=THUMOS._clean_step(step['text']),
            ) for step in anno['steps']],
        } for video_uid, anno in self.annodata.items() if
            (video_uid in self.session_list) and (video_uid in self.metadata)]
        self.step_categories = list(
            set([EK100._clean_step(step['text']) for steps in self._annos for step in steps['steps']]))
        self.annos: list[dict]

    def get_anno(self, ):
        annodata_path = f'{self.root}/annodata.json'
        if os.path.exists(annodata_path):
            print(f'load {annodata_path}...')
            annodata = json.load(open(annodata_path))
        else:
            annodata = {}
            CLASS_NAMES = self.data_info['class_names']
            for file in os.listdir(self.anno_root):
                path = os.path.join(self.anno_root, file)
                key = os.path.splitext(os.path.basename(path))[0].split('.npy')[0]
                labels = np.load(path)
                steps = []
                for i in range(1, len(CLASS_NAMES)):
                    l, s, e = get_labels_start_end_time(labels[:, i], bg_class=[0])
                    for j in range(len(l)):
                        steps.append({'start': s[j] / self.data_info['fps'], 'end': e[j] / self.data_info['fps'],
                                      'text': CLASS_NAMES[i]})

                steps = sorted(steps, key=lambda x: x['start'])
                annodata[key] = {'video_uid': key, 'steps': steps}
            json.dump(annodata, open(annodata_path, 'w'), indent=4)
        return annodata

    def get_metadata(self, ):
        metadata_path = f'{self.root}/metadata.json'
        if os.path.exists(metadata_path):
            print(f'load {metadata_path}...')
            metadata = json.load(open(metadata_path))
        else:
            metadata = {}
            for file in tqdm.tqdm(os.listdir(self.embed_dir), desc=f'prepare {metadata_path}...'):
                path = os.path.join(self.embed_dir, file)
                duration = (len(np.load(path)) - 1) / self.frame_fps
                key = os.path.splitext(os.path.basename(path))[0]
                key = key.split('.npy')[0]
                metadata[key] = {'duration': duration, 'rgb_path': path}
            json.dump(metadata, open(metadata_path, 'w'), indent=4)
        return metadata

    # PutOnHair -> put on hair
    @staticmethod
    def _clean_step(text):
        if text == 'Background':
            return 'background'
        result = ''
        for char in text:
            if char.isupper():
                result += ' ' + char.lower()
            else:
                result += char
        return result.strip()

    def get_framelabel(self, ):
        classnames = self.data_info['class_names']
        frame_anno = {}
        for file in os.listdir(self.anno_root):
            path = os.path.join(self.anno_root, file)
            key = os.path.splitext(os.path.basename(path))[0].split('.npy')[0]
            labels = np.load(path)
            gt = np.argmax(labels, axis=-1)
            frame_anno[key] = [CrossTask._clean_step(classnames[j]) for j in gt]
        return frame_anno

    def __len__(self):
        return len(self.annos)


class EK100Wind_ls(EK100, StreamMixIn):
    evaluation_kwargs = DictWithTo(evaluator='generate_after_embed', max_new_tokens=512, do_sample=False,
                                   use_cache=True, temperature=1.0, top_p=1.0)
    sys_message = {"role": "system", "content": 'What is the action in the last frame?'}

    def __init__(self, *, split: str, is_training: bool, short_len: int, short_sr: int, long_len: int, long_sr: int,
                 stride: int, imbalance_ratio: float, **kwargs):
        super().__init__(split=split, is_training=is_training, **kwargs)
        self.is_training = is_training
        self.framelabel = self.get_framelabel()
        self.short_length = short_len
        self.long_length = long_len
        self.short_sample_rate = short_sr
        self.long_sample_rate = long_sr
        self.stride = stride
        self.imbalance_ratio = imbalance_ratio
        self._init_dataset()
        self.total_num = len(self.annos)
        self.categories = self.step_categories

    def _init_dataset(self):
        self.count = 0
        self.annos, self.labels = [], []
        self.anno_dict, index_action = {}, []
        for anno in self._annos:
            video_uid, steps = anno['video_uid'], self.framelabel[anno['video_uid']]
            seed = np.random.randint(self.short_length) if self.is_training else 0
            for work_end in range(seed, len(steps), self.stride):
                work_end = max(1, work_end)
                work_start = max(0, work_end - self.short_length)

                # decide the short-term
                sub_steps = steps[work_start:work_end:self.short_sample_rate]
                work_indices = np.arange(work_start, work_end).clip(0)
                work_indices = work_indices[::self.short_sample_rate]

                # decide the long-term context
                long_start, long_end = work_start - self.long_length, work_start - 1
                long_indices = uniform_sampler(long_start, long_end, self.long_sample_rate).clip(0)

                final_indices = np.concatenate((long_indices, work_indices))
                response = sub_steps[-1].lower()
                if response == 'background':
                    conversation = [EK100Wind_ls.sys_message,
                                    {"role": "stream", 'num_frames': len(final_indices), 'learn': False},
                                    {"role": "silence", "content": self.tokenizer.silence_token}]
                else:
                    conversation = [EK100Wind_ls.sys_message,
                                    {"role": "stream", 'num_frames': len(final_indices), 'learn': False},
                                    {"role": "assistant", "content": response, 'learn': True}]

                #############
                text = self.tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False)
                input_ids = self.tokenizer(text, return_offsets_mapping=True, add_special_tokens=False,
                                           return_tensors="pt",
                                           padding=False)
                length = len(input_ids.input_ids[0])
                self.annos.append({
                    'conversation': conversation,
                    'load_ranges': [ self.metadata[video_uid]['rgb_path'], final_indices],
                    'input_length': length,
                    'text': text,
                    'video': video_uid,
                    'index': work_indices[-1]
                })
                ###############
                self.labels.append(response)
                if response not in self.anno_dict:
                    self.anno_dict[response] = 0
                self.anno_dict[response] += 1

        frequency = sorted([self.anno_dict[k] for k in self.anno_dict])[::-1]
        # assert len(self.anno_dict) == len(self.step_categories) + 1 , "stride is too large that leads to missing categories."
        assert frequency[0] == self.anno_dict['background']
        print("Intial imbalance ratio: background/most foreground = {}, background/least foreground = {}".format(
            frequency[0] / frequency[1], frequency[0] / frequency[-1]))
        if self.imbalance_ratio > 0:
            # resample for background
            index = np.where(np.array(self.labels) == 'background')[0]
            # TODO: random sampling or uniform sampling
            sampled_list = random.sample(index.tolist(), int(self.imbalance_ratio * frequency[1]))
            orign_list = np.where(np.array(self.labels) != 'background')[0]
            sampled_index = sorted(orign_list.tolist() + list(sampled_list))
            self.annos = [self.annos[k] for k in sampled_index]
            self.labels = [self.labels[k] for k in sampled_index]
            assert np.sum(np.array(self.labels) == 'background') == len(sampled_list)
            print(
                "After resampling, imbalance ratio: background/most foreground = {}, background/least foreground = {}".format(
                    len(sampled_list) / frequency[1], len(sampled_list) / frequency[-1]))

    def trunc_pad_anno(self, ):
        if len(self.annos) > self.total_num:
            self.annos = self.annos[:self.total_num]
            self.labels = self.labels[:self.total_num]
        if len(self.annos) < self.total_num:
            diff = self.total_num - len(self.annos)
            self.annos = self.annos + self.annos[:diff]
            self.labels = self.labels + self.labels[:diff]

    def __getitem__(self, index):
        self.count += 1
        anno = self.annos[index]
        conversation, load_ranges = anno['conversation'], anno['load_ranges']
        rgb, ranger = load_ranges
        rgb_frames = torch.cat([torch.as_tensor(np.load(rgb, mmap_mode='r').astype(np.float32))[ranger]])
        # conversation = [{"role": "system", "content": self.system_prompt}] + conversation
        text = self.tokenizer.apply_chat_template(conversation, tokenize=False,
                                                  add_generation_prompt=not self.is_training)
        # print(text)
        # 3. learn ranges
        learn_ranges = self.tokenizer.get_learn_ranges(conversation) if self.is_training else []
        # for i in range(len(learn_ranges)):
        #     print(i, repr(text[learn_ranges[i].start: learn_ranges[i].stop]))

        return text, rgb_frames, learn_ranges, index, self.evaluation_kwargs

    def shuffle(self):
        self._init_dataset()
        self.trunc_pad_anno()
        print('[shuffle the data]')


def build_ek100_wind_ls(split='train', **kwargs):
    return EK100Wind_ls(split=split, **kwargs)

##########
class Ego4DGoal:
    root = '../datasets/Ego4D-GoalStep'
    video_root = os.path.join(root, 'features')
    anno_root = os.path.join(root, 'target_perframe')

    with open(f'{root}/ego4d-goal_info.json', 'r') as f:
        data_info = json.load(f)

    def __init__(self, split: str, vision_pretrained: str, flow_pretrained: str, embed_mark: str, **kwargs):
        super().__init__(**kwargs)
        # self.embed_dir = f"{self.video_root}/{embed_mark}_{vision_pretrained}"
        self.embed_dir = f"{self.video_root}/dinov2-giant"
        self.frame_fps = self.data_info['fps']
        self.metadata = self.get_metadata()
        self.annodata = self.get_anno()
        assert split in ['train', 'test']
        self.session_list = self.data_info['train_session_set']  # ['video_validation_0000190'] #
        if split == 'test':
            self.session_list = self.data_info['val_session_set']  # ['video_test_0000004']  #
        self._annos = [{
            'video_uid': video_uid,
            'steps': [dict(
                start=step['start'],
                end=step['end'],
                text=Ego4DGoal._clean_step(step['text']),
            ) for step in anno['steps']],
        } for video_uid, anno in self.annodata.items() if
            (video_uid in self.session_list) and (video_uid in self.metadata)]
        self.step_categories = list(
            set([Ego4DGoal._clean_step(step['text']) for steps in self._annos for step in steps['steps']]))
        self.annos: list[dict]

    def get_anno(self, ):
        annodata_path = f'{self.root}/annodata.json'
        if os.path.exists(annodata_path):
            print(f'load {annodata_path}...')
            annodata = json.load(open(annodata_path))
        else:
            annodata = {}
            CLASS_NAMES = self.data_info['class_names']
            for file in os.listdir(self.anno_root):
                path = os.path.join(self.anno_root, file)
                key = os.path.splitext(os.path.basename(path))[0].split('.npy')[0]
                labels = np.load(path)
                steps = []
                for i in range(1, len(CLASS_NAMES)):
                    l, s, e = get_labels_start_end_time(labels[:, i], bg_class=[0])
                    for j in range(len(l)):
                        steps.append({'start': s[j] / self.data_info['fps'], 'end': e[j] / self.data_info['fps'],
                                      'text': CLASS_NAMES[i]})

                steps = sorted(steps, key=lambda x: x['start'])
                annodata[key] = {'video_uid': key, 'steps': steps}
            json.dump(annodata, open(annodata_path, 'w'), indent=4)
        return annodata

    def get_metadata(self, ):
        metadata_path = f'{self.root}/metadata.json'
        if os.path.exists(metadata_path):
            print(f'load {metadata_path}...')
            metadata = json.load(open(metadata_path))
        else:
            metadata = {}
            for file in tqdm.tqdm(os.listdir(self.embed_dir), desc=f'prepare {metadata_path}...'):
                path = os.path.join(self.embed_dir, file)
                duration = len(torch.load(path)) / self.frame_fps
                key = os.path.splitext(os.path.basename(path))[0]
                key = key.split('.pt')[0]
                metadata[key] = {'duration': duration, 'rgb_path': path}
            json.dump(metadata, open(metadata_path, 'w'), indent=4)
        return metadata

    # PutOnHair -> put on hair
    @staticmethod
    def _clean_step(text):
        if text == 'Background':
            return 'background'
        result = ''
        for char in text:
            if char.isupper():
                result += ' ' + char.lower()
            else:
                result += char
        return result.strip()

    def get_framelabel(self, ):
        classnames = self.data_info['class_names']
        frame_anno = {}
        num_bg, num_all = 0, 0
        for file in os.listdir(self.anno_root):
            path = os.path.join(self.anno_root, file)
            key = os.path.splitext(os.path.basename(path))[0].split('.npy')[0]
            labels = np.load(path)
            gt = np.argmax(labels, axis=-1)
            frame_anno[key] = [Ego4DGoal._clean_step(classnames[j]) for j in gt]
            num_bg += np.sum(gt == 0)
            num_all += len(gt)
        bg_ratio = num_bg/num_all
        return frame_anno

    def __len__(self):
        return len(self.annos)

class Ego4DGoalWind_ls(Ego4DGoal, StreamMixIn):
    evaluation_kwargs = DictWithTo(evaluator='generate_after_embed', max_new_tokens=512, do_sample=False,
                                   use_cache=True, temperature=1.0, top_p=1.0)
    sys_message = {"role": "system", "content": 'What is the action in the last frame?'}

    def __init__(self, *, split: str, is_training: bool, short_len: int, short_sr: int, long_len: int, long_sr: int,
                 stride: int, imbalance_ratio: float, **kwargs):
        super().__init__(split=split, is_training=is_training, **kwargs)
        self.is_training = is_training
        self.framelabel = self.get_framelabel()
        self.short_length = short_len
        self.long_length = long_len
        self.short_sample_rate = short_sr
        self.long_sample_rate = long_sr
        self.stride = stride
        self.imbalance_ratio = imbalance_ratio
        self._init_dataset()
        self.total_num = len(self.annos)
        self.categories = self.step_categories

    def _init_dataset(self):
        self.count = 0
        self.annos, self.labels = [], []
        self.anno_dict, index_action = {}, []
        for anno in self._annos:
            video_uid, steps = anno['video_uid'], self.framelabel[anno['video_uid']]
            seed = np.random.randint(self.short_length) if self.is_training else 0
            for work_end in range(seed, len(steps), self.stride):
                work_end = max(1, work_end)
                work_start = max(0, work_end - self.short_length)

                # decide the short-term
                sub_steps = steps[work_start:work_end:self.short_sample_rate]
                work_indices = np.arange(work_start, work_end).clip(0)
                work_indices = work_indices[::self.short_sample_rate]

                # decide the long-term context
                long_start, long_end = work_start - self.long_length, work_start - 1
                long_indices = uniform_sampler(long_start, long_end, self.long_sample_rate).clip(0)

                final_indices = np.concatenate((long_indices, work_indices))
                response = sub_steps[-1].lower()
                if response == 'background':
                    conversation = [EK100Wind_ls.sys_message,
                                    {"role": "stream", 'num_frames': len(final_indices), 'learn': False},
                                    {"role": "silence", "content": self.tokenizer.silence_token}]
                else:
                    conversation = [EK100Wind_ls.sys_message,
                                    {"role": "stream", 'num_frames': len(final_indices), 'learn': False},
                                    {"role": "assistant", "content": response, 'learn': True}]

                #############
                text = self.tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False)
                input_ids = self.tokenizer(text, return_offsets_mapping=True, add_special_tokens=False,
                                           return_tensors="pt",
                                           padding=False)
                length = len(input_ids.input_ids[0])
                self.annos.append({
                    'conversation': conversation,
                    'load_ranges': [self.metadata[video_uid]['rgb_path'], final_indices],
                    'input_length': length,
                    'text': text,
                    'video': video_uid,
                    'index': work_indices[-1]
                })
                ###############
                self.labels.append(response)
                if response not in self.anno_dict:
                    self.anno_dict[response] = 0
                self.anno_dict[response] += 1

        frequency = sorted([self.anno_dict[k] for k in self.anno_dict])[::-1]
        # assert len(self.anno_dict) == len(self.step_categories) + 1 , "stride is too large that leads to missing categories."
        assert frequency[0] == self.anno_dict['background']
        print("Intial imbalance ratio: background/most foreground = {}, background/least foreground = {}".format(
            frequency[0] / frequency[1], frequency[0] / frequency[-1]))
        if self.imbalance_ratio > 0:
            # resample for background
            index = np.where(np.array(self.labels) == 'background')[0]
            # TODO: random sampling or uniform sampling
            sampled_list = random.sample(index.tolist(), int(self.imbalance_ratio * frequency[1]))
            orign_list = np.where(np.array(self.labels) != 'background')[0]
            sampled_index = sorted(orign_list.tolist() + list(sampled_list))
            self.annos = [self.annos[k] for k in sampled_index]
            self.labels = [self.labels[k] for k in sampled_index]
            assert np.sum(np.array(self.labels) == 'background') == len(sampled_list)
            print(
                "After resampling, imbalance ratio: background/most foreground = {}, background/least foreground = {}".format(
                    len(sampled_list) / frequency[1], len(sampled_list) / frequency[-1]))

    def trunc_pad_anno(self, ):
        if len(self.annos) > self.total_num:
            self.annos = self.annos[:self.total_num]
            self.labels = self.labels[:self.total_num]
        if len(self.annos) < self.total_num:
            diff = self.total_num - len(self.annos)
            self.annos = self.annos + self.annos[:diff]
            self.labels = self.labels + self.labels[:diff]

    def __getitem__(self, index):
        self.count += 1
        anno = self.annos[index]
        conversation, load_ranges = anno['conversation'], anno['load_ranges']
        rgb, ranger = load_ranges
        rgb_frames = torch.cat([torch.load(rgb).float()[ranger]])
        #rgb_frames = torch.cat([torch.as_tensor(np.load(rgb, mmap_mode='r').astype(np.float32))[ranger]])
        # conversation = [{"role": "system", "content": self.system_prompt}] + conversation
        text = self.tokenizer.apply_chat_template(conversation, tokenize=False,
                                                  add_generation_prompt=not self.is_training)
        # print(text)
        # 3. learn ranges
        learn_ranges = self.tokenizer.get_learn_ranges(conversation) if self.is_training else []
        # for i in range(len(learn_ranges)):
        #     print(i, repr(text[learn_ranges[i].start: learn_ranges[i].stop]))

        return text, rgb_frames, learn_ranges, index, self.evaluation_kwargs

    def shuffle(self):
        self._init_dataset()
        self.trunc_pad_anno()
        print('[shuffle the data]')


def build_ego4dgoal_wind_ls(split='train', **kwargs):
    return Ego4DGoalWind_ls(split=split, **kwargs)


def build_concat_train_dataset(train_datasets: list, is_training=True, **kwargs):
    if train_datasets is None or len(train_datasets) == 0:
        return None
    list_datasets = [globals()[f"build_{dataset}"](is_training=is_training, **kwargs) for dataset in train_datasets]
    #return ConcatDataset(list_datasets)
    return list_datasets[0]


####################################### tokenizer
def get_stream_placeholder_jinja2(model_config: LiveLlamaConfig) -> str:
    return f"'{model_config.frame_token_interval}'.join([{model_config.frame_num_tokens} * '{model_config.v_placeholder}'] * message['num_frames'])"


def get_stream_placeholder_len(num_frames: int, model_config: LiveLlamaConfig) -> str:
    return num_frames * model_config.frame_num_tokens * len(model_config.v_placeholder) + len(
        model_config.frame_token_interval) * (num_frames - 1)


def get_stream_learn_ranges(num_frames: int, model_config: LiveLlamaConfig) -> torch.Tensor:
    len_frame_placeholder_with_interval = model_config.frame_num_tokens * len(model_config.v_placeholder) + len(
        model_config.frame_token_interval)
    intermediate_interval_idxs = torch.arange(
        len_frame_placeholder_with_interval,
        len_frame_placeholder_with_interval * num_frames + 1,
        len_frame_placeholder_with_interval
    ) - len(model_config.frame_token_interval)
    len_learn = len(model_config.frame_token_interval) if model_config.frame_token_interval else len(
        model_config.v_placeholder)
    learn_ranges = torch.stack([
        intermediate_interval_idxs,
        intermediate_interval_idxs + len_learn
    ], dim=1)
    return learn_ranges


def chat_template(self, stream_placeholder_jinja2: str, text2visual_connector: str, visual2text_connector: str):
    """
    User: ...
    Str<v><v><v><v><v><v><v>
    'Assistant: aciton </s>' or '~'(no response/background)
    """
    template = (
        "{% if messages[0]['role'] == 'system' %}"
        "{{ bos_token + messages[0]['content'] }}"  # system
        "{% set messages = messages[1:] %}"
        "{% endif %}"
        "{% for message in messages %}"
        "{% if message['role'] == 'assistant' %}"
        "{{ V2T_PLACEHOLDER + message['content'] + eos_token }}"
        "{% elif message['role'] == 'silence' %}"
        "{{ V2TS_PLACEHOLDER + message['content']}}"
        "{% elif message['role'] == 'stream' and message['num_frames'] >= 0: %}"
        "{{ T2V_PLACEHOLDER + STREAM_PLACEHOLDER}}"
        "{% endif %}"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
        "{{ V2TS_PLACEHOLDER}}"
        "{% endif %}"
    )
    template = template.replace('STREAM_PLACEHOLDER', stream_placeholder_jinja2)
    template = template.replace('V2T_PLACEHOLDER', f"'{visual2text_connector}'")
    template = template.replace('V2TS_PLACEHOLDER', f"'{visual2text_connector[:-1]}'")  # remove the last white space' '
    template = template.replace('T2V_PLACEHOLDER', f"'{text2visual_connector}'")
    return template

def chat_template_transition(tokenizer, text2visual_connector, visual2text_connector):
    return {
        (None, 'system'): tokenizer.bos_token,
        ('system', 'stream'): text2visual_connector,
        ('stream', 'assistant'): visual2text_connector,
        ('stream', 'silence'): visual2text_connector[:-1],
        'assistant': visual2text_connector,
        'eos_token': tokenizer.eos_token,
        'silence': visual2text_connector[:-1],
    }


def chat_template_offsets(tokenizer, text2visual_connector, visual2text_connector):
    return {k: len(v) for k, v in chat_template_transition(tokenizer, text2visual_connector, visual2text_connector).items()}


def get_learn_ranges(conversation: list[dict], *, chat_template_offsets: dict[tuple, int],
                     model_config: LiveLlamaConfig):
    offset = 0
    learn_ranges = []
    last_role = None
    for message in conversation:
        role = message['role']
        offset += chat_template_offsets[(last_role, role)]

        last_role = role
        if role == 'stream':
            if message.get('learn', False):
                ValueError('Not implemented learnable stream')
            offset += get_stream_placeholder_len(message['num_frames'], model_config)
        else:
            if role == 'assistant':
                if message.get('learn', False):
                    learn_ranges.append(range(offset - chat_template_offsets['assistant'],
                                              offset + len(message['content']) + chat_template_offsets['eos_token']))
            if role == 'silence':
                learn_ranges.append(range(offset - chat_template_offsets['silence'], offset + len(message['content'])))
            offset += len(message['content'])
    return learn_ranges


##################################### model

class LiveMixin(AutoModelForCausalLM):

    def visual_embed(self, rgb_frames: torch.Tensor):
        frames = rgb_frames
        frames = frames.to(self.dtype)
        frames = self.connector(frames)
        return frames

    def joint_embed(
            self,
            input_ids: torch.Tensor = None,
            rgb_frames: torch.Tensor = None,
            num_frames: torch.Tensor = None,
    ):
        if rgb_frames is None:
            return self.get_input_embeddings()(input_ids)
        inputs_embeds = self.get_input_embeddings()(input_ids.clamp(max=self.vocab_size - 1))
        rgb_frames = self.visual_embed(rgb_frames)
        for bz in range(rgb_frames.shape[0]):
            v_mask = input_ids[bz, :] == self.config.v_placeholder_id
            if v_mask.any():
                inputs_embeds[bz, v_mask, :] = rgb_frames[bz, :num_frames[bz],:]
                assert torch.sum(v_mask) == num_frames[bz]

        return inputs_embeds

    def trim_past_key_values(self, past_key_values, start, stop):
        return [[past_keys[:, :, start:stop], past_values[:, :, start:stop]] for past_keys, past_values in
                past_key_values]


class LiveLlamaForCausalLM(LlamaForCausalLM, LiveMixin):
    config_class = LiveLlamaConfig
    _keys_to_ignore_on_load_missing = ['vision_encoder', 'connector']

    def __init__(self, config: LiveLlamaConfig):
        super().__init__(config)

        self.connector = torch.nn.Sequential(
            torch.nn.Linear(config.visual_dim, config.hidden_size, bias=True),
            GELUActivation(config.hidden_size),
            torch.nn.Linear(config.hidden_size, config.hidden_size, bias=True),
        )
        if config.dropout > 0:
            self.dropout = nn.Dropout(config.dropout)
        else:
            self.dropout = None

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            rgb_frames: torch.FloatTensor = None,
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
            num_frames: torch.LongTensor = None,
            **kwargs,
    ):
        if inputs_embeds is None:
            inputs_embeds = self.joint_embed(input_ids, rgb_frames, num_frames)
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
            dropout=self.dropout,
            **kwargs,
        )

        loss, bg_loss, fg_loss = None, None, None
        if labels is not None:
            logits = outputs[0]
            v_mask = labels == self.config.silence_token_id
            bg_mask = v_mask.sum(-1).clip(0, 1)
            fg_mask = 1-bg_mask

            if self.weight is not None:
                raise ValueError("WMCE is NOT supported yet!")
            else:
                logits = torch.transpose(logits, 1, 2)
                loss = nn.functional.cross_entropy(logits, labels, reduction='none')
                tmpt_loss = loss.sum(dim=-1) / ((labels >= 0).sum(dim=-1) + 1e-8)
                bg_loss = torch.sum(tmpt_loss*bg_mask)/(torch.sum(bg_mask) + 1e-8)
                fg_loss = torch.sum(tmpt_loss*fg_mask)/(torch.sum(fg_mask) + 1e-8)
                # weight = v_mask * self.config.stream_loss_weight + ~v_mask
                # loss = loss * weight
                # loss = torch.mean(loss.sum(dim=-1) / ((labels >= 0) * weight).sum(dim=-1))
                weight = bg_mask * self.config.stream_loss_weight + (1 - bg_mask)
                tmpt_loss = loss.sum(dim=-1) / (labels >= 0).sum(dim=-1)
                loss = torch.mean(tmpt_loss*weight)

        if not return_dict:
            return (loss, bg_loss, fg_loss) + outputs[1:] if loss is not None else outputs  # , bg_loss, fg_loss

        outputs.loss = loss
        return outputs

    def generate_after_embed(self, input_ids, rgb_frames, **kwargs):
        return super().generate(inputs_embeds=self.joint_embed(input_ids, rgb_frames), **kwargs)


def train(model, tokenizer, args):
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.lora_modules,
        lora_dropout=args.lora_dropout, #0.05,
        task_type="CAUSAL_LM",
        modules_to_save=args.finetune_modules,
        inference_mode=False,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # build dataset
    train_dataset = build_concat_train_dataset(tokenizer=tokenizer, **asdict(args))
    data_collator = get_data_collator(tokenizer=tokenizer, **asdict(args))

    # prepare loss
    if model.base_model.model.config.criterion == 'CE':
        model.base_model.model.weight = None
    else:
        print('Unknown loss function!')
        exit(2)

    args.gradient_checkpointing_kwargs = {'use_reentrant': False}


    trainer = TrainerWithTensorBoard(
        model=model, tokenizer=tokenizer,
        args=args,
        train_dataset=train_dataset,
        callbacks=[DataShuffleCallback],
        # eval_dataset=eval_dataset_dict,
        data_collator=data_collator,
        # compute_metrics=compute_metrics_dict,
    )
    trainer.train()
    trainer.save_model()


###################################### inference
class LiveInfer_no_cache:
    def __init__(self, model, tokenizer, args) -> None:
        self.model, self.tokenizer = model, tokenizer
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)

        # long short division
        self.short_length = args.short_len
        self.long_length = args.long_len
        self.long_sample_rate = args.long_sr
        self.short_sample_rate = args.short_sr

        # visual
        self.hidden_size = self.model.config.hidden_size
        self.frame_fps = args.frame_fps
        self.frame_interval = 1 / self.frame_fps
        self.frame_num_tokens = self.model.config.frame_num_tokens
        self.frame_v_placeholder = self.model.config.v_placeholder * self.frame_num_tokens
        self.frame_token_interval_id = self.model.config.frame_token_interval_id
        self.frame_placeholder_ids = torch.tensor(self.model.config.v_placeholder_id).repeat(
            self.model.config.frame_num_tokens).reshape(1, -1)
        self.frame_v_ids = torch.tensor(self.model.config.v_placeholder_id).reshape(1, -1).to(self.device)
        self.silence_token_id = self.model.config.silence_token_id
        self.silence_threshold = args.silence_threshold
        if self.silence_threshold > 0:
            print('********************* you are use THRESHOLD for background detection*******************'
                  'Simply set silence_threshold = 0 will disable the threshold-based inference model')

        # generation
        self.system_prompt = args.system_prompt
        self.generate_length = args.max_response_length
        self.inplace_output_ids = torch.zeros(1, self.generate_length, dtype=torch.long).to(self.device)
        self.eos_token_id = self.model.config.eos_token_id
        self.rotary_emb = LlamaRotaryEmbedding(config=self.model.config)

        if 'crosstask' in args.train_datasets[0]:
            sys_prompt = CrossTaskWind_ls.sys_message
        elif 'thumos' in args.train_datasets[0]:
            sys_prompt = THUMOWind_ls.sys_message
        elif 'ek100' in args.train_datasets[0]:
            sys_prompt = EK100Wind_ls.sys_message
        elif 'ego4dgoal' in args.train_datasets[0]:
            sys_prompt = Ego4DGoalWind_ls.sys_message
        else:
            raise ValueError("Please provide the correct dataset names!!")

        self._sys_ids = self.tokenizer.apply_chat_template([sys_prompt,
                                                            {"role": "stream", 'num_frames': 0, 'learn': False}],
                                                         return_tensors='pt').to(self.device)

        self._added_generation_ids = self.tokenizer.apply_chat_template([{}], add_generation_prompt=True,
                                                                        return_tensors='pt').to(self.device)

        self._eos_ids = torch.tensor([[self.eos_token_id]]).to(self._added_generation_ids.dtype).to(self.device)

        self.conversations = []
        self.reset()

    def fast_greedy_generate(self, inputs_embeds, past_key_values):
        for i in range(self.inplace_output_ids.size(1)):
            outputs = self.model(inputs_embeds=inputs_embeds, past_key_values=past_key_values, use_cache=True)
            past_key_values = outputs.past_key_values
            new_token_id = outputs.logits[:, -1:].argmax(dim=-1)
            self.inplace_output_ids[:, i] = new_token_id
            if new_token_id == self.eos_token_id:
                break
            inputs_embeds = self.model.get_input_embeddings()(new_token_id)
        return self.inplace_output_ids[:, :i + 1], past_key_values

    def _call_for_response(self, video_time):
        inputs_embeds = self.model.get_input_embeddings()(self.last_ids)
        output_ids, past_key_values = self.fast_greedy_generate(inputs_embeds=inputs_embeds, past_key_values=self.past_key_values)

        out_label = self.tokenizer.decode(torch.cat([self.last_ids, output_ids],  dim=-1)[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        response = f'(Video Time = {video_time}s) Assistant:{out_label}'

        return response, out_label.strip('.').lower().strip()

    def _call_for_streaming(self, ):
        while self.frame_embeds_queue:
            video_time, frame_index, frame_embeds = self.frame_embeds_queue.popleft()

            inputs_embeds = torch.cat([
                self.model.get_input_embeddings()(self._sys_ids).view(1, -1, self.hidden_size),
                frame_embeds.view(1, -1, self.hidden_size),
                self.model.get_input_embeddings()(self._added_generation_ids).view(1, -1, self.hidden_size), ], dim=1)

            # 2. inference
            outputs = self.model(inputs_embeds=inputs_embeds, past_key_values=None, use_cache=True)
            past_key_values = outputs.past_key_values
            self.past_key_values = past_key_values

            # 4. decide whether to call the assistant to respond
            next_score = outputs.logits[:, -1:].softmax(dim=-1)
            if self.silence_threshold > 0:
                if next_score[:, :, self.silence_token_id] < self.silence_threshold:
                    next_score[:, :, self.silence_token_id].zero_()
            self.last_ids = next_score.argmax(dim=-1)
            if self.last_ids != self.silence_token_id:
                return video_time
        return None

    def reset(self, ):
        self.video_time = 0
        self.frame_embeds_queue = collections.deque()
        self.last_frame_idx = -1
        self.rgb_tensor = None
        self.last_ids = torch.tensor([[]], dtype=torch.long).to(self.device)

    def input_video_stream(self, video_time):
        frame_idx = int(video_time * self.frame_fps)
        ##### decide the short, long frame indext - every time a new frame comes, the short-long index changes
        work_end = frame_idx + 1
        work_start = max(0, work_end - self.short_length)
        work_indices = np.arange(work_start, work_end).clip(0)
        self.work_indices = work_indices[::self.short_sample_rate]
        long_start, long_end = work_start - self.long_length, work_start - 1
        self.long_indices = uniform_sampler(long_start, long_end, self.long_sample_rate).clip(0)
        # print(self.work_indices, '\t ', self.long_indices)

        if frame_idx > self.last_frame_idx:
            ranger = np.concatenate((self.long_indices, self.work_indices))
            frames_embeds = self.model.visual_embed(self.rgb_tensor[ranger])
            self.frame_embeds_queue.extend(
                [(r / self.frame_fps, fi, frame_embeds) for r, fi, frame_embeds in
                 zip([frame_idx], [frame_idx], [frames_embeds])])
        self.last_frame_idx = frame_idx
        self.video_time = video_time

    def load_video(self, video, rgb_frames):
        self.rgb_tensor = rgb_frames.to(self.device)
        self.num_video_frames = self.rgb_tensor.size(0)
        self.video_duration = self.rgb_tensor.size(0) / self.frame_fps
        logger.warning(f'{video} -> {self.rgb_tensor.shape}, {self.frame_fps} FPS')

    def __call__(self, ):
        while not self.frame_embeds_queue:
            continue
        video_time = self._call_for_streaming()
        response, out_label = None, 'background'
        if video_time is not None:
            response, out_label = self._call_for_response(video_time)
        return response, out_label

def load_videos_and_gt_thumos(phase='test'):
    root = '../datasets/thumos14'
    video_root = os.path.join(root, '')
    anno_root = os.path.join(root, 'target_perframe')
    with open(f'{root}/data_info.json', 'r') as f:
        data_info = json.load(f)['THUMOS']
    embed_mark, vision_pretrained, flow_pretrained = 'kinetics', 'resnet50', 'bninception'
    embed_dir = f"{video_root}/rgb_{embed_mark}_{vision_pretrained}"
    flow_dir = f"{video_root}/flow_{embed_mark}_{flow_pretrained}"
    frame_fps = data_info['fps']
    session_list = data_info['test_session_set']
    if phase == 'train':
        session_list = data_info['train_session_set']
    classnames = data_info['class_names']
    categories = set()

    video_dict = {}
    for sess in session_list:
        rgb_frames = torch.as_tensor(np.load(os.path.join(embed_dir, sess + '.npy'), mmap_mode='r').astype(np.float32))
        flow_frames = torch.as_tensor(np.load(os.path.join(flow_dir, sess + '.npy'), mmap_mode='r').astype(np.float32))

        # get label
        labels = np.load(os.path.join(anno_root, sess + '.npy'))
        steps = []
        for i in range(len(labels)):
            l = np.where(labels[i, :] == 1)[0]
            if 21 in l:
                l = 'ambiguous'  # if ambiguous in labels, make the label as ambiguous
            else:
                l = THUMOS._clean_step(classnames[l[0]])  # else make the label as the first option
            # l = ', '.join([THUMOS._clean_step(classnames[j]) for j in l])
            steps.append(l)
            categories.add(l)
        assert len(steps) == len(rgb_frames)
        video_dict[sess] = {'rgb': rgb_frames, 'flow': flow_frames, 'label': steps}
    return video_dict, classnames, frame_fps

def load_videos_and_gt_crosstask(phase='test'):
    root = '../datasets/CrossTask'
    video_root = os.path.join(root, '')
    anno_root = os.path.join(root, 'target_perframe')
    with open(f'{root}/data_info.json', 'r') as f:
        data_info = json.load(f)['CrossTask']
    embed_dir = f"{video_root}/vitg14_rgb_fps1_h224_w224"
    frame_fps = data_info['fps']
    session_list = data_info['test_session_set']
    if phase == 'train':
        session_list = data_info['train_session_set']
    session_list = [i for i in session_list if i not in CrossTask.DISCARD]
    classnames = data_info['class_names']
    categories = set()

    video_dict = {}
    for sess in session_list:
        rgb_frames = torch.as_tensor(np.load(os.path.join(embed_dir, sess + '.npy'), mmap_mode='r').astype(np.float32))

        # get label
        labels = np.load(os.path.join(anno_root, sess + '.npy'))
        steps = []
        for i in range(len(labels)):
            l = np.where(labels[i, :] == 1)[0]
            l = CrossTask._clean_step(classnames[l[0]])  # else make the label as the first option
            # l = ', '.join([THUMOS._clean_step(classnames[j]) for j in l])
            steps.append(l)
            categories.add(l)
        assert len(steps) == len(rgb_frames)
        video_dict[sess] = {'rgb': rgb_frames, 'label': steps}
    return video_dict, classnames, frame_fps

def load_videos_and_gt_ek100(phase='test'):
    root = '../datasets/EK100'
    video_root = os.path.join(root, '')
    anno_root = os.path.join(root, 'target_perframe')
    with open(f'../datasets/CrossTask/data_info.json', 'r') as f:
        data_info = json.load(f)['EK100']

    embed_dir = f"{video_root}/rgb_kinetics_bninception"
    frame_fps = data_info['fps']
    session_list = data_info['test_session_set']
    if phase == 'train':
        session_list = data_info['train_session_set']
    classnames = data_info['class_names']
    categories = set()

    video_dict = {}
    for sess in session_list:
        rgb_frames = torch.as_tensor(np.load(os.path.join(embed_dir, sess + '.npy'), mmap_mode='r').astype(np.float32))

        # get label
        labels = np.load(os.path.join(anno_root, sess + '.npy'))
        steps = []
        for i in range(len(labels)):
            l = np.where(labels[i, :] == 1)[0]
            l = EK100._clean_step(classnames[l[0]])  # else make the label as the first option
            steps.append(l)
            categories.add(l)
        assert len(steps) == len(rgb_frames)
        video_dict[sess] = {'rgb': rgb_frames, 'label': steps}
    return video_dict, classnames, frame_fps

def load_videos_and_gt_ego4dgoal(phase='val'):
    root = '../datasets/Ego4D-GoalStep'
    video_root = os.path.join(root, 'features')
    anno_root = os.path.join(root, 'target_perframe')

    with open(f'{root}/ego4d-goal_info.json', 'r') as f:
        data_info = json.load(f)

    embed_dir = f"{video_root}/dinov2-giant"
    frame_fps = data_info['fps']
    session_list = data_info['test_session_set']
    if phase == 'train':
        session_list = data_info['train_session_set']
    elif phase == 'val':
        session_list = data_info['val_session_set']
    classnames = data_info['class_names']
    categories = set()

    video_dict = {}
    for sess in session_list:
        rgb_frames = torch.load(os.path.join(embed_dir, sess + '.pt')).float()

        # get label
        labels = np.load(os.path.join(anno_root, sess + '.npy'))
        steps = []
        for i in range(len(labels)):
            l = np.where(labels[i, :] == 1)[0]
            l = Ego4DGoal._clean_step(classnames[l[0]])  # else make the label as the first option
            steps.append(l)
            categories.add(l)
        assert len(steps) == len(rgb_frames)
        video_dict[sess] = {'rgb': rgb_frames, 'label': steps}
    return video_dict, classnames, frame_fps


def evalutate(model, tokenizer, args):
    # args.resume_from_checkpoint = 'outputs/thumos/live1'
    if args.resume_from_checkpoint:
        model = PeftModel.from_pretrained(model, args.resume_from_checkpoint, is_trainable=False)
    else:
        logger.warning(f'!!! Fail to load checkpoint: {args.resume_from_checkpoint}. Return a new initialized model.')
    model.requires_grad_(False)

    # a = dict(model.named_parameters())

    if 'thumos' in args.train_datasets[0]:
        video_dict, classnames, frame_fps = load_videos_and_gt_thumos(args.test_set)
        choices = [THUMOS._clean_step(step) for step in classnames]
        args.frame_fps = frame_fps
    elif 'crosstask' in args.train_datasets[0]:
        video_dict, classnames, frame_fps = load_videos_and_gt_crosstask(args.test_set)
        choices = [CrossTask._clean_step(step) for step in classnames]
        args.frame_fps = frame_fps
    elif 'ek100' in args.train_datasets[0]:
        video_dict, classnames, frame_fps = load_videos_and_gt_ek100(args.test_set)
        choices = [EK100._clean_step(step) for step in classnames]
        args.frame_fps = frame_fps
    elif 'ego4dgoal' in args.train_datasets[0]:
        video_dict, classnames, frame_fps = load_videos_and_gt_ego4dgoal(args.test_set)
        choices = [Ego4DGoal._clean_step(step) for step in classnames]
        args.frame_fps = frame_fps
    else:
        raise ValueError("dataset is not supported")

    liveinfer = LiveInfer_no_cache(model, tokenizer, args)

    final_pred, final_gt = {}, {}
    overall_start, overall_frame = time.time(), 0
    memory_usage, peak_memory_usage, to_show = [], [], []
    for j, sess in enumerate(video_dict.keys()):
        final_pred[sess], final_gt[sess] = [], []
        rgb, label = video_dict[sess]['rgb'], video_dict[sess]['label']
        liveinfer.load_video(sess, rgb)
        pbar = tqdm.tqdm(total=len(rgb), bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}{postfix}]")
        history = {'video_uid': sess, 'length': len(rgb), 'conversation': []}
        vid_result, timecosts = [], []
        overall_frame += len(rgb)
        torch.cuda.reset_peak_memory_stats()
        vid_memory_usage, vid_peak_memory = [], []
        for i in range(len(rgb)):
            start_time = time.time()
            liveinfer.input_video_stream(i / liveinfer.frame_fps)
            response, out_label = liveinfer()
            current_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # in MB
            vid_memory_usage.append(current_memory / 1024)  # convert MB to GB
            vid_peak_memory.append(torch.cuda.max_memory_allocated() / (1024 ** 3))  # convert MB to GB
            end_time = time.time()
            timecosts.append(end_time - start_time)
            fps = (i + 1) / sum(timecosts)
            pbar.set_postfix_str(f"Average Processing FPS: {fps:.1f}")
            pbar.update(1)
            if response:
                history['conversation'].append(
                    {'role': 'assistant', 'content': response, 'time': liveinfer.video_time, 'fps': fps,
                     'cost': timecosts[-1]})
                print(response, label[i])
            if not response:
                history['conversation'].append({'time': liveinfer.video_time, 'fps': fps, 'cost': timecosts[-1]})

            # get the fuzzy match
            if 'thumos' in args.train_datasets[0]: # remove the ambitious
                match_score = [(Levenshtein.distance(out_label, choice), choice) for choice in choices[:-1]]
            else:
                match_score = [(Levenshtein.distance(out_label, choice), choice) for choice in choices]
            match_pred = min(match_score)[1]
            if out_label != match_pred:
                to_show.append([sess, out_label, match_pred])

            vid_result.append([response, out_label, match_pred, label[i]])
            final_gt[sess].append(label[i])
            final_pred[sess].append(match_pred)

        memory_usage.append(np.mean(np.array(vid_memory_usage)))
        peak_memory_usage.append(np.max(np.array(vid_peak_memory)))
        final_pred[sess], final_gt[sess] = np.array(final_pred[sess]), np.array(final_gt[sess])

        # write results
        pred_path = args.resume_from_checkpoint + '/pred{}{}'.format(args.silence_threshold, '_pp' + str(args.post_threshold) if args.post_process else '')

        os.makedirs(pred_path, exist_ok=True)
        with open(os.path.join(pred_path, sess + '.txt'), 'w') as f:
            f.write('%-60s %-20s %-20s %-20s\n' % ('Response', 'Prediction', 'Matched', 'GT'))
            for i1, j1, k1, s1 in vid_result:
                f.write('%-60s \t %-20s \t %-20s \t %-20s\n' % (i1, j1, k1, s1))

        # reset the liveinfer for the next video
        liveinfer.reset()
        # if j > 1: break

    overall_fps = overall_frame / (time.time() - overall_start)
    overall_memory, overall_peak_memory = np.mean(np.array(memory_usage)), np.max(np.array(peak_memory_usage))

    ###
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    print('Number of parameters: {}. Model Size: {:.2f} MB'.format(sum(p.numel() for p in model.parameters()),
                                                                   param_size / 1024 ** 2))
    model_size = param_size / 1024 ** 3
    extra_dict = {'FPS': overall_fps, 'Model size(GB)': model_size, 'Avg Memory(GB)': overall_memory,
                  'Peak Memory(GB)': overall_peak_memory, }

    if 'thumos' in args.train_datasets[0]:
        thumos_results_new(args.resume_from_checkpoint + '_{}{}{}'.format(args.test_set, args.silence_threshold, '_pp' + str(args.post_threshold) if args.post_process else ''),
                        final_pred, final_gt, choices, bg='background', ignore='ambiguous', extra_dict=extra_dict)
    elif 'crosstask' in args.train_datasets[0]:
        crosstask_results_new(args.resume_from_checkpoint + '_{}{}{}'.format(args.test_set, args.silence_threshold, '_pp' + str(args.post_threshold) if args.post_process else ''),
                       final_pred, final_gt, choices, bg='background', extra_dict=extra_dict)
    
    elif 'ek100' in args.train_datasets[0]:
        ek100_results_new(args.resume_from_checkpoint + '_{}{}{}'.format(args.test_set, args.silence_threshold, '_pp' + str(args.post_threshold) if args.post_process else ''), final_pred, final_gt, choices,
                          bg='background', extra_dict=extra_dict)
    elif 'ego4dgoal' in args.train_datasets[0]:
        ego4dgoal_results_new(args.resume_from_checkpoint + '_{}{}{}'.format(args.test_set, args.silence_threshold, '_pp' + str(args.post_threshold) if args.post_process else ''),
                          final_pred, final_gt, choices, bg='background', extra_dict=extra_dict)
    
    
    if len(to_show) > 0:
        print(to_show)
    else:
        print('all matched')

if __name__ == '__main__':
    args, = HfArgumentParser(LiveOneTHUMOSTrainingArguments).parse_args_into_dataclasses()

    ## build model
    config_class, model_class = LiveLlamaConfig, LiveLlamaForCausalLM
    config_class = config_class.from_pretrained(args.llm_pretrained, **asdict(args))
    config_class._is_training = True if not args.test else False
    config_class.visual_dim = args.visual_dim
    config_class.dropout = args.head_dropout
    model = model_class.from_pretrained(args.llm_pretrained, config=config_class,
                                        torch_dtype='auto', attn_implementation=args.attn_implementation)
    ## build tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.llm_pretrained, use_fast=True, padding_side='left')
    tokenizer.add_special_tokens({'additional_special_tokens': [model.config.v_placeholder]})
    v_placeholder_id = len(tokenizer) - 1

    assert model.config.silence_token, "You MUST provide a token to represent 'silence'"
    silence_token_id = int(tokenizer(model.config.silence_token, return_offsets_mapping=True, add_special_tokens=False, return_tensors="pt")[
                    'input_ids'].numpy()[0, 0])
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.silence_token = args.silence_token
    model.config.update(dict(v_placeholder_id=v_placeholder_id, eos_token_id=tokenizer.eos_token_id, silence_token_id=silence_token_id))
    tokenizer.chat_template = chat_template(tokenizer, get_stream_placeholder_jinja2(model.config), args.text2visual_connector, args.visual2text_connector)
    tokenizer.get_learn_ranges = partial(get_learn_ranges, chat_template_offsets=chat_template_offsets(tokenizer, args.text2visual_connector, args.visual2text_connector),
                                         model_config=model.config)

    if args.test:
        evalutate(model, tokenizer, args)
    else:
        train(model, tokenizer, args)


