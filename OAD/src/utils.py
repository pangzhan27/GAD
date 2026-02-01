import math, random
import numpy as np

class DictWithTo(dict):
    def to(self, *args, **kwargs):
        return self

def rand_bool():
    return bool(random.getrandbits(1))

def round_time_by_fps(time: float, fps: int, min_time: float, max_time: float):
    return min(max(round(time * fps) / fps, min_time), max_time)

def ceil_time_by_fps(time: float, fps: int, min_time: float, max_time: float):
    return min(max(math.ceil(time * fps) / fps, min_time), max_time)

def floor_time_by_fps(time: float, fps: int, min_time: float, max_time: float):
    return min(max(math.floor(time * fps) / fps, min_time), max_time)

def get_trainable_parameters(model):
    r"""
    Returns the number of trainable parameters and the number of all parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        # Due to the design of 4bit linear layers from bitsandbytes
        # one needs to multiply the number of parameters by 2 to get
        # the correct number of parameters
        if param.__class__.__name__ == "Params4bit":
            if hasattr(param, "element_size"):
                num_bytes = param.element_size()
            elif not hasattr(param, "quant_storage"):
                num_bytes = 1
            else:
                num_bytes = param.quant_storage.itemsize
            num_params = num_params * 2 * num_bytes

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params

    print(f"trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param:.4f}")

def get_labels_start_end_time(frame_wise_labels, bg_class=[0]):
    labels = []
    starts = []
    ends = []
    last_label = frame_wise_labels[0]
    if frame_wise_labels[0] not in bg_class:
        labels.append(frame_wise_labels[0])
        starts.append(0)
    for i in range(len(frame_wise_labels)):
        if frame_wise_labels[i] != last_label:
            if frame_wise_labels[i] not in bg_class:
                labels.append(frame_wise_labels[i])
                starts.append(i)
            if last_label not in bg_class:
                ends.append(i)
            last_label = frame_wise_labels[i]
    if last_label not in bg_class:
        ends.append(i+1)
    return labels, starts, ends

def uniform_sampler(start, end,  sample_rate):
    if start < 0:
        start = (end + 1) % sample_rate
    indices = np.arange(start, end + 1)[::sample_rate]
    return indices.astype(np.int32)

def uniform_sampler_fix_len(start, end, num_samples, sample_rate):
    if start < 0:
        start = (end + 1) % sample_rate
    indices = np.arange(start, end + 1)[::sample_rate]
    padding = num_samples - indices.shape[0]
    if padding > 0:
        indices = np.concatenate((np.zeros(padding), indices))
    return np.sort(indices).astype(np.int32)

def bisect_right(a, x, lo=0, hi=None):
    """Return the index where to insert item x in list a, assuming a is sorted.

    The return value i is such that all e in a[:i] have e <= x, and all e in
    a[i:] have e > x.  So if x already appears in the list, a.insert(x) will
    insert just after the rightmost x already there.

    Optional args lo (default 0) and hi (default len(a)) bound the
    slice of a to be searched.
    """

    if lo < 0:
        raise ValueError('lo must be non-negative')
    if hi is None:
        hi = len(a)
    while lo < hi:
        mid = (lo+hi)//2
        if x < a[mid]: hi = mid
        else: lo = mid+1
    return lo

def check_tokenization(step_categories,tokenizer):
    for i, k in enumerate(sorted(step_categories)):
        input_ids = tokenizer(' ' + k, return_offsets_mapping=True, add_special_tokens=False, return_tensors="pt",
                                   padding=False)
        tokens = []
        for j in input_ids.input_ids[0]:
            tmpt = tokenizer.decode(j, skip_special_tokens=False, clean_up_tokenization_spaces=True)
            tokens.append(tmpt)
        print(i, k, tokens)
