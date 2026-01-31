
This repository provides official implementation of:
> **ON Discriminative vs. Generative Classifier: Rethinking MLLMs for Action Understanding (ICLR 2026)**  
>Zhanzhong Pang, Dibyadip Chatterjee, Fadime Sener, Angela Yao

[![ICLR 2026](https://img.shields.io/badge/ICLR-2026-blueviolet?style=flat-square)](https://openreview.net/forum?id=ppceQOZrAX)


### TLDR
This work studies fine-tuning MLLMs for video action understanding, revealing the limitations of generative classifer due to semantic label overlap, and proposing a unified generative-assisted discriminative (GAD) classifier that reconciles generative and discriminative objectives.

### Requirements

Ensure you have Miniconda and Python version >= 3.10 installed, then run:
```sh
conda install -y pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install transformers accelerate deepspeed peft editdistance Levenshtein tensorboard gradio moviepy submitit
```

### Tasks
We investigate temporal action understanding tasks, spanning basic step and task
recognition and step forecasting (Offline) and online action
detection(OAD). Please find the corresponding codes for each task.

### Citation
If you find our work useful, please cite:
```
@inproceedings{gen_disc_gad,
  author       = {Zhanzhong Pang, Dibyadip Chatterjee, Fadime Sener, Angela Yao},
  title        = {ON Discriminative vs. Generative Classifier: Rethinking MLLMs for Action Understanding},
  booktitle    = {ICLR},
  year         = {2026},
}
```

## Acknowledgements

This codebase is built upon [`Videollm-online`](https://github.com/showlab/videollm-online) and  [`MAT`](https://github.com/Echo0125/MAT-Memory-and-Anticipation-Transformer).