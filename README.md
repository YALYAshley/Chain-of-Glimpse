# Chain-of-Glimpse


Chain-of-Glimpse: Search-Guided Progressive Object-Grounded Reasoning for Video Understanding

## 📝 Change Logs
* [2026-04-16]: Create the Projects.
* [TODO] All code will be released.

## 🛠️ Requirements and Installation
Basic Dependencies:
```bash
conda create -n chain-of-glimpse python=3.10
conda activate chain-of-glimpse
pip install -e .
pip install flash-attn --no-build-isolation
pip install deepspeed<=0.16.9
```
if cannot install flash-attn, please link to [Flash-attention](https://github.com/Dao-AILab/flash-attention/releases)

# RL dependencies
```bash
cd src/trainer/rl
pip install -e .
pip install vllm==0.8.2
pip install tensordict==0.6.2
pip install "sglang[all]>=0.4.5.post3"
pip install torch==2.6.0 torchaudio==2.6.0 torchvision==0.21.0
pip install ray==2.44.0
cd ../../..
```

# SFT dependencies
```bash
cd src/trainer/offline
pip install -e ".[torch,metrics]"
```


## 📊 Datasets
We conduct experiments on [MSVD-QA](https://github.com/xudejing/video-question-answering) , [MSRVTT-QA](https://github.com/xudejing/video-question-answering), and [ActivityNet-QA](https://github.com/MILVLG/activitynet-qa) datasets. 
Then extract video frames from each video at 10 fps (https://github.com/boheumd/MA-LMM/blob/main/data/extract_frames.py), based on the annotations of each dataset, object detection with pre-trained RAM++ (https://github.com/IDEA-Research/Grounded-Segment-Anything).
Suppose your data structure is like:
```bash
datasets
│   ├── MSVD
│   |   ├── frames
│   |   ├── videos
|   |   └── annoations
|   |   └── objectdetected
...
```


## 🗝️ Training & Evaluation

**[Training]** 
```bash

```

**[Testing]** 
```bash

```

## Acknowledgements
This project builds upon [grounded-rl](https://github.com/Gabesarch/grounded-rl), [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) and [EasyR1](https://github.com/hiyouga/EasyR1) to enable visually-grounded reinforcement learning. We sincerely thank the authors for developing such high-performance training frameworks.
