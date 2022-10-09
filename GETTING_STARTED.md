## Getting Started

This document provides brief installation and training instructions.

- For general information, please see [`README.md`](README.md)
- For tasks descriptions, please see [`TASKS.md`](TASKS.md)

**Notes:**

- The code has been tested with PyTorch 1.10, CUDA 11.3 and cuDNN 8.2
- All experiments in the [initial paper](https://arxiv.org/abs/2203.06173) were performed using IsaacGym **Preview 2**
- The code should be compatible with IsaacGym Preview 3/4 (not tested extensively)

### Installation instructions

Create a conda environment:

```
conda create -n mvp python=3.7
conda activate mvp
```

Install [PyTorch](https://pytorch.org/get-started/locally/):

```
conda install pytorch torchvision -c pytorch
```

For RL experiments, install [IsaacGym](https://developer.nvidia.com/isaac-gym):

```
cd /path/to/isaac-gym/python
pip install -e .
```

Clone this repo:

```
cd /path/to/code
git clone git@github.com:ir413/mvp.git
```

Install Python dependencies:

```
cd /path/to/code/mvp
pip install -r requirements.txt
```

Install this repo:

```
cd /path/to/code/mvp
pip install -e .
```

### Reinforcement Learning with PPO

Train `FrankaPick` from states:

```
python tools/train_ppo.py task=FrankaPick
```

Train `FrankaPick` from pixels:

```
python tools/train_ppo.py task=FrankaPickPixels
```

Test a policy after N iterations:

```
python tools/train_ppo.py test=True headless=False logdir=/path/to/job resume=N
```

### Imitation Learning with BC

TODO
