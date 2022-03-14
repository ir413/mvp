## Installation instructions

**Notes:**

- The code has been tested with PyTorch 1.10, CUDA 11.3 and cuDNN 8.2
- All experiments were performed using IsaacGym **Preview 2**
- The code has *not* been tested using IsaacGym **Preview 3**

Create a conda environment:

```
conda create -n mvp python=3.7
conda activate mvp
```

Install [PyTorch](https://pytorch.org/get-started/locally/):

```
conda install pytorch torchvision -c pytorch
```

Install IsaacGym (download [here](https://developer.nvidia.com/isaac-gym)):

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

Please see [`README.md`](README.md) for example training commands.
