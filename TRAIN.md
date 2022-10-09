## Training instructions

This document provides brief usage examples for PPO and BC training.

- For general inforation, please see [`README.md`](README.md)
- For installation instructions, please see [`INSTALL.md`](INSTALL.md)
- For tasks descriptions, please see [`TASKS.md`](TASKS.md)

### Reinforcement Learning with PPO

Train `FrankaPick` from states:

```
python tools/train.py task=FrankaPick
```

Train `FrankaPick` from pixels:

```
python tools/train.py task=FrankaPickPixels
```

Test a policy after N iterations:

```
python tools/train.py test=True headless=False logdir=/path/to/job resume=N
```

### Imitation Learning with BC

TODO
