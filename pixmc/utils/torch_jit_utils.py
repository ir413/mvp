#!/usr/bin/env python3

"""Torch jit utils."""

import torch
import numpy as np
from isaacgym.torch_utils import *


@torch.jit.script
def quat_axis(q, axis=0):
    # type: (Tensor, int) -> Tensor
    basis_vec = torch.zeros(q.shape[0], 3, device=q.device)
    basis_vec[:, axis] = 1
    return quat_rotate(q, basis_vec)
